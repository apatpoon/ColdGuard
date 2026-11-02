from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_snapshot(argv: list[str]) -> dict[str, object]:
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def environment_snapshot() -> dict[str, object]:
    return {
        "schema": "coldguard.vllm_hil_environment.v1",
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "vllm_version": command_snapshot(["vllm", "--version"]),
        "nvidia_smi": command_snapshot(["nvidia-smi", "-q"]),
        "lscpu": command_snapshot(["lscpu"]),
        "memory": command_snapshot(["free", "-b"]),
        "disk": command_snapshot(["df", "-B1", str(ROOT)]),
        "pip_freeze": command_snapshot([sys.executable, "-m", "pip", "freeze"]),
    }


class NvmlSampler:
    def __init__(self, output: Path, interval_s: float, gpu_index: int = 0):
        import pynvml

        self.pynvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.output = output
        self.interval_s = interval_s
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> dict[str, object]:
        p = self.pynvml
        memory = p.nvmlDeviceGetMemoryInfo(self.handle)
        utilization = p.nvmlDeviceGetUtilizationRates(self.handle)
        try:
            power_w = p.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except p.NVMLError:
            power_w = None
        processes = []
        try:
            process_rows = p.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except p.NVMLError:
            process_rows = []
        for process in process_rows:
            used = getattr(process, "usedGpuMemory", None)
            processes.append({
                "pid": int(process.pid),
                "used_memory_mib": None if used is None else used / 1024**2,
            })
        return {
            "wall_time_utc": utc_now(),
            "monotonic_s": time.monotonic(),
            "memory_used_mib": memory.used / 1024**2,
            "memory_free_mib": memory.free / 1024**2,
            "gpu_utilization_pct": int(utilization.gpu),
            "memory_utilization_pct": int(utilization.memory),
            "temperature_c": int(p.nvmlDeviceGetTemperature(self.handle, p.NVML_TEMPERATURE_GPU)),
            "power_w": power_w,
            "graphics_clock_mhz": int(p.nvmlDeviceGetClockInfo(self.handle, p.NVML_CLOCK_GRAPHICS)),
            "memory_clock_mhz": int(p.nvmlDeviceGetClockInfo(self.handle, p.NVML_CLOCK_MEM)),
            "compute_processes": processes,
        }

    def _loop(self) -> None:
        with self.output.open("w", encoding="utf-8") as handle:
            while not self._stop.is_set():
                row = self.sample()
                self.samples.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 10))
        self.pynvml.nvmlShutdown()


def load_manifest(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["model_id"]): row for row in payload["models"]}
    if len(rows) != len(payload["models"]):
        raise ValueError("duplicate model_id in manifest")
    return rows


def treatment_schedule(config: dict[str, object], session: int) -> list[dict[str, object]]:
    design = config["design"]
    if session < 1 or session > int(design["sessions"]):
        raise ValueError("session is outside configured range")
    treatments = [
        {"model_id": model, "cache_regime": cache, "pressure_mib": int(pressure)}
        for model in design["factors"]["model_ids"]
        for cache in design["factors"]["cache_regimes"]
        for pressure in design["factors"]["pressure_mib"]
    ]
    schedule: list[dict[str, object]] = []
    sequence = 0
    seed = int(design["seed"]) + session * 100_000
    roles = [
        ("warmup", int(design.get("warmup_blocks_per_session", 0))),
        ("measured", int(design["blocks_per_session"])),
    ]
    for role, number_of_blocks in roles:
        for block in range(1, number_of_blocks + 1):
            order = [dict(row) for row in treatments]
            role_offset = 0 if role == "warmup" else 1_000_000
            block_seed = seed + role_offset + block
            random.Random(block_seed).shuffle(order)
            for position, row in enumerate(order, start=1):
                sequence += 1
                schedule.append({
                    "session": session,
                    "role": role,
                    "block": block,
                    "position_within_block": position,
                    "sequence": sequence,
                    "block_seed": block_seed,
                    **row,
                })
    return schedule


def write_schedule(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wait_for_safe_idle(sampler: NvmlSampler, config: dict[str, object]) -> dict[str, object]:
    measurement = config["measurement"]
    deadline = time.monotonic() + float(measurement["cooldown_timeout_s"])
    last = sampler.sample()
    while time.monotonic() < deadline:
        last = sampler.sample()
        if not last["compute_processes"] and float(last["temperature_c"]) <= float(
            measurement["cooldown_temperature_c"]
        ):
            return last
        time.sleep(1.0)
    raise TimeoutError(f"GPU did not reach safe idle state: {last}")


def warm_page_cache(model_path: Path) -> dict[str, object]:
    started = time.monotonic()
    total = 0
    files = 0
    for path in sorted(model_path.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        files += 1
        with path.open("rb", buffering=0) as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                total += len(chunk)
    return {"mode": "warm", "files": files, "bytes": total, "elapsed_s": time.monotonic() - started}


def drop_page_cache() -> dict[str, object]:
    started = time.monotonic()
    os.sync()
    path = Path("/proc/sys/vm/drop_caches")
    path.write_text("3\n", encoding="utf-8")
    time.sleep(1.0)
    return {"mode": "cold", "drop_caches": True, "elapsed_s": time.monotonic() - started}


def health_ok(url: str, timeout_s: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def stream_chat(url: str, served_model: str, prompt: str, max_tokens: int, timeout_s: float = 300.0) -> dict[str, object]:
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first_content = None
    output_fragments: list[str] = []
    usage: dict[str, object] | None = None
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_content is None:
                        first_content = time.monotonic()
                    output_fragments.append(str(content))
    ended = time.monotonic()
    text = "".join(output_fragments)
    return {
        "status": "COMPLETED",
        "ttft_s": None if first_content is None else first_content - started,
        "total_s": ended - started,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_chars": len(text),
        "usage": usage,
    }


def run_request_batch(
    url: str,
    served_model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
) -> dict[str, object]:
    started = time.monotonic()
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_chat, url, served_model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
    ended = time.monotonic()
    completed = [row for row in results if row["status"] == "COMPLETED"]
    output_tokens = sum(
        int((row.get("usage") or {}).get("completion_tokens", 0)) for row in completed
    )
    return {
        "concurrency": concurrency,
        "wall_s": ended - started,
        "requests": results,
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "aggregate_output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / (ended - started) if output_tokens else None,
    }


def start_pressure(mib: int, run_dir: Path) -> tuple[subprocess.Popen[Any] | None, dict[str, object] | None]:
    if mib == 0:
        return None, None
    ready = run_dir / "pressure_ready.json"
    stop = run_dir / "pressure_stop"
    argv = [
        sys.executable,
        str(ROOT / "scripts" / "vllm_gpu_pressure.py"),
        "--mib",
        str(mib),
        "--ready-file",
        str(ready),
        "--stop-file",
        str(stop),
    ]
    stdout = (run_dir / "pressure_stdout.log").open("w", encoding="utf-8")
    stderr = (run_dir / "pressure_stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and process.poll() is None and not ready.exists():
        time.sleep(0.05)
    stdout.close()
    stderr.close()
    if not ready.exists():
        if process.poll() is None:
            process.kill()
        raise RuntimeError("GPU pressure process failed to become ready")
    return process, json.loads(ready.read_text(encoding="utf-8"))


def stop_pressure(process: subprocess.Popen[Any] | None, run_dir: Path) -> None:
    if process is None:
        return
    (run_dir / "pressure_stop").write_text(utc_now() + "\n", encoding="utf-8")
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def terminate_process_group(process: subprocess.Popen[Any], timeout_s: float) -> dict[str, object]:
    if process.poll() is not None:
        return {"initially_running": False, "returncode": process.returncode}
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
        return {"initially_running": True, "signal": "SIGTERM", "returncode": process.returncode}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
        return {"initially_running": True, "signal": "SIGKILL_AFTER_TIMEOUT", "returncode": process.returncode}


def confirmed_baseline(samples: list[dict[str, object]], baseline_mib: float, tolerance_mib: float, consecutive: int) -> bool:
    if len(samples) < consecutive:
        return False
    threshold = baseline_mib + tolerance_mib
    return all(float(row["memory_used_mib"]) <= threshold for row in samples[-consecutive:])


def energy_j(samples: list[dict[str, object]], start_s: float, end_s: float) -> float | None:
    rows = [
        row for row in samples
        if start_s <= float(row["monotonic_s"]) <= end_s and row.get("power_w") is not None
    ]
    if len(rows) < 2:
        return None
    total = 0.0
    for left, right in zip(rows, rows[1:]):
        dt = float(right["monotonic_s"]) - float(left["monotonic_s"])
        total += dt * (float(left["power_w"]) + float(right["power_w"])) / 2.0
    return total


def run_one(
    row: dict[str, object],
    config: dict[str, object],
    model: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    run_id = (
        f"{int(row['sequence']):03d}_s{int(row['session'])}_{row.get('role', 'trace')}_"
        f"b{int(row['block']):02d}_{row['model_id']}_{row['cache_regime']}_"
        f"p{int(row['pressure_mib']):04d}"
    )
    run_dir = output_root / "runs" / run_id
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        previous = json.loads(completion_path.read_text(encoding="utf-8"))
        return {"run_id": run_id, "status": previous["status"], "resumed_skip": True}
    if run_dir.exists():
        stale = output_root / "incomplete" / f"{run_id}_{int(time.time())}"
        stale.parent.mkdir(parents=True, exist_ok=True)
        run_dir.replace(stale)
    run_dir.mkdir(parents=True)
    write_json(run_dir / "treatment.json", row)
    started_at = utc_now()
    run_started = time.monotonic()
    measurement = config["measurement"]
    server = config["server"]
    sampler = NvmlSampler(
        run_dir / "telemetry.jsonl",
        interval_s=float(measurement["telemetry_interval_ms"]) / 1000.0,
    )
    baseline_sample = wait_for_safe_idle(sampler, config)
    baseline_mib = float(baseline_sample["memory_used_mib"])
    model_path = Path(str(model["local_path"]))
    cache_evidence = (
        warm_page_cache(model_path)
        if row["cache_regime"] == "warm"
        else drop_page_cache()
    )
    write_json(run_dir / "cache_evidence.json", cache_evidence)
    pressure_process = None
    server_process = None
    result: dict[str, object] = {}
    sampler.start()
    try:
        pressure_process, pressure_evidence = start_pressure(int(row["pressure_mib"]), run_dir)
        write_json(run_dir / "pressure_evidence.json", pressure_evidence)
        port = int(server["base_port"]) + int(row["sequence"])
        served_model = str(row["model_id"])
        argv = [
            "vllm",
            "serve",
            str(model_path),
            "--host",
            str(server["host"]),
            "--port",
            str(port),
            "--served-model-name",
            served_model,
            "--dtype",
            str(server["dtype"]),
            "--max-model-len",
            str(server["max_model_len"]),
            "--gpu-memory-utilization",
            str(server["gpu_memory_utilization"]),
            "--generation-config",
            "vllm",
            "--disable-log-stats",
        ]
        write_json(run_dir / "server_argv.json", argv)
        stdout = (run_dir / "server_stdout.log").open("w", encoding="utf-8")
        stderr = (run_dir / "server_stderr.log").open("w", encoding="utf-8")
        server_started = time.monotonic()
        server_process = subprocess.Popen(
            argv,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        health_url = f"http://{server['host']}:{port}/health"
        deadline = server_started + float(server["startup_timeout_s"])
        ready_at = None
        while time.monotonic() < deadline:
            if server_process.poll() is not None:
                break
            if health_ok(health_url):
                ready_at = time.monotonic()
                break
            time.sleep(0.1)
        stdout.close()
        stderr.close()
        if ready_at is None:
            raise RuntimeError(f"vLLM server failed readiness; returncode={server_process.poll()}")

        chat_url = f"http://{server['host']}:{port}/v1/chat/completions"
        warmup = stream_chat(
            chat_url,
            served_model,
            str(measurement["prompt"]),
            int(measurement["max_tokens"]),
        )
        levels = list(map(int, measurement["concurrency_levels"]))
        random.Random(int(row["block_seed"]) + int(row["sequence"])).shuffle(levels)
        batches = []
        request_started = time.monotonic()
        for repetition in range(1, int(measurement["request_repetitions"]) + 1):
            for concurrency in levels:
                batch = run_request_batch(
                    chat_url,
                    served_model,
                    str(measurement["prompt"]),
                    int(measurement["max_tokens"]),
                    concurrency,
                )
                batch["repetition"] = repetition
                batches.append(batch)
        request_ended = time.monotonic()
        result.update({
            "status": "COMPLETED",
            "server_ready_s": ready_at - server_started,
            "warmup": warmup,
            "request_batches": batches,
            "all_requests_completed": all(batch["failed"] == 0 for batch in batches),
            "request_window_s": request_ended - request_started,
            "server_energy_to_ready_j": energy_j(sampler.samples, server_started, ready_at),
            "request_energy_j": energy_j(sampler.samples, request_started, request_ended),
        })
    except Exception as exc:
        result.update({
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        shutdown = None
        if server_process is not None:
            shutdown_started = time.monotonic()
            shutdown = terminate_process_group(server_process, float(server["shutdown_timeout_s"]))
            shutdown["elapsed_s"] = time.monotonic() - shutdown_started
        stop_pressure(pressure_process, run_dir)
        settle_deadline = time.monotonic() + 120.0
        returned = False
        while time.monotonic() < settle_deadline:
            if confirmed_baseline(
                sampler.samples,
                baseline_mib,
                float(measurement["baseline_tolerance_mib"]),
                int(measurement["baseline_consecutive_samples"]),
            ):
                returned = True
                break
            time.sleep(max(0.05, sampler.interval_s))
        sampler.stop()
        result["shutdown"] = shutdown
        result["gpu_memory_returned_to_baseline"] = returned
        result["baseline_memory_mib"] = baseline_mib
        result["peak_memory_mib"] = max(
            (float(sample["memory_used_mib"]) for sample in sampler.samples), default=None
        )
        powers = [float(sample["power_w"]) for sample in sampler.samples if sample.get("power_w") is not None]
        result["max_power_w"] = max(powers) if powers else None
        result["max_temperature_c"] = max(
            (float(sample["temperature_c"]) for sample in sampler.samples), default=None
        )
        result["telemetry_samples"] = len(sampler.samples)
        result["guardrails_passed"] = bool(
            result.get("status") == "COMPLETED"
            and result.get("all_requests_completed")
            and returned
        )
        if result.get("status") == "COMPLETED" and not result["guardrails_passed"]:
            result["status"] = "FAILED_GUARDRAIL"

    completion = {
        "schema": "coldguard.vllm_hil_run.v1",
        "run_id": run_id,
        "treatment": row,
        "model": {
            "model_id": model["model_id"],
            "repository": model["repository"],
            "resolved_revision": model["resolved_revision"],
            "total_bytes": model["total_bytes"],
            "local_path": model["local_path"],
        },
        "started_at": started_at,
        "ended_at": utc_now(),
        "wall_s": time.monotonic() - run_started,
        **result,
    }
    write_json(completion_path, completion)
    manifest_files = [
        path for path in run_dir.iterdir()
        if path.is_file() and path.name not in {"manifest.json"}
    ]
    write_json(run_dir / "manifest.json", {
        "schema": "coldguard.vllm_hil_manifest.v1",
        "run_id": run_id,
        "status": completion["status"],
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(manifest_files)
        },
    })
    return {"run_id": run_id, "status": completion["status"], "resumed_skip": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = load_manifest(args.model_manifest)
    schedule = treatment_schedule(config, args.session)
    required = {str(row["model_id"]) for row in schedule}
    missing = sorted(required - set(models))
    if missing:
        raise ValueError(f"model manifest is missing: {missing}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "runs").mkdir(exist_ok=True)
    shutil.copy2(args.config, args.output_root / "experiment_config.yaml")
    shutil.copy2(args.model_manifest, args.output_root / "model_manifest.json")
    write_schedule(args.output_root / "schedule.csv", schedule)
    environment_path = args.output_root / "environment.json"
    if not environment_path.exists():
        write_json(environment_path, environment_snapshot())
    results = []
    for row in schedule:
        result = run_one(row, config, models[str(row["model_id"])], args.output_root)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        write_json(args.output_root / "run_index.json", results)
    failed = sum(row["status"] != "COMPLETED" for row in results)
    write_json(args.output_root / "session_completion.json", {
        "schema": "coldguard.vllm_hil_session.v1",
        "session": args.session,
        "completed_at": utc_now(),
        "planned_runs": len(schedule),
        "completed_runs": len(schedule) - failed,
        "failed_runs": failed,
        "passed": failed == 0,
    })
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

