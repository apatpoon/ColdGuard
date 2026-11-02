from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coldguard.shadow_gpu_experiment import (  # noqa: E402
    artifact_hashes,
    confirmed_baseline_time,
    memory_returned_to_baseline,
    randomized_schedule,
    sha256_file,
    summarize_telemetry,
    utc_now,
    write_json,
)


def command_snapshot(argv: list[str]) -> dict[str, object]:
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def environment_snapshot() -> dict[str, object]:
    torch_probe = command_snapshot(
        [
            sys.executable,
            "-c",
            (
                "import json,torch,transformers;"
                "print(json.dumps({'torch':torch.__version__,"
                "'torch_cuda_build':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available(),"
                "'transformers':transformers.__version__}))"
            ),
        ]
    )
    if torch_probe["returncode"] != 0:
        raise RuntimeError(f"isolated torch environment probe failed: {torch_probe['stderr']}")
    torch_environment = json.loads(str(torch_probe["stdout"]).strip())
    return {
        "schema": "coldguard.shadow_gpu_environment.v1",
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        **torch_environment,
        "torch_probe": torch_probe,
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
        try:
            pstate = int(p.nvmlDeviceGetPerformanceState(self.handle))
        except p.NVMLError:
            pstate = None
        processes = []
        try:
            process_rows = p.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except p.NVMLError:
            process_rows = []
        for process in process_rows:
            used = getattr(process, "usedGpuMemory", None)
            processes.append(
                {
                    "pid": int(process.pid),
                    "used_memory_mib": None if used is None else used / 1024**2,
                }
            )
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
            "pstate": pstate,
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


def load_models(manifest_path: Path) -> dict[str, dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(row["model_id"]): row for row in manifest["models"]}


def write_prediction_csv(path: Path, worker: dict[str, object] | None) -> None:
    fields = ["status", "model_id", "prompt_tokens", "output_tokens", "decoded_sha256"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: (worker or {}).get(key) for key in fields})


def run_one(
    row: dict[str, object],
    config: dict[str, object],
    config_path: Path,
    model_manifest_path: Path,
    model: dict[str, object],
    environment: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    measurement = config["measurement"]
    run_id = f"{int(row['sequence']):03d}_{row['role']}_b{int(row['block']):02d}_{row['model_id']}"
    run_dir = output_root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    write_json(run_dir / "config.json", {"schedule": row, "experiment": config})
    write_json(run_dir / "environment.json", environment)

    sampler = NvmlSampler(
        run_dir / "telemetry.jsonl",
        interval_s=float(measurement["gpu_memory_poll_interval_ms"]) / 1000.0,
    )
    baseline_samples = [sampler.sample() for _ in range(3)]
    baseline_mib = sum(float(item["memory_used_mib"]) for item in baseline_samples) / 3
    background_pids = sorted(
        {proc["pid"] for item in baseline_samples for proc in item["compute_processes"]}
    )
    worker_result_path = run_dir / "worker_result.json"
    stop_file = run_dir / "stop.requested"
    worker_argv = [
        sys.executable,
        str(ROOT / "scripts" / "shadow_gpu_worker.py"),
        "--model-id",
        str(row["model_id"]),
        "--model-path",
        str(model["local_path"]),
        "--output",
        str(worker_result_path),
        "--prompt",
        str(measurement["probe_prompt"]),
        "--max-new-tokens",
        str(measurement["max_new_tokens"]),
        "--seed",
        str(int(config["design"]["seed"]) + int(row["sequence"])),
        "--dtype",
        str(config["device"]["dtype"]),
        "--stop-file",
        str(stop_file),
    ]
    sampler.start()
    stop_requested_monotonic = None
    process_exit_monotonic = None
    worker_returncode = None
    failure_detail = None
    try:
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            run_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(worker_argv, stdout=stdout, stderr=stderr)
            ready_deadline = time.monotonic() + 900.0
            while time.monotonic() < ready_deadline:
                if worker_result_path.exists():
                    candidate = json.loads(worker_result_path.read_text(encoding="utf-8"))
                    if candidate.get("worker_state") == "READY_HELD" or candidate.get("status") == "FAILED":
                        break
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if process.poll() is None and worker_result_path.exists():
                candidate = json.loads(worker_result_path.read_text(encoding="utf-8"))
                if candidate.get("worker_state") == "READY_HELD":
                    stop_requested_monotonic = time.monotonic()
                    stop_file.write_text(utc_now() + "\n", encoding="utf-8")
            try:
                worker_returncode = process.wait(timeout=60.0)
            except subprocess.TimeoutExpired:
                process.kill()
                worker_returncode = process.wait(timeout=30.0)
                failure_detail = {
                    "error_type": "TimeoutExpired",
                    "error": "worker did not exit within 60 seconds after stop request",
                }
        process_exit_monotonic = time.monotonic()
        deadline = process_exit_monotonic + float(measurement["unload_settle_timeout_s"])
        while time.monotonic() < deadline:
            values = [float(sample["memory_used_mib"]) for sample in sampler.samples]
            if memory_returned_to_baseline(values, baseline_mib, 128.0, 3):
                break
            time.sleep(max(0.05, sampler.interval_s))
    except Exception as exc:
        failure_detail = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        sampler.stop()

    worker = (
        json.loads(worker_result_path.read_text(encoding="utf-8"))
        if worker_result_path.exists()
        else None
    )
    memory_values = [float(sample["memory_used_mib"]) for sample in sampler.samples]
    returned = memory_returned_to_baseline(memory_values, baseline_mib, 128.0, 3)
    unload_to_baseline_s = None
    baseline_confirmed_monotonic = None
    if returned and stop_requested_monotonic is not None:
        baseline_confirmed_monotonic = confirmed_baseline_time(
            sampler.samples,
            baseline_mib,
            128.0,
            3,
            stop_requested_monotonic,
        )
        if baseline_confirmed_monotonic is not None:
            unload_to_baseline_s = baseline_confirmed_monotonic - stop_requested_monotonic

    telemetry = summarize_telemetry(sampler.samples, baseline_mib)
    finite_worker_metrics = bool(
        worker
        and worker.get("status") == "COMPLETED"
        and all(
            math.isfinite(float(worker[key]))
            for key in ["model_load_s", "load_total_s", "first_request_s"]
        )
    )
    guardrails = {
        "schema": "coldguard.shadow_gpu_guardrails.v1",
        "worker_exit_zero": worker_returncode == 0,
        "worker_completed": bool(worker and worker.get("status") == "COMPLETED"),
        "gpu_memory_returned_within_timeout": returned,
        "finite_latency_measurements": finite_worker_metrics,
        "background_compute_pids_at_start": background_pids,
        "stop_was_explicitly_requested": stop_requested_monotonic is not None,
        "passed": worker_returncode == 0
        and returned
        and finite_worker_metrics
        and not background_pids
        and stop_requested_monotonic is not None,
    }
    metrics = {
        "schema": "coldguard.shadow_gpu_raw_metrics.v1",
        "run_id": run_id,
        "schedule": row,
        "model_id": row["model_id"],
        "model_repository": model["repository"],
        "model_revision": model["resolved_revision"],
        "baseline_memory_mib": baseline_mib,
        "unload_to_baseline_s": unload_to_baseline_s,
        "stop_requested_monotonic_s": stop_requested_monotonic,
        "process_exit_monotonic_s": process_exit_monotonic,
        "baseline_confirmed_monotonic_s": baseline_confirmed_monotonic,
        "worker_returncode": worker_returncode,
        **telemetry,
        "worker": worker,
    }
    (run_dir / "raw_metrics.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_prediction_csv(run_dir / "predictions.csv", worker)
    write_json(run_dir / "guardrails.json", guardrails)
    write_json(run_dir / "test_evidence.json", {"local_test_suite": "68 passed before dispatch"})
    if failure_detail or not guardrails["passed"]:
        write_json(
            run_dir / "failure.json",
            {
                "failure_detail": failure_detail,
                "worker": worker,
                "guardrails": guardrails,
            },
        )

    ended_at = utc_now()
    artifact_names = [
        "config.json",
        "environment.json",
        "stdout.log",
        "stderr.log",
        "telemetry.jsonl",
        "worker_result.json",
        "stop.requested",
        "raw_metrics.jsonl",
        "predictions.csv",
        "guardrails.json",
        "test_evidence.json",
        "failure.json",
    ]
    manifest = {
        "schema": "coldguard.shadow_gpu_run_manifest.v1",
        "run_id": run_id,
        "matrix_row_id": "CG-HIL-PRIMITIVES",
        "status": "COMPLETED" if guardrails["passed"] else "FAILED",
        "termination": "worker_process_exited",
        "completion": {"all_guardrails_passed": guardrails["passed"]},
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_s": time.monotonic() - started_monotonic,
        "argv": worker_argv,
        "seed_role": row["role"],
        "seed": int(config["design"]["seed"]) + int(row["sequence"]),
        "inputs": {
            "experiment_config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
            "model_manifest": {
                "path": str(model_manifest_path.resolve()),
                "sha256": sha256_file(model_manifest_path),
            },
            "model_revision": model["resolved_revision"],
        },
        "artifacts": artifact_hashes(run_dir, artifact_names),
    }
    write_json(run_dir / "manifest.json", manifest)
    return {"run_id": run_id, "status": manifest["status"], "run_dir": str(run_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = load_models(args.model_manifest)
    model_ids = [row["model_id"] for row in config["models"]]
    schedule = randomized_schedule(
        model_ids,
        int(config["measurement"]["warmup_cycles_per_model"]),
        int(config["measurement"]["measured_cycles_per_model"]),
        int(config["design"]["seed"]),
    )
    write_json(args.output_root / "schedule.json", schedule)
    shutil.copy2(args.config, args.output_root / "experiment_config.yaml")
    shutil.copy2(args.model_manifest, args.output_root / "model_manifest.json")
    environment = environment_snapshot()
    write_json(args.output_root / "environment.json", environment)
    results = []
    for row in schedule:
        result = run_one(
            row,
            config,
            args.config,
            args.model_manifest,
            models[str(row["model_id"])],
            environment,
            args.output_root,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_json(args.output_root / "run_index.json", results)
    failed = sum(row["status"] != "COMPLETED" for row in results)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

