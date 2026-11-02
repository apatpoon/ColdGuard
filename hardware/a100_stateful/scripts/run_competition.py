from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import (
    MultiGpuSampler,
    artifacts_manifest,
    atomic_json,
    gpu_summary,
    health_ok,
    memory_map,
    memory_returned,
    process_tree_snapshot,
    read_json,
    stream_chat,
    terminate_process_group,
    utc_now,
    wait_for_idle,
    warm_model_files,
)


STOP_EVENT = threading.Event()


def request_stop(signum: int, _frame: Any) -> None:
    print(json.dumps({"stop_requested": True, "signal": signum, "utc": utc_now()}), flush=True)
    STOP_EVENT.set()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_state(result_dir: Path, **fields: Any) -> None:
    atomic_json(result_dir / "current_state.json", {"updated_utc": utc_now(), **fields})


def write_gpu_state(result_dir: Path, gpu: int, **fields: Any) -> None:
    atomic_json(result_dir / f"current_state_gpu{gpu}.json", {"updated_utc": utc_now(), "physical_gpu": gpu, **fields})


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Service:
    logical_id: str
    process: subprocess.Popen[Any]
    stdout_handle: Any
    stderr_handle: Any
    started_s: float
    ready_s: float
    attempt_dir: Path


class StatefulReplay:
    def __init__(self, config: dict[str, Any], package_root: Path, run_dir: Path, gpu: int, policy: str, events: list[dict[str, Any]], risks: dict[tuple[int, str], dict[str, float]]):
        self.config = config
        self.package_root = package_root
        self.run_dir = run_dir
        self.gpu = gpu
        self.policy = policy
        self.events = events
        self.risks = risks
        self.resident: dict[str, Service] = {}
        self.last_use: dict[str, tuple[int, int, int]] = {}
        self.ever_loaded: set[str] = set()
        self.load_generation: dict[str, int] = {}
        self.event_log = run_dir / "events.jsonl"
        self.eviction_log = run_dir / "evictions.jsonl"
        self.model_path = Path(config["model_root"]) / config["model"]["directory"]

    def port(self, logical_id: str) -> int:
        logical_index = self.config["logical_models"].index(logical_id)
        return (
            int(self.config["port_base"])
            + self.gpu * int(self.config["port_stride_per_gpu"])
            + logical_index
            + 1
        )

    def server_argv(self, logical_id: str) -> list[str]:
        model = self.config["model"]
        server = self.config["server"]
        return [
            "vllm", "serve", str(self.model_path),
            "--host", str(server["host"]),
            "--port", str(self.port(logical_id)),
            "--served-model-name", logical_id,
            "--dtype", str(model["dtype"]),
            "--tensor-parallel-size", "1",
            "--max-model-len", str(model["max_model_len"]),
            "--gpu-memory-utilization", str(model["gpu_memory_utilization"]),
            "--max-num-seqs", str(model["max_num_seqs"]),
            "--generation-config", "vllm",
            "--disable-log-stats",
            "--enforce-eager",
        ]

    def server_env(self, logical_id: str) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(self.gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "COLDGUARD_LOGICAL_MODEL": logical_id,
        })
        return env

    def start_service(self, logical_id: str) -> tuple[Service, float]:
        generation = self.load_generation.get(logical_id, 0) + 1
        self.load_generation[logical_id] = generation
        attempt_dir = self.run_dir / "services" / logical_id / f"load_{generation:03d}"
        attempt_dir.mkdir(parents=True)
        argv = self.server_argv(logical_id)
        atomic_json(attempt_dir / "argv.json", argv)
        atomic_json(attempt_dir / "environment.json", {"CUDA_VISIBLE_DEVICES": str(self.gpu), "logical_model_id": logical_id})
        stdout_handle = (attempt_dir / "stdout.log").open("w", encoding="utf-8")
        stderr_handle = (attempt_dir / "stderr.log").open("w", encoding="utf-8")
        started = time.monotonic()
        process = subprocess.Popen(argv, stdout=stdout_handle, stderr=stderr_handle, env=self.server_env(logical_id), start_new_session=True)
        atomic_json(attempt_dir / "process_initial.json", process_tree_snapshot(process.pid))
        url = f"http://{self.config['server']['host']}:{self.port(logical_id)}/health"
        deadline = started + float(self.config["server"]["startup_timeout_s"])
        ready = None
        while time.monotonic() < deadline:
            if STOP_EVENT.is_set():
                terminate_process_group(process, float(self.config["server"]["shutdown_timeout_s"]))
                stdout_handle.close()
                stderr_handle.close()
                raise InterruptedError(f"stop requested while starting {logical_id}")
            if process.poll() is not None:
                break
            if health_ok(url):
                ready = time.monotonic()
                break
            time.sleep(0.2)
        if ready is None:
            if process.poll() is None:
                terminate_process_group(process, float(self.config["server"]["shutdown_timeout_s"]))
            stdout_handle.close()
            stderr_handle.close()
            raise RuntimeError(f"service {logical_id} failed readiness; rc={process.poll()}")
        service = Service(logical_id, process, stdout_handle, stderr_handle, started, ready, attempt_dir)
        atomic_json(attempt_dir / "process_ready.json", process_tree_snapshot(process.pid))
        self.resident[logical_id] = service
        self.ever_loaded.add(logical_id)
        return service, ready - started

    def stop_service(self, logical_id: str, reason: str) -> dict[str, Any]:
        service = self.resident.pop(logical_id)
        atomic_json(service.attempt_dir / "process_before_shutdown.json", process_tree_snapshot(service.process.pid))
        started = time.monotonic()
        shutdown = terminate_process_group(service.process, float(self.config["server"]["shutdown_timeout_s"]))
        shutdown["elapsed_s"] = time.monotonic() - started
        shutdown["reason"] = reason
        service.stdout_handle.close()
        service.stderr_handle.close()
        atomic_json(service.attempt_dir / "shutdown.json", shutdown)
        return shutdown

    def score(self, day: int, model_id: str, field: str) -> float:
        return float(self.risks.get((day, model_id), {}).get(field, 1.0))

    def next_use(self, current_index: int, model_id: str) -> int:
        for future in range(current_index + 1, len(self.events)):
            if self.events[future]["logical_model_id"] == model_id:
                return future
        return 10**9

    def choose_victim(self, event_index: int, day: int) -> tuple[str, dict[str, Any]]:
        candidates = sorted(self.resident)
        if not candidates:
            raise RuntimeError("capacity eviction requested with no resident model")
        detail: dict[str, Any] = {"candidates": candidates}
        if self.policy == "lru":
            victim = min(candidates, key=lambda model: self.last_use.get(model, (-1, -1, -1)))
        elif self.policy == "ttl_3d":
            idle = {model: day - self.last_use.get(model, (day - 100, 0, 0))[0] for model in candidates}
            expired = [model for model in candidates if idle[model] >= 3]
            pool = expired or candidates
            victim = max(pool, key=lambda model: (idle[model], model))
            detail.update({"idle_days": idle, "ttl_fallback": not bool(expired)})
        elif self.policy in {"coldguard_risk", "direct_risk"}:
            field = "coldguard_risk_3d" if self.policy == "coldguard_risk" else "direct_risk_3d"
            scores = {model: self.score(day, model, field) for model in candidates}
            victim = min(candidates, key=lambda model: (scores[model], self.last_use.get(model, (-1, -1, -1))))
            detail.update({"risk_field": field, "scores": scores})
            if self.policy == "coldguard_risk":
                detail["within_qualification_threshold"] = scores[victim] <= float(self.config["coldguard"]["qualification_threshold"])
        elif self.policy == "oracle":
            next_uses = {model: self.next_use(event_index, model) for model in candidates}
            victim = max(candidates, key=lambda model: (next_uses[model], model))
            detail["next_event_index"] = next_uses
        else:
            raise ValueError(f"unknown policy: {self.policy}")
        return victim, detail

    def request(self, logical_id: str) -> dict[str, Any]:
        return stream_chat(
            f"http://{self.config['server']['host']}:{self.port(logical_id)}/v1/chat/completions",
            logical_id,
            str(self.config["measurement"]["prompt"]),
            int(self.config["measurement"]["max_tokens"]),
        )

    def cleanup(self) -> None:
        for logical_id in list(self.resident):
            try:
                self.stop_service(logical_id, "replay_cleanup")
            except Exception:
                pass

    def run(self) -> dict[str, Any]:
        event_rows = []
        for index, event in enumerate(self.events):
            if STOP_EVENT.is_set():
                raise InterruptedError("stop requested during replay")
            logical_id = event["logical_model_id"]
            day = int(event["day"])
            hit = logical_id in self.resident
            miss_type = "hit"
            startup_s = 0.0
            evicted: list[str] = []
            if not hit:
                miss_type = "eviction_miss" if logical_id in self.ever_loaded else "compulsory_miss"
                while len(self.resident) >= int(self.config["capacity"]["maximum_resident_models"]):
                    victim, detail = self.choose_victim(index, day)
                    shutdown = self.stop_service(victim, f"capacity_for_{logical_id}")
                    eviction = {
                        "event_index": int(event["event_index"]),
                        "day": day,
                        "request_model": logical_id,
                        "victim": victim,
                        "policy": self.policy,
                        "detail": detail,
                        "shutdown": shutdown,
                    }
                    append_jsonl(self.eviction_log, eviction)
                    evicted.append(victim)
                _, startup_s = self.start_service(logical_id)
            request = self.request(logical_id)
            if request.get("status") != "COMPLETED":
                raise RuntimeError(f"request failed for {logical_id}: {request}")
            self.last_use[logical_id] = (day, int(event["position_within_day"]), int(event["event_index"]))
            row = {
                **event,
                "policy": self.policy,
                "physical_gpu": self.gpu,
                "cache_hit": hit,
                "miss_type": miss_type,
                "startup_s": startup_s,
                "ttft_s": request.get("ttft_s"),
                "request_total_s": request.get("total_s"),
                "recovery_to_first_token_s": startup_s + float(request.get("ttft_s") or 0.0),
                "recovery_to_completion_s": startup_s + float(request.get("total_s") or 0.0),
                "evicted_models": evicted,
                "resident_after": sorted(self.resident),
                "request": request,
            }
            append_jsonl(self.event_log, row)
            event_rows.append(row)
        return {"events": event_rows}


def parse_events(rows: list[dict[str, str]], block: int, replicate: int) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if int(row["block"]) != block or int(row["replicate"]) != replicate:
            continue
        output.append({
            "event_index": int(row["event_index"]),
            "day": int(row["day"]),
            "position_within_day": int(row["position_within_day"]),
            "logical_model_id": row["logical_model_id"],
            "block": int(row["block"]),
            "replicate": int(row["replicate"]),
            "request_seed": int(row["request_seed"]),
        })
    return output


def parse_risks(rows: list[dict[str, str]]) -> dict[tuple[int, str], dict[str, float]]:
    return {
        (int(row["day"]), row["logical_model_id"]): {
            "coldguard_risk_3d": float(row["coldguard_risk_3d"]),
            "direct_risk_3d": float(row["direct_risk_3d"]),
        }
        for row in rows
    }


def run_unit(config: dict[str, Any], package_root: Path, result_dir: Path, schedule: dict[str, str], request_rows: list[dict[str, str]], risks: dict[tuple[int, str], dict[str, float]]) -> dict[str, Any]:
    run_id = (
        f"{int(schedule['run_order']):03d}_b{int(schedule['block']):02d}"
        f"_q{int(schedule['parallel_round']):02d}_g{int(schedule['physical_gpu'])}"
        f"_r{int(schedule['replicate'])}_{schedule['policy']}"
    )
    run_dir = result_dir / "policy_runs" / run_id
    completion_path = run_dir / "completion.json"
    if completion_path.is_file():
        prior = read_json(completion_path)
        if prior.get("status") == "COMPLETED" and prior.get("guardrails_passed") is True:
            return {"run_id": run_id, "status": "COMPLETED", "resumed_skip": True}
    if run_dir.exists():
        archive = result_dir / "failed_attempts" / f"{run_id}_{int(time.time())}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        run_dir.replace(archive)
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "schedule.json", schedule)
    block = int(schedule["block"])
    gpu = int(schedule["physical_gpu"])
    replicate = int(schedule["replicate"])
    events = parse_events(request_rows, block, replicate)
    sampler = MultiGpuSampler(run_dir / "telemetry.jsonl", float(config["measurement"]["telemetry_interval_ms"]) / 1000, [gpu])
    baseline_sample = wait_for_idle(sampler, [gpu], float(config["measurement"]["cooldown_temperature_c"]), float(config["measurement"]["cooldown_timeout_s"]), STOP_EVENT)
    baseline = memory_map(baseline_sample)
    replay = StatefulReplay(config, package_root, run_dir, gpu, schedule["policy"], events, risks)
    started = time.monotonic()
    started_utc = utc_now()
    result: dict[str, Any] = {}
    sampler.start()
    try:
        write_gpu_state(result_dir, gpu, phase="policy_replay", run_id=run_id, block=block, replicate=replicate, policy=schedule["policy"])
        outcome = replay.run()
        result.update({"status": "COMPLETED", **outcome})
    except Exception as exc:
        result.update({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        replay.cleanup()
        deadline = time.monotonic() + float(config["server"]["memory_settle_timeout_s"])
        returned = False
        while time.monotonic() < deadline:
            if memory_returned(sampler.samples, baseline, float(config["measurement"]["baseline_tolerance_mib"]), int(config["measurement"]["baseline_consecutive_samples"])):
                returned = True
                break
            time.sleep(max(0.1, sampler.interval_s))
        sampler.stop()
        result["gpu_memory_returned_to_baseline"] = returned
        result["gpu_metrics"] = gpu_summary(sampler.samples, baseline)
        result["telemetry_samples"] = len(sampler.samples)
        result["telemetry_sampling_errors"] = sampler.sampling_errors
        events_done = result.get("events") or []
        result["guardrails_passed"] = bool(
            result.get("status") == "COMPLETED"
            and len(events_done) == len(events)
            and all(row["request"]["status"] == "COMPLETED" for row in events_done)
            and returned
            and not sampler.sampling_errors
        )
        if result.get("status") == "COMPLETED" and not result["guardrails_passed"]:
            result["status"] = "FAILED_GUARDRAIL"
    completion = {
        "schema": "coldguard.a100_stateful_policy_run.v2",
        "run_id": run_id,
        "schedule": schedule,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "wall_s": time.monotonic() - started,
        **result,
    }
    atomic_json(completion_path, completion)
    atomic_json(run_dir / "artifact_manifest.json", artifacts_manifest(run_dir, exclude={"artifact_manifest.json"}))
    return {"run_id": run_id, "status": completion["status"], "resumed_skip": False}


def capacity_pilot_gpu(config: dict[str, Any], package_root: Path, result_dir: Path, risks: dict[tuple[int, str], dict[str, float]], gpu: int) -> dict[str, Any]:
    pilot_dir = result_dir / "capacity_pilot" / f"gpu_{gpu}"
    completion_path = pilot_dir / "completion.json"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    sampler = MultiGpuSampler(pilot_dir / "telemetry.jsonl", float(config["measurement"]["telemetry_interval_ms"]) / 1000, [gpu])
    baseline_sample = wait_for_idle(sampler, [gpu], float(config["measurement"]["cooldown_temperature_c"]), float(config["measurement"]["cooldown_timeout_s"]), STOP_EVENT)
    baseline = memory_map(baseline_sample)
    replay = StatefulReplay(config, package_root, pilot_dir, gpu, "lru", [], risks)
    rows = []
    error = None
    sampler.start()
    try:
        write_gpu_state(result_dir, gpu, phase="capacity_pilot")
        for logical_id in config["logical_models"][:4]:
            if STOP_EVENT.is_set():
                raise InterruptedError("stop requested during capacity pilot")
            _, startup_s = replay.start_service(logical_id)
            request = replay.request(logical_id)
            if request.get("status") != "COMPLETED":
                raise RuntimeError(f"capacity pilot request failed: {logical_id}")
            rows.append({"logical_model_id": logical_id, "startup_s": startup_s, "request": request})
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        replay.cleanup()
        deadline = time.monotonic() + float(config["server"]["memory_settle_timeout_s"])
        returned = False
        while time.monotonic() < deadline:
            if memory_returned(sampler.samples, baseline, float(config["measurement"]["baseline_tolerance_mib"]), int(config["measurement"]["baseline_consecutive_samples"])):
                returned = True
                break
            time.sleep(max(0.1, sampler.interval_s))
        sampler.stop()
    passed = error is None and len(rows) == 4 and returned and not sampler.sampling_errors
    completion = {
        "schema": "coldguard.a100_stateful_capacity_pilot_gpu.v2",
        "physical_gpu": gpu,
        "passed": passed,
        "rows": rows,
        "error": error,
        "memory_returned": returned,
        "telemetry_sampling_errors": sampler.sampling_errors,
        "gpu_metrics": gpu_summary(sampler.samples, baseline),
    }
    atomic_json(completion_path, completion)
    atomic_json(pilot_dir / "artifact_manifest.json", artifacts_manifest(pilot_dir, exclude={"artifact_manifest.json"}))
    return completion


def capacity_pilot(config: dict[str, Any], package_root: Path, result_dir: Path, risks: dict[tuple[int, str], dict[str, float]]) -> bool:
    gpus = [int(value) for value in config["physical_gpus"]]
    pilot_root = result_dir / "capacity_pilot"
    aggregate_path = pilot_root / "completion.json"
    if aggregate_path.is_file() and read_json(aggregate_path).get("passed") is True:
        return True
    if pilot_root.exists():
        archive = result_dir / "failed_capacity_pilots" / f"attempt_{int(time.time())}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        pilot_root.replace(archive)
    write_state(result_dir, phase="capacity_pilot_all_gpus", physical_gpus=gpus)
    with ThreadPoolExecutor(max_workers=len(gpus), thread_name_prefix="capacity-pilot") as pool:
        futures = {
            pool.submit(capacity_pilot_gpu, config, package_root, result_dir, risks, gpu): gpu
            for gpu in gpus
        }
        rows = [future.result() for future in as_completed(futures)]
    rows.sort(key=lambda row: int(row["physical_gpu"]))
    passed = len(rows) == len(gpus) and all(row.get("passed") is True for row in rows)
    atomic_json(result_dir / "capacity_pilot" / "completion.json", {
        "schema": "coldguard.a100_stateful_capacity_pilot.v2",
        "passed": passed,
        "gpu_results": rows,
        "simultaneous_services_planned": len(gpus) * int(config["capacity"]["maximum_resident_models"]),
    })
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if read_json(args.result_dir / "preflight.json").get("passed") is not True:
        raise RuntimeError("preflight did not pass")
    request_rows = load_csv(args.package_root / "workload" / "request_sequence.csv")
    risk_rows = load_csv(args.package_root / "workload" / "risk_scores.csv")
    schedule = load_csv(args.package_root / "workload" / "run_schedule.csv")
    risks = parse_risks(risk_rows)
    warm_evidence = warm_model_files(Path(config["model_root"]) / config["model"]["directory"])
    atomic_json(args.result_dir / "warm_cache_evidence.json", warm_evidence)
    if not capacity_pilot(config, args.package_root, args.result_dir, risks):
        write_state(args.result_dir, phase="capacity_pilot_failed")
        return 3
    outcomes = []
    grouped_schedule: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in schedule:
        key = (int(row["block"]), int(row["parallel_round"]))
        grouped_schedule.setdefault(key, []).append(row)
    for (block, parallel_round), rows in sorted(grouped_schedule.items()):
        if STOP_EVENT.is_set():
            break
        rows.sort(key=lambda row: int(row["physical_gpu"]))
        write_state(
            args.result_dir,
            phase="parallel_policy_round",
            block=block,
            parallel_round=parallel_round,
            assignments=[{"gpu": int(row["physical_gpu"]), "policy": row["policy"], "replicate": int(row["replicate"])} for row in rows],
        )
        with ThreadPoolExecutor(max_workers=len(rows), thread_name_prefix=f"b{block}-q{parallel_round}") as pool:
            futures = [pool.submit(run_unit, config, args.package_root, args.result_dir, row, request_rows, risks) for row in rows]
            round_outcomes = [future.result() for future in as_completed(futures)]
        round_outcomes.sort(key=lambda row: row["run_id"])
        outcomes.extend(round_outcomes)
        atomic_json(args.result_dir / "run_index.json", outcomes)
        for outcome in round_outcomes:
            print(json.dumps(outcome), flush=True)
        if STOP_EVENT.is_set():
            write_state(args.result_dir, phase="stopped_by_signal", block=block, parallel_round=parallel_round)
            break
        failures = [row for row in round_outcomes if row["status"] != "COMPLETED"]
        if len(failures) >= 2:
            write_state(args.result_dir, phase="aborted_after_parallel_round_failures", block=block, parallel_round=parallel_round, failures=failures)
            break
    completed = sum(row["status"] == "COMPLETED" for row in outcomes)
    passed = completed == int(config["guardrails"]["expected_run_count"])
    atomic_json(args.result_dir / "session_completion.json", {
        "schema": "coldguard.a100_stateful_session.v2",
        "completed_utc": utc_now(),
        "passed": passed,
        "planned_runs": int(config["guardrails"]["expected_run_count"]),
        "completed_runs": completed,
        "failed_or_missing": int(config["guardrails"]["expected_run_count"]) - completed,
    })
    write_state(args.result_dir, phase="stateful_replay_complete" if passed else "stateful_replay_incomplete")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
