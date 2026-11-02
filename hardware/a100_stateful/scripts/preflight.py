from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
from collections import Counter
from pathlib import Path

from common import atomic_json, environment_snapshot, utc_now, validate_model


def system_memory_gib() -> float:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def service_port(config: dict, gpu: int, logical_index: int) -> int:
    return int(config["port_base"]) + gpu * int(config["port_stride_per_gpu"]) + logical_index + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    guardrails = config["guardrails"]
    model_root = Path(config["model_root"])

    import pynvml
    import torch
    import vllm

    pynvml.nvmlInit()
    gpu_rows = []
    active_processes = {}
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else str(name)
        memory_gib = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**3
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        gpu_rows.append({"index": index, "name": name, "memory_gib": memory_gib})
        if processes:
            active_processes[str(index)] = [int(row.pid) for row in processes]
    pynvml.nvmlShutdown()

    model_spec = dict(config["model"])
    model_spec["local_path"] = str(model_root / model_spec["directory"])
    model_record = validate_model(model_spec, model_root)
    requests = list(csv.DictReader((args.package_root / "workload" / "request_sequence.csv").open()))
    schedule = list(csv.DictReader((args.package_root / "workload" / "run_schedule.csv").open()))
    risks = list(csv.DictReader((args.package_root / "workload" / "risk_scores.csv").open()))
    disk = shutil.disk_usage(args.result_dir)
    gpus = [int(value) for value in config["physical_gpus"]]
    policies = set(config["policies"])
    request_counts = Counter(int(row["replicate"]) for row in requests)
    gpu_run_counts = Counter(int(row["physical_gpu"]) for row in schedule)
    policy_block_counts = Counter((int(row["block"]), row["policy"]) for row in schedule)
    round_groups: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in schedule:
        round_groups.setdefault((int(row["block"]), int(row["parallel_round"])), []).append(row)
    schedule_balanced = bool(
        len(schedule) == int(guardrails["expected_run_count"])
        and set(gpu_run_counts) == set(gpus)
        and set(gpu_run_counts.values()) == {int(guardrails["expected_runs_per_gpu"])}
        and set(policy_block_counts.values()) == {len(gpus)}
        and all(
            len(rows) == len(gpus)
            and {int(row["physical_gpu"]) for row in rows} == set(gpus)
            and len({row["policy"] for row in rows}) == len(gpus)
            and {row["policy"] for row in rows}.issubset(policies)
            for rows in round_groups.values()
        )
        and all(
            {
                int(row["request_seed"])
                for row in schedule
                if int(row["block"]) == block and row["policy"] == policy
            }
            == set(int(value) for value in config["request_order_seeds"])
            for block in (1, 2, 3)
            for policy in policies
        )
    )
    all_ports = [
        service_port(config, gpu, logical_index)
        for gpu in gpus
        for logical_index, _ in enumerate(config["logical_models"])
    ]
    checks = {
        "gpu_count": len(gpu_rows) >= int(guardrails["required_gpu_count"]),
        "gpu_names": all(guardrails["required_gpu_name_contains"] in gpu_rows[gpu]["name"] for gpu in gpus),
        "gpu_memory": all(gpu_rows[gpu]["memory_gib"] >= float(guardrails["minimum_gpu_memory_gib"]) for gpu in gpus),
        "no_existing_compute_processes": not active_processes,
        "system_memory": system_memory_gib() >= float(guardrails["minimum_system_memory_gib"]),
        "result_disk": disk.free / 1024**3 >= float(guardrails["minimum_result_disk_free_gib"]),
        "model": bool(model_record["valid"]),
        "runtime": vllm.__version__ == "0.32.0" and torch.cuda.device_count() >= len(gpus),
        "request_rows": len(requests) == int(guardrails["expected_request_rows"]),
        "request_replicates": len(request_counts) == len(gpus) and set(request_counts.values()) == {int(guardrails["expected_requests_per_replicate"])},
        "schedule_balanced": schedule_balanced,
        "risk_rows": len(risks) > 0,
        "ports_unique": len(all_ports) == len(set(all_ports)),
        "ports_available": all(port_available(config["server"]["host"], port) for port in all_ports),
        "capacity_design": (
            int(config["capacity"]["maximum_resident_models"])
            * float(config["model"]["planned_reservation_gib"])
            <= float(config["capacity"]["planned_gpu_budget_gib"])
        ),
    }
    passed = all(checks.values())
    args.result_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.result_dir / "preflight.json", {
        "schema": "coldguard.a100_stateful_preflight.v2",
        "captured_utc": utc_now(),
        "passed": passed,
        "checks": checks,
        "gpu_snapshot": gpu_rows,
        "active_compute_processes": active_processes,
        "system_memory_gib": system_memory_gib(),
        "result_disk_free_gib": disk.free / 1024**3,
        "model": model_record,
        "reserved_ports": all_ports,
    })
    atomic_json(args.result_dir / "environment.json", environment_snapshot(args.result_dir))
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
