from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


GIB = 1024**3


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    status: str
    returncode: int | None
    stdout: str
    stderr: str


def run_command(argv: Sequence[str], timeout_s: float = 15.0) -> CommandResult:
    executable = shutil.which(argv[0])
    if executable is None:
        return CommandResult(list(argv), "UNAVAILABLE", None, "", "executable not found")
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            list(argv),
            "TIMEOUT",
            None,
            exc.stdout or "",
            exc.stderr or "",
        )
    status = "OK" if completed.returncode == 0 else "ERROR"
    return CommandResult(
        list(argv), status, completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    )


def parse_nvidia_smi_csv(text: str) -> list[dict[str, object]]:
    if not text.strip():
        return []
    rows: list[dict[str, object]] = []
    reader = csv.reader(io.StringIO(text))
    for raw in reader:
        if len(raw) != 7:
            raise ValueError(f"expected 7 nvidia-smi fields, received {len(raw)}")
        index, name, uuid, driver, memory_total, memory_used, temperature = (
            value.strip() for value in raw
        )
        rows.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "driver_version": driver,
                "memory_total_mib": float(memory_total),
                "memory_used_mib": float(memory_used),
                "temperature_c": float(temperature),
            }
        )
    return rows


def linux_memory() -> dict[str, float] | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        amount_kib = float(raw.strip().split()[0])
        values[key] = amount_kib * 1024 / GIB
    if "MemTotal" not in values:
        return None
    return {
        "total_gib": values["MemTotal"],
        "available_gib": values.get("MemAvailable", values.get("MemFree", 0.0)),
    }


def optional_package_versions() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for package_name, import_name in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("nvidia-ml-py", "pynvml"),
    ]:
        try:
            module = __import__(import_name)
        except Exception as exc:  # environment audit must preserve import failures
            result[package_name] = {
                "status": "UNAVAILABLE",
                "version": None,
                "detail": f"{type(exc).__name__}: {exc}",
            }
            continue
        result[package_name] = {
            "status": "AVAILABLE",
            "version": getattr(module, "__version__", "UNKNOWN"),
            "detail": None,
        }
    return result


def torch_cuda_snapshot() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    snapshot: dict[str, object] = {
        "status": "AVAILABLE",
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not torch.cuda.is_available():
        return snapshot
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_gib": total_bytes / GIB,
                "free_gib": free_bytes / GIB,
            }
        )
    snapshot["devices"] = devices
    return snapshot


def assess_readiness(
    gpus: list[dict[str, object]],
    memory: dict[str, float] | None,
    disk_free_gib: float,
    torch_snapshot: dict[str, object],
) -> dict[str, object]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not gpus:
        blockers.append("nvidia-smi did not expose an NVIDIA GPU")
    else:
        primary = gpus[0]
        if float(primary["memory_total_mib"]) < 23_000:
            blockers.append("primary GPU exposes less than 23,000 MiB")
        if float(primary["memory_used_mib"]) > 2_048:
            warnings.append("more than 2 GiB of GPU memory is already in use")
        if "3090" not in str(primary["name"]).lower():
            warnings.append("primary GPU is not identified as an RTX 3090; record the actual model")
    if platform.system() != "Linux":
        warnings.append("Ubuntu/Linux is the reference environment; other systems need a separate path")
    if memory is None:
        warnings.append("host RAM could not be read from /proc/meminfo")
    elif memory["total_gib"] < 48:
        warnings.append("less than 48 GiB host RAM; the 7B profile may require a reduced configuration")
    if disk_free_gib < 80:
        blockers.append("less than 80 GiB free disk space in the selected output path")
    elif disk_free_gib < 150:
        warnings.append("less than 150 GiB free disk space; model cache headroom is limited")
    if torch_snapshot.get("status") != "AVAILABLE":
        warnings.append("PyTorch is not installed yet; this is expected before environment setup")
    elif not torch_snapshot.get("cuda_available", False):
        blockers.append("PyTorch is installed but CUDA is not available")
    return {
        "status": "READY_FOR_SETUP" if not blockers else "NOT_READY",
        "blockers": blockers,
        "warnings": warnings,
    }


def collect_probe(output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_path.parent)
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    smi = run_command(query)
    try:
        gpus = parse_nvidia_smi_csv(smi.stdout) if smi.status == "OK" else []
        parse_error = None
    except ValueError as exc:
        gpus = []
        parse_error = str(exc)
    torch_snapshot = torch_cuda_snapshot()
    memory = linux_memory()
    disk_free_gib = disk.free / GIB
    report = {
        "schema": "coldguard.shadow_gpu_probe.v1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "memory": memory,
            "probe_output_parent": str(output_path.parent.resolve()),
            "disk_free_gib": disk_free_gib,
        },
        "nvidia_smi": asdict(smi),
        "nvidia_smi_parse_error": parse_error,
        "gpus": gpus,
        "nvcc": asdict(run_command(["nvcc", "--version"])),
        "packages": optional_package_versions(),
        "torch_cuda": torch_snapshot,
    }
    report["readiness"] = assess_readiness(gpus, memory, disk_free_gib, torch_snapshot)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect a read-only RTX 3090 readiness report for ColdGuard."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coldguard_shadow_gpu_environment_probe.json"),
    )
    args = parser.parse_args()
    report = collect_probe(args.output)
    print(json.dumps(report["readiness"], indent=2, ensure_ascii=False))
    print(f"report={args.output.resolve()}")
    return 0 if report["readiness"]["status"] == "READY_FOR_SETUP" else 2


if __name__ == "__main__":
    raise SystemExit(main())

