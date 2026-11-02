from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MIB = 1024**2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * MIB) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def randomized_schedule(
    model_ids: Sequence[str],
    warmup_cycles: int,
    measured_cycles: int,
    seed: int,
) -> list[dict[str, object]]:
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be non-empty and unique")
    if warmup_cycles < 0 or measured_cycles < 1:
        raise ValueError("warmup_cycles must be non-negative and measured_cycles positive")
    schedule: list[dict[str, object]] = []
    sequence = 0
    for role, blocks in [("warmup", warmup_cycles), ("measured", measured_cycles)]:
        for block in range(blocks):
            order = list(model_ids)
            # Independent deterministic block streams avoid dependence on earlier list length.
            block_seed = seed + (0 if role == "warmup" else 1_000_000) + block
            random.Random(block_seed).shuffle(order)
            for position, model_id in enumerate(order):
                sequence += 1
                schedule.append(
                    {
                        "sequence": sequence,
                        "role": role,
                        "block": block + 1,
                        "position_within_block": position + 1,
                        "block_seed": block_seed,
                        "model_id": model_id,
                    }
                )
    return schedule


def summarize_telemetry(
    samples: Iterable[Mapping[str, object]], baseline_memory_mib: float
) -> dict[str, float | int | None]:
    rows = list(samples)
    if not rows:
        return {
            "samples": 0,
            "peak_memory_used_mib": None,
            "peak_memory_above_baseline_mib": None,
            "peak_gpu_utilization_pct": None,
            "max_temperature_c": None,
            "mean_power_w": None,
            "max_power_w": None,
        }

    def finite_values(key: str) -> list[float]:
        values = []
        for row in rows:
            value = row.get(key)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        return values

    memory = finite_values("memory_used_mib")
    utilization = finite_values("gpu_utilization_pct")
    temperature = finite_values("temperature_c")
    power = finite_values("power_w")
    peak_memory = max(memory) if memory else None
    return {
        "samples": len(rows),
        "peak_memory_used_mib": peak_memory,
        "peak_memory_above_baseline_mib": (
            peak_memory - baseline_memory_mib if peak_memory is not None else None
        ),
        "peak_gpu_utilization_pct": max(utilization) if utilization else None,
        "max_temperature_c": max(temperature) if temperature else None,
        "mean_power_w": sum(power) / len(power) if power else None,
        "max_power_w": max(power) if power else None,
    }


def memory_returned_to_baseline(
    memory_samples_mib: Sequence[float],
    baseline_mib: float,
    tolerance_mib: float,
    consecutive_samples: int,
) -> bool:
    if consecutive_samples < 1:
        raise ValueError("consecutive_samples must be positive")
    if len(memory_samples_mib) < consecutive_samples:
        return False
    threshold = baseline_mib + tolerance_mib
    return all(value <= threshold for value in memory_samples_mib[-consecutive_samples:])


def confirmed_baseline_time(
    samples: Sequence[Mapping[str, object]],
    baseline_mib: float,
    tolerance_mib: float,
    consecutive_samples: int,
    not_before_monotonic_s: float,
) -> float | None:
    if consecutive_samples < 1:
        raise ValueError("consecutive_samples must be positive")
    eligible = [
        sample
        for sample in samples
        if float(sample["monotonic_s"]) >= not_before_monotonic_s
    ]
    threshold = baseline_mib + tolerance_mib
    for index in range(0, len(eligible) - consecutive_samples + 1):
        window = eligible[index : index + consecutive_samples]
        if all(float(sample["memory_used_mib"]) <= threshold for sample in window):
            return float(window[-1]["monotonic_s"])
    return None


def artifact_hashes(run_dir: Path, names: Sequence[str]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for name in names:
        path = run_dir / name
        if path.exists() and path.is_file():
            output[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return output

