from __future__ import annotations

import json
from pathlib import Path

import pytest

from coldguard.shadow_gpu_experiment import (
    artifact_hashes,
    confirmed_baseline_time,
    memory_returned_to_baseline,
    randomized_schedule,
    summarize_telemetry,
)


def test_randomized_schedule_is_block_complete_and_reproducible() -> None:
    first = randomized_schedule(["small", "medium", "large"], 1, 3, 20261030)
    second = randomized_schedule(["small", "medium", "large"], 1, 3, 20261030)
    assert first == second
    assert len(first) == 12
    for role, blocks in [("warmup", 1), ("measured", 3)]:
        for block in range(1, blocks + 1):
            members = [
                row["model_id"]
                for row in first
                if row["role"] == role and row["block"] == block
            ]
            assert sorted(members) == ["large", "medium", "small"]


def test_randomized_schedule_rejects_duplicate_models() -> None:
    with pytest.raises(ValueError, match="unique"):
        randomized_schedule(["same", "same"], 0, 2, 1)


def test_memory_return_requires_consecutive_terminal_samples() -> None:
    assert not memory_returned_to_baseline([100, 500, 105, 110], 100, 20, 3)
    assert memory_returned_to_baseline([500, 119, 115, 110], 100, 20, 3)


def test_confirmed_baseline_time_starts_at_stop_request() -> None:
    samples = [
        {"monotonic_s": 1.0, "memory_used_mib": 100},
        {"monotonic_s": 2.0, "memory_used_mib": 6000},
        {"monotonic_s": 3.0, "memory_used_mib": 115},
        {"monotonic_s": 4.0, "memory_used_mib": 110},
        {"monotonic_s": 5.0, "memory_used_mib": 105},
    ]
    confirmed = confirmed_baseline_time(samples, 100, 20, 3, 2.5)
    assert confirmed == pytest.approx(5.0)


def test_telemetry_summary_uses_raw_samples() -> None:
    summary = summarize_telemetry(
        [
            {
                "memory_used_mib": 100,
                "gpu_utilization_pct": 0,
                "temperature_c": 40,
                "power_w": 20,
            },
            {
                "memory_used_mib": 6100,
                "gpu_utilization_pct": 95,
                "temperature_c": 72,
                "power_w": 300,
            },
        ],
        baseline_memory_mib=100,
    )
    assert summary["peak_memory_above_baseline_mib"] == pytest.approx(6000)
    assert summary["peak_gpu_utilization_pct"] == pytest.approx(95)
    assert summary["mean_power_w"] == pytest.approx(160)


def test_artifact_hashes_are_content_bound(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    first = artifact_hashes(tmp_path, ["a.json", "missing.json"])
    (tmp_path / "a.json").write_text(json.dumps({"x": 2}), encoding="utf-8")
    second = artifact_hashes(tmp_path, ["a.json"])
    assert "missing.json" not in first
    assert first["a.json"]["sha256"] != second["a.json"]["sha256"]

