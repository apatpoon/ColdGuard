from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_vllm_hil_session", ROOT / "scripts" / "run_vllm_hil_session.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

TRACE_SPEC = importlib.util.spec_from_file_location(
    "run_vllm_trace_replay", ROOT / "scripts" / "run_vllm_trace_replay.py"
)
TRACE_MODULE = importlib.util.module_from_spec(TRACE_SPEC)
assert TRACE_SPEC.loader is not None
TRACE_SPEC.loader.exec_module(TRACE_MODULE)


def config() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "config" / "shadow_gpu_3090_vllm_hil.yaml").read_text(encoding="utf-8")
    )


def test_schedule_is_balanced_in_every_block() -> None:
    schedule = MODULE.treatment_schedule(config(), 1)
    assert len(schedule) == 5 * 4 * 2 * 2
    assert sum(row["role"] == "warmup" for row in schedule) == 16
    assert sum(row["role"] == "measured" for row in schedule) == 64
    for role, number_of_blocks in [("warmup", 1), ("measured", 4)]:
        for block in range(1, number_of_blocks + 1):
            rows = [
                row for row in schedule if row["role"] == role and row["block"] == block
            ]
            assert len(rows) == 16
            treatments = {
                (row["model_id"], row["cache_regime"], row["pressure_mib"]) for row in rows
            }
            assert len(treatments) == 16


def test_schedule_is_reproducible_but_session_specific() -> None:
    first = MODULE.treatment_schedule(config(), 1)
    assert first == MODULE.treatment_schedule(config(), 1)
    assert first != MODULE.treatment_schedule(config(), 2)


def test_energy_integral() -> None:
    samples = [
        {"monotonic_s": 0.0, "power_w": 100.0},
        {"monotonic_s": 1.0, "power_w": 200.0},
        {"monotonic_s": 2.0, "power_w": 100.0},
    ]
    assert MODULE.energy_j(samples, 0.0, 2.0) == 300.0


def test_confirmed_baseline_uses_consecutive_tail() -> None:
    samples = [
        {"memory_used_mib": 500.0},
        {"memory_used_mib": 120.0},
        {"memory_used_mib": 121.0},
        {"memory_used_mib": 119.0},
    ]
    assert MODULE.confirmed_baseline(samples, 100.0, 30.0, 3)
    assert not MODULE.confirmed_baseline(samples, 100.0, 10.0, 3)


def test_trace_schedule_crosses_every_event_with_cache_regime() -> None:
    events = [
        {
            "trace_event_id": "primary:1",
            "model_id": "qwen_3b",
            "selected_by_coldguard": True,
            "selected_by_direct": False,
        },
        {
            "trace_event_id": "primary:2",
            "model_id": "qwen_7b",
            "selected_by_coldguard": True,
            "selected_by_direct": True,
        },
    ]
    schedule = TRACE_MODULE.trace_schedule(events, config(), 1)
    assert len(schedule) == 4
    observed = {
        (row["trace_event_id"], row["cache_regime"], row["pressure_mib"])
        for row in schedule
    }
    assert observed == {
        ("primary:1", "warm", 0),
        ("primary:1", "cold", 0),
        ("primary:2", "warm", 0),
        ("primary:2", "cold", 0),
    }


