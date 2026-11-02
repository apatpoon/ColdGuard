from __future__ import annotations

import pandas as pd
import pytest

from coldguard.simulator import SimulationConfig, simulate_policy


def _actions(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_safe_action_releases_exposure_without_reload() -> None:
    frame = _actions(
        [
            {
                "action_id": "safe",
                "action_day": 1,
                "model_gib": 40.0,
                "exposure_gib_hours": 2_880.0,
                "revives_within_horizon": False,
                "revival_day": float("nan"),
            }
        ]
    )
    result = simulate_policy(frame, SimulationConfig(16.0, 1, seed=7))
    assert result["actions"] == 1
    assert result["reloads"] == 0
    assert result["simulated_released_gib_hours"] == pytest.approx(2_880.0)
    assert result["p95_added_latency_s"] == 0.0


def test_single_revival_matches_loading_equation() -> None:
    frame = _actions(
        [
            {
                "action_id": "revive",
                "action_day": 1,
                "model_gib": 40.0,
                "exposure_gib_hours": 960.0,
                "revives_within_horizon": True,
                "revival_day": 2,
                "revival_time_seconds": 10.0,
            }
        ]
    )
    result = simulate_policy(
        frame,
        SimulationConfig(16.0, 1, fixed_overhead_s=2.0, slo_thresholds_s=(3.0, 5.0)),
    )
    assert result["mean_queue_wait_s"] == 0.0
    assert result["p95_added_latency_s"] == pytest.approx(4.5)
    assert result["reload_slo_violation_rate_3s"] == 1.0
    assert result["reload_slo_violation_rate_5s"] == 0.0


def test_single_loading_slot_creates_fifo_wait() -> None:
    frame = _actions(
        [
            {
                "action_id": action_id,
                "action_day": 1,
                "model_gib": 8.0,
                "exposure_gib_hours": 192.0,
                "revives_within_horizon": True,
                "revival_day": 2,
                "revival_time_seconds": 10.0,
            }
            for action_id in ["a", "b"]
        ]
    )
    result = simulate_policy(
        frame,
        SimulationConfig(8.0, 1, fixed_overhead_s=1.0, slo_thresholds_s=(3.0,)),
    )
    # Each load takes two seconds. The FIFO latencies are two and four seconds.
    assert result["mean_queue_wait_s"] == pytest.approx(1.0)
    assert result["mean_added_latency_s"] == pytest.approx(3.0)
    assert result["p95_added_latency_s"] == pytest.approx(3.9)
    assert result["max_queue_length"] == 1
    assert result["reload_slo_violation_rate_3s"] == 0.5


def test_seeded_within_day_order_is_reproducible() -> None:
    frame = _actions(
        [
            {
                "action_id": f"a{index}",
                "action_day": 1,
                "model_gib": 64.0,
                "exposure_gib_hours": 1_000.0,
                "revives_within_horizon": True,
                "revival_day": 2,
            }
            for index in range(20)
        ]
    )
    config = SimulationConfig(8.0, 1, fixed_overhead_s=2.0, seed=2026)
    first = simulate_policy(frame, config)
    second = simulate_policy(frame, config)
    assert first == second


def test_invalid_duplicate_action_id_is_rejected() -> None:
    frame = _actions(
        [
            {
                "action_id": "duplicate",
                "action_day": 1,
                "model_gib": 1.0,
                "exposure_gib_hours": 1.0,
                "revives_within_horizon": False,
                "revival_day": float("nan"),
            },
            {
                "action_id": "duplicate",
                "action_day": 2,
                "model_gib": 1.0,
                "exposure_gib_hours": 1.0,
                "revives_within_horizon": False,
                "revival_day": float("nan"),
            },
        ]
    )
    with pytest.raises(ValueError, match="action_id"):
        simulate_policy(frame, SimulationConfig(8.0, 1))
