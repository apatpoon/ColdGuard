from __future__ import annotations

from datetime import timedelta

import pandas as pd

from coldguard.freeinference import _active_mask, assign_splits, temporal_design


def hours(value: int) -> pd.Timedelta:
    return pd.Timedelta(timedelta(hours=value))


def test_active_mask_combines_all_serving_signals() -> None:
    frame = pd.DataFrame(
        {
            "req_per_min": [0, 1, 0, 0, 0, 0],
            "running": [0, 0, 1, 0, 0, 0],
            "queued": [0, 0, 0, 1, 0, 0],
            "gen_throughput": [0, 0, 0, 0, 1, 0],
            "prefill_tps": [0, 0, 0, 0, 0, 1],
        }
    )
    assert _active_mask(frame).tolist() == [False, True, True, True, True, True]


def test_temporal_split_has_maturity_embargo() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    serving = pd.DataFrame(
        {"dt": pd.date_range(start, periods=168, freq="1h", tz="UTC")}
    )
    design = temporal_design(serving, horizon_hours=int(6))
    landmarks = pd.DataFrame(
        {
            "dt": [
                start + hours(66),
                start + hours(67),
                start + hours(73),
                start + hours(114),
                start + hours(115),
                start + hours(121),
            ]
        }
    )
    result = assign_splits(landmarks, design)
    assert result["split"].tolist() == [
        "train",
        "embargo",
        "calibration",
        "calibration",
        "embargo",
        "test",
    ]
