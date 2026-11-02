from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from coldguard.shadow_gpu_analysis import (
    fit_empirical_service_curve,
    summarize_measured_cycles,
)


def _cycle(model: str, size: float, block: int, load: float) -> dict[str, object]:
    return {
        "role": "measured",
        "block": block,
        "model_id": model,
        "model_repository": f"repo/{model}",
        "model_revision": "abc",
        "nominal_params_b": size,
        "model_artifact_bytes": int(size * 1024**3),
        "model_artifact_gib": size,
        "load_total_s": load,
        "model_load_s": load - 0.1,
        "first_request_s": load / 2,
        "resident_device_memory_mib": size * 1024,
        "peak_memory_above_baseline_mib": size * 1100,
        "unload_to_baseline_s": 0.7,
    }


def test_summary_uses_measured_cycles_only() -> None:
    rows = [_cycle("small", 1.0, block, 1.0 + 0.01 * block) for block in range(1, 6)]
    warmup = _cycle("small", 1.0, 1, 99.0)
    warmup["role"] = "warmup"
    summary = summarize_measured_cycles(pd.DataFrame(rows + [warmup]))
    assert summary.loc[0, "n_cycles"] == 5
    assert summary.loc[0, "load_total_s_mean"] == pytest.approx(1.03)
    assert summary.loc[0, "load_total_s_ci95_low"] < 1.03
    assert summary.loc[0, "load_total_s_ci95_high"] > 1.03


def test_service_curve_recovers_linear_relation() -> None:
    rows = []
    for model, size in [("small", 1.0), ("large", 4.0)]:
        for block in range(1, 6):
            rows.append(_cycle(model, size, block, 0.5 + 0.25 * size))
    curve = fit_empirical_service_curve(pd.DataFrame(rows))
    assert curve["intercept_s"] == pytest.approx(0.5)
    assert curve["slope_s_per_artifact_gib"] == pytest.approx(0.25)
    assert curve["implied_artifact_gib_per_s"] == pytest.approx(4.0)
    assert curve["r_squared"] == pytest.approx(1.0)
    assert curve["params_intercept_s"] == pytest.approx(0.5)
    assert curve["slope_s_per_billion_params"] == pytest.approx(0.25)
