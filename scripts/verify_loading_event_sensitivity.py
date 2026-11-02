from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/loading_event_sensitivity"


def _one(frame: pd.DataFrame, method: str, scenario: str) -> pd.Series:
    rows = frame.loc[
        (frame["cold_definition"] == "sm_le_0.1_pct")
        & (frame["method"] == method)
        & (frame["scenario"] == scenario)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def main() -> None:
    events = pd.read_csv(ARTIFACT / "event_level_transfer_scenarios.csv")
    summary = pd.read_csv(ARTIFACT / "scenario_summary.csv")
    loo = pd.read_csv(ARTIFACT / "leave_one_event_out.csv")
    hazard = _one(summary, "hazard_hierarchical", "allocation_proxy")
    direct = _one(summary, "direct_monotone", "allocation_proxy")
    assert int(hazard.events_total) == 9
    assert int(direct.events_total) == 8
    assert np.isclose(hazard.p95_added_s, 17.3389, atol=5e-4)
    assert np.isclose(direct.p95_added_s, 28.9971, atol=5e-4)
    assert int(hazard.events_over_30s) == 0
    assert int(direct.events_over_30s) == 1

    hazard_fp16 = _one(summary, "hazard_hierarchical", "fp16_checkpoint")
    direct_fp16 = _one(summary, "direct_monotone", "fp16_checkpoint")
    assert int(hazard_fp16.events_with_volume) == 8
    assert int(direct_fp16.events_with_volume) == 8
    assert np.isclose(hazard_fp16.volume_coverage, 8 / 9)
    assert np.isclose(direct_fp16.volume_coverage, 1.0)
    assert np.isclose(hazard_fp16.p95_added_s, 5.6438, atol=5e-4)
    assert np.isclose(direct_fp16.p95_added_s, 9.5088, atol=5e-4)

    hazard_int8 = _one(summary, "hazard_hierarchical", "int8_checkpoint")
    direct_int8 = _one(summary, "direct_monotone", "int8_checkpoint")
    hazard_rank = _one(summary, "hazard_hierarchical", "fp16_per_rank_parallel")
    direct_rank = _one(summary, "direct_monotone", "fp16_per_rank_parallel")
    assert np.isclose(hazard_int8.p95_added_s, 3.8219, atol=5e-4)
    assert np.isclose(direct_int8.p95_added_s, 5.7544, atol=5e-4)
    assert np.isclose(hazard_rank.p95_added_s, 5.6438, atol=5e-4)
    assert np.isclose(direct_rank.p95_added_s, 5.6438, atol=5e-4)
    assert np.isclose(hazard.loo_p95_min_s, 14.0299, atol=5e-4)
    assert np.isclose(hazard.loo_p95_max_s, 17.6130, atol=5e-4)
    assert np.isclose(direct.loo_p95_min_s, 12.1872, atol=5e-4)
    assert np.isclose(direct.loo_p95_max_s, 30.1512, atol=5e-4)

    strict_direct = events.loc[
        (events["cold_definition"] == "sm_le_0.1_pct")
        & (events["method"] == "direct_monotone")
    ]
    largest = strict_direct.sort_values("model_gib").iloc[-1]
    assert np.isclose(largest.model_gib, 561.2155, atol=5e-4)
    assert np.isclose(largest.params_b, 82.0)
    assert int(largest.parallel_ranks) == 4
    assert np.isclose(largest.allocation_proxy_added_s, 37.0760, atol=5e-4)
    assert np.isclose(largest.fp16_weight_added_s, 11.5461, atol=5e-4)
    assert np.isclose(largest.fp16_per_rank_added_s, 4.3865, atol=5e-4)
    assert len(loo) == int(summary["events_with_volume"].sum())
    assert events.loc[events["params_b"].notna(), "fp16_weight_gib"].gt(0).all()
    assert events["parallel_ranks"].ge(1).all()

    result = {
        "status": "PASS",
        "claims_checked": 30,
        "event_rows": int(len(events)),
        "scenario_rows": int(len(summary)),
        "loo_rows": int(len(loo)),
        "strict_hazard_events": 9,
        "strict_direct_events": 8,
        "strict_hazard_params_known": 8,
        "strict_direct_params_known": 8,
    }
    (ARTIFACT / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

