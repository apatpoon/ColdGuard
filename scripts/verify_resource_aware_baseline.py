from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/resource_aware_baseline"


def one(frame: pd.DataFrame, definition: str, horizon: int, method: str) -> pd.Series:
    rows = frame.loc[
        frame.cold_definition.eq(definition)
        & frame.horizon.eq(horizon)
        & frame.method.eq(method)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def main() -> None:
    operational = pd.read_csv(ARTIFACT / "operational_policy_audit.csv")
    candidates = pd.read_csv(ARTIFACT / "candidate_sequence_audit.csv")
    bounds = pd.read_csv(ARTIFACT / "policy_bounds.csv")
    curves = pd.read_csv(ARTIFACT / "tipping_curves.csv")
    actions = pd.read_parquet(ARTIFACT / "selected_actions.parquet")
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))

    assert len(operational) == 12
    assert len(bounds) == 12
    assert len(candidates) == 200
    assert len(curves) == 606
    assert len(actions) == 9240
    assert operational.cold_definition.nunique() == 2
    assert set(operational.horizon) == {1, 3, 7}
    assert set(operational.method) == {
        "size_benefit_recency",
        "reload_cost_recency",
    }
    assert operational.calibration_qualified.sum() == 6
    assert operational.test_route.eq("FORWARD_SUPPORTED").sum() == 2
    assert operational.loc[
        operational.method.eq("size_benefit_recency"), "calibration_qualified"
    ].eq(False).all()
    assert operational.loc[
        operational.method.eq("reload_cost_recency"), "calibration_qualified"
    ].eq(True).all()

    size_candidates = candidates.loc[candidates.method.eq("size_benefit_recency")]
    assert len(size_candidates) == 6
    assert size_candidates.passed.eq(False).all()
    assert size_candidates.actions.ge(200).all()
    cost_candidates = candidates.loc[candidates.method.eq("reload_cost_recency")]
    assert len(cost_candidates) == 194
    assert cost_candidates.groupby(["cold_definition", "horizon"]).passed.last().eq(False).all()

    strict_1 = one(bounds, "sm_le_0.1_pct", 1, "reload_cost_recency")
    primary_1 = one(bounds, "sm_le_1.0_pct", 1, "reload_cost_recency")
    assert (strict_1.proposed_actions, strict_1.observed_actions, strict_1.unresolved_actions) == (
        2054,
        1683,
        371,
    )
    assert (primary_1.proposed_actions, primary_1.observed_actions, primary_1.unresolved_actions) == (
        3353,
        2775,
        578,
    )
    assert np.isclose(strict_1.lower_action_upper, 0.0596346606, atol=1e-6)
    assert np.isclose(strict_1.lower_resource_upper, 0.0329264897, atol=1e-6)
    assert np.isclose(strict_1.upper_action_upper, 0.2900622072, atol=1e-6)
    assert np.isclose(strict_1.upper_resource_upper, 0.1571310638, atol=1e-6)
    assert np.isclose(primary_1.lower_action_upper, 0.0616522998, atol=1e-6)
    assert np.isclose(primary_1.lower_resource_upper, 0.0364107586, atol=1e-6)
    assert np.isclose(primary_1.upper_action_upper, 0.2820353841, atol=1e-6)
    assert np.isclose(primary_1.upper_resource_upper, 0.1740742573, atol=1e-6)
    assert np.isclose(strict_1.dual_bound_first_crossing_fraction, 0.19)
    assert np.isclose(primary_1.dual_bound_first_crossing_fraction, 0.18)

    cost_bounds = bounds.loc[bounds.method.eq("reload_cost_recency")]
    size_bounds = bounds.loc[bounds.method.eq("size_benefit_recency")]
    assert cost_bounds.full_stream_route.eq("SHADOW_ONLY_UNRESOLVED_BOUND").all()
    assert size_bounds.full_stream_route.eq("ABSTAIN_NO_CALIBRATION_SUPPORT").all()
    assert cost_bounds.proposed_actions.sum() == len(actions)
    assert cost_bounds.unresolved_actions.gt(0).all()
    assert cost_bounds.upper_action_upper.gt(0.10).all()
    assert summary["complete_case_forward_supported"] == 2
    assert summary["full_stream_route_counts"] == {
        "SHADOW_ONLY_UNRESOLVED_BOUND": 6,
        "ABSTAIN_NO_CALIBRATION_SUPPORT": 6,
    }
    assert "not reproductions" in summary["interpretation"]
    assert "not identified weight bytes" in summary["allocation_semantics"]

    report = (ARTIFACT / "RESOURCE_AWARE_BASELINE_CN.md").read_text(encoding="utf-8")
    assert "完全一致" in report
    assert "当前研究仍不授权真实内存动作" in report

    result = {
        "status": "PASS",
        "claims_checked": 39,
        "policies": len(operational),
        "candidate_rows": len(candidates),
        "full_stream_actions": len(actions),
        "calibration_qualified": int(operational.calibration_qualified.sum()),
        "complete_case_forward_supported": int(
            operational.test_route.eq("FORWARD_SUPPORTED").sum()
        ),
        "full_stream_shadow_only": int(
            bounds.full_stream_route.eq("SHADOW_ONLY_UNRESOLVED_BOUND").sum()
        ),
        "current_study_authorizes_real_memory_action": False,
    }
    (ARTIFACT / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
