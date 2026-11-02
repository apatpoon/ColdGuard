from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _one(
    frame: pd.DataFrame,
    definition: str,
    horizon: int,
    method: str,
    split: str | None = None,
) -> pd.Series:
    rows = frame[
        (frame["cold_definition"] == definition)
        & (frame["horizon"] == horizon)
        & (frame["method"] == method)
    ]
    if split is not None:
        rows = rows[rows["split"] == split]
    assert len(rows) == 1, (definition, horizon, method, len(rows))
    return rows.iloc[0]


def _close(actual: float, expected: float, tolerance: float = 5e-6) -> None:
    assert np.isclose(actual, expected, atol=tolerance, rtol=0.0), (actual, expected)


def main() -> None:
    artifact = ROOT / "artifacts/revision_round1"
    policy = pd.read_csv(artifact / "operational_policy_audit.csv")
    predictive = pd.read_csv(artifact / "predictive_metrics.csv")
    coverage = pd.read_csv(artifact / "coverage/coverage_summary.csv")
    censoring = pd.read_csv(artifact / "censoring/disappearance_sensitivity.csv")

    primary_h3 = _one(policy, "sm_le_1.0_pct", 3, "hazard_hierarchical")
    assert primary_h3["test_actions"] == 737
    _close(primary_h3["test_action_upper"], 0.0459, 5e-5)
    _close(primary_h3["test_resource_upper"], 0.0483, 5e-5)

    strict_h3 = _one(policy, "sm_le_0.1_pct", 3, "hazard_hierarchical")
    assert strict_h3["test_actions"] == 461
    _close(strict_h3["test_action_upper"], 0.0652, 5e-5)
    _close(strict_h3["test_resource_upper"], 0.0592, 5e-5)
    _close(strict_h3["test_trace_exposure_gib_hours"] / 1e6, 4.236, 5e-4)

    strict_h7 = _one(policy, "sm_le_0.1_pct", 7, "hazard_hierarchical")
    assert strict_h7["test_route"] == "EXPIRED_RECALIBRATE"
    _close(strict_h7["test_resource_upper"], 0.1367, 5e-5)

    assert int(policy["calibration_qualified"].sum()) == 31
    assert int((policy["test_route"] == "FORWARD_SUPPORTED").sum()) == 19

    hazard_p7 = _one(predictive, "sm_le_1.0_pct", 7, "hazard_hierarchical", "test")
    direct_p7 = _one(predictive, "sm_le_1.0_pct", 7, "direct_monotone", "test")
    assert hazard_p7["brier"] < direct_p7["brier"]
    assert hazard_p7["roc_auc"] > direct_p7["roc_auc"]

    assert coverage["cluster_joint_coverage"].min() >= 0.9866
    assert coverage["naive_joint_coverage"].min() < 0.95

    strict_h3_sensitivity = censoring[
        (censoring["cold_definition"] == "sm_le_0.1_pct")
        & (censoring["horizon"] == 3)
    ].set_index("scenario")
    assert strict_h3_sensitivity.loc["optimistic_no_revival", "test_action_upper"] <= 0.10
    assert strict_h3_sensitivity.loc["optimistic_no_revival", "test_resource_upper"] <= 0.10
    assert strict_h3_sensitivity.loc["pessimistic_revival", "test_action_upper"] > 0.10
    assert strict_h3_sensitivity.loc["pessimistic_revival", "test_resource_upper"] > 0.10
    assert censoring["no_lookahead_permutation_pass"].all()

    summary = {
        "status": "PASS",
        "claims_checked": 20,
        "policy_rows": len(policy),
        "predictive_rows": len(predictive),
        "coverage_scenarios": len(coverage),
        "censoring_sensitivity_rows": len(censoring),
    }
    output = artifact / "claim_verification_round1.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
