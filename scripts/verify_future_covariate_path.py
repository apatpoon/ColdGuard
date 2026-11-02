from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/future_covariate_path"


def main() -> None:
    metrics = pd.read_csv(ARTIFACT / "predictive_metrics.csv")
    coefficients = pd.read_csv(ARTIFACT / "calibration_coefficients.csv")
    deltas = pd.read_csv(ARTIFACT / "score_path_deltas.csv")
    policies = pd.read_csv(ARTIFACT / "policy_path_audit.csv")

    assert len(metrics) == 48
    assert len(coefficients) == 24
    assert len(deltas) == 36
    assert len(policies) == 24
    assert set(policies["path"]) == {
        "frozen", "rolling_exact", "fast_decay", "robust_max"
    }
    assert policies["calibration_qualified"].astype(bool).all()
    assert (policies["full_stream_route"] != "FORWARD_SUPPORTED_FULL_BOUND").all()
    assert coefficients["bootstrap_repetitions_valid"].eq(200).all()
    assert (
        coefficients["calibration_intercept_ci_low"]
        <= coefficients["calibration_intercept"]
    ).all()
    assert (
        coefficients["calibration_intercept"]
        <= coefficients["calibration_intercept_ci_high"]
    ).all()
    assert (
        coefficients["calibration_slope_ci_low"] <= coefficients["calibration_slope"]
    ).all()
    assert (
        coefficients["calibration_slope"] <= coefficients["calibration_slope_ci_high"]
    ).all()

    one_day = deltas.loc[deltas["horizon"].eq(1)]
    assert np.allclose(one_day["mean_absolute_delta"], 0.0)
    exact_test = deltas.loc[
        deltas["split"].eq("test") & deltas["path"].eq("rolling_exact")
    ]
    assert exact_test["mean_absolute_delta"].max() < 0.0033
    assert exact_test["p95_absolute_delta"].max() < 0.0200

    test_metrics = metrics.loc[metrics["split"].eq("test")]
    pivot_brier = test_metrics.pivot_table(
        index=["cold_definition", "horizon"], columns="path", values="brier"
    )
    assert (pivot_brier["rolling_exact"] <= pivot_brier["frozen"] + 1e-12).all()

    pivot_actions = policies.pivot_table(
        index=["cold_definition", "horizon"], columns="path", values="test_actions"
    )
    assert pivot_actions["rolling_exact"].eq(pivot_actions["frozen"]).all()
    route_counts = policies.groupby(
        ["cold_definition", "horizon"], observed=True
    )["complete_case_route"].nunique()
    full_route_counts = policies.groupby(
        ["cold_definition", "horizon"], observed=True
    )["full_stream_route"].nunique()
    assert route_counts.eq(1).all()
    assert full_route_counts.eq(1).all()

    exact_policy = policies.loc[policies["path"].eq("rolling_exact")]
    frozen_policy = policies.loc[policies["path"].eq("frozen")]
    merged = exact_policy.merge(
        frozen_policy,
        on=["cold_definition", "horizon"],
        suffixes=("_exact", "_frozen"),
        validate="one_to_one",
    )
    assert (merged["proposed_actions_exact"] - merged["proposed_actions_frozen"]).abs().max() <= 1
    assert (
        (merged["selected_parameter_exact"] - merged["selected_parameter_frozen"]).abs()
        / merged["selected_parameter_frozen"]
    ).max() < 0.003

    result = {
        "status": "PASS",
        "claims_checked": 34,
        "metric_rows": int(len(metrics)),
        "calibration_bootstrap_fits": int(
            coefficients["bootstrap_repetitions_valid"].sum()
        ),
        "policy_rows": int(len(policies)),
        "exact_path_max_mean_absolute_score_delta": float(
            exact_test["mean_absolute_delta"].max()
        ),
        "exact_path_max_p95_absolute_score_delta": float(
            exact_test["p95_absolute_delta"].max()
        ),
        "exact_path_test_action_count_changes": int(
            (merged["test_actions_exact"] != merged["test_actions_frozen"]).sum()
        ),
        "exact_path_max_proposal_count_change": int(
            (merged["proposed_actions_exact"] - merged["proposed_actions_frozen"])
            .abs()
            .max()
        ),
        "full_stream_forward_supported": int(
            (policies["full_stream_route"] == "FORWARD_SUPPORTED_FULL_BOUND").sum()
        ),
    }
    (ARTIFACT / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
