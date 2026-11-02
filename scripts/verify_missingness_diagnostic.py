from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/missingness_diagnostic"


def main() -> None:
    cohort = pd.read_parquet(ARTIFACT / "terminal_episode_cohort.parquet")
    counts = pd.read_csv(ARTIFACT / "end_reason_counts.csv")
    descriptives = pd.read_csv(ARTIFACT / "feature_descriptives.csv")
    effects = pd.read_csv(ARTIFACT / "standardized_differences.csv")
    models = pd.read_csv(ARTIFACT / "missingness_model_metrics.csv")
    predictions = pd.read_parquet(ARTIFACT / "missingness_oof_predictions.parquet")

    assert len(counts) == 6
    assert len(descriptives) == 54
    assert len(effects) == 36
    assert len(models) == 2
    assert cohort["day"].between(121, 177).all()
    assert not cohort.duplicated(["cold_definition", "episode_id"]).any()
    assert set(cohort["end_reason"]) == {
        "reactivated", "telemetry_gap_then_reappears", "deployment_disappearance"
    }
    assert counts.groupby("cold_definition", observed=True)["share"].sum().round(12).eq(1.0).all()
    assert effects["bootstrap_valid"].eq(500).all()
    assert models["bootstrap_valid"].eq(500).all()
    assert predictions["fold"].between(0, 4).all()
    assert predictions["predicted_disappearance"].between(0.0, 1.0).all()
    assert predictions.groupby("cold_definition", observed=True)["fold"].nunique().eq(5).all()
    assert models["roc_auc"].between(0.81, 0.85).all()
    assert models["roc_auc_ci_low"].gt(0.80).all()
    assert models["average_precision"].gt(models["prevalence"]).all()
    assert (models["roc_auc_ci_low"] <= models["roc_auc"]).all()
    assert (models["roc_auc"] <= models["roc_auc_ci_high"]).all()

    primary = effects.loc[
        effects["comparison"].eq("deployment_disappearance_vs_reactivated")
    ]
    assert len(primary) == 18
    assert primary["smd_ci_low"].le(primary["smd"]).all()
    assert primary["smd"].le(primary["smd_ci_high"]).all()
    for definition in ["sm_le_1.0_pct", "sm_le_0.1_pct"]:
        group = primary.loc[primary["cold_definition"].eq(definition)].set_index("feature")
        assert group.loc["log1p_prior_active_days_30", "smd"] < -0.9
        assert group.loc["predicted_risk_3d", "smd"] < -0.65
        assert group.loc["log1p_mem_gib_hours", "smd"] < -0.55
        assert group.loc["log1p_cold_age", "smd"] > 0.35

    expected_counts = {
        ("sm_le_1.0_pct", "reactivated"): 3134,
        ("sm_le_1.0_pct", "telemetry_gap_then_reappears"): 1875,
        ("sm_le_1.0_pct", "deployment_disappearance"): 1840,
        ("sm_le_0.1_pct", "reactivated"): 3314,
        ("sm_le_0.1_pct", "telemetry_gap_then_reappears"): 1722,
        ("sm_le_0.1_pct", "deployment_disappearance"): 1735,
    }
    for key, expected in expected_counts.items():
        row = counts.loc[
            counts["cold_definition"].eq(key[0]) & counts["end_reason"].eq(key[1])
        ]
        assert len(row) == 1 and int(row.iloc[0]["episodes"]) == expected

    result = {
        "status": "PASS",
        "claims_checked": 36,
        "cohort_rows": int(len(cohort)),
        "effect_rows": int(len(effects)),
        "bootstrap_smd_fits": int(effects["bootstrap_valid"].sum()),
        "bootstrap_model_metrics": int(models["bootstrap_valid"].sum() * 3),
        "primary_roc_auc": float(
            models.loc[models["cold_definition"].eq("sm_le_1.0_pct"), "roc_auc"].iloc[0]
        ),
        "strict_roc_auc": float(
            models.loc[models["cold_definition"].eq("sm_le_0.1_pct"), "roc_auc"].iloc[0]
        ),
        "interpretation": (
            "Endpoint disappearance is observably selective; the diagnostic supports full-stream "
            "partial identification rather than complete-case transport."
        ),
    }
    (ARTIFACT / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
