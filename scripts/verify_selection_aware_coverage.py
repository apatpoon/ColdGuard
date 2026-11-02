from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/selection_aware_coverage"


def one(frame: pd.DataFrame, scenario: str, protocol: str) -> pd.Series:
    rows = frame.loc[frame.scenario.eq(scenario) & frame.protocol.eq(protocol)]
    assert len(rows) == 1
    return rows.iloc[0]


def main() -> None:
    detail = pd.read_csv(ARTIFACT / "selection_replications.csv")
    summary = pd.read_csv(ARTIFACT / "selection_summary.csv")
    design = json.loads((ARTIFACT / "design.json").read_text())
    assert len(detail) == 3000
    assert len(summary) == 10
    assert design["replications_per_scenario"] == 300
    assert len(design["thresholds"]) == 12
    for column in [
        "selection_rate",
        "calibration_joint_coverage",
        "forward_joint_coverage",
        "forward_true_safe",
        "false_qualification",
        "false_promotion",
    ]:
        finite = summary[column].notna()
        assert summary.loc[finite, column].between(0, 1).all()
        low = summary.loc[finite, f"{column}_ci_low"] if f"{column}_ci_low" in summary else None
        high = summary.loc[finite, f"{column}_ci_high"] if f"{column}_ci_high" in summary else None
        if low is not None:
            assert (low <= summary.loc[finite, column] + 1e-12).all()
            assert (high + 1e-12 >= summary.loc[finite, column]).all()

    pointwise = summary.loc[summary.protocol.eq("fixed_sequence_pointwise")]
    stable = pointwise.loc[~pointwise.scenario.eq("two_axis_test_shift")]
    assert stable.calibration_joint_coverage.min() >= 0.976
    assert stable.forward_joint_coverage.min() >= 0.976
    assert stable.forward_true_safe.min() >= 0.995

    heavy = one(summary, "two_axis_heavy_weight", "fixed_sequence_pointwise")
    assert int(heavy.selected_replications) == 246
    assert np.isclose(heavy.selection_rate, 0.82)
    assert int(heavy.unsafe_forward_selected) == 1
    assert np.isclose(heavy.false_promotion, 0.0)

    drift = one(summary, "two_axis_test_shift", "fixed_sequence_pointwise")
    assert int(drift.selected_replications) == 258
    assert int(drift.unsafe_forward_selected) == 62
    assert np.isclose(drift.forward_true_safe, 196 / 258)
    assert np.isclose(drift.unsafe_caught_by_forward_audit, 61 / 62)
    assert np.isclose(drift.false_promotion, 1 / 258)
    assert drift.forward_true_safe_ci_low < drift.forward_true_safe < drift.forward_true_safe_ci_high

    heavy_bonf = one(summary, "two_axis_heavy_weight", "threshold_bonferroni")
    drift_bonf = one(summary, "two_axis_test_shift", "threshold_bonferroni")
    assert int(heavy_bonf.selected_replications) == 46
    assert int(drift_bonf.selected_replications) == 45
    assert np.isclose(heavy_bonf.calibration_joint_coverage, 1.0)
    assert np.isclose(drift_bonf.forward_true_safe, 1.0)

    result = {
        "status": "PASS",
        "claims_checked": 32,
        "replication_rows": int(len(detail)),
        "summary_rows": int(len(summary)),
        "scenarios": int(summary.scenario.nunique()),
        "protocols": int(summary.protocol.nunique()),
        "drift_pointwise_selected": 258,
        "drift_pointwise_unsafe": 62,
        "drift_pointwise_unsafe_caught": 61,
        "drift_pointwise_false_promotions": 1,
    }
    (ARTIFACT / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
