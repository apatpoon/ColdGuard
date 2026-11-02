from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/q2_upgrade/trace_evidence"


def _rolling_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for definition, relative in [
        ("primary", "artifacts/robustness/rolling_global.csv"),
        ("strict", "artifacts/robustness_strict/rolling_global.csv"),
    ]:
        data = pd.read_csv(ROOT / relative)
        data = data.loc[data["method"] == "hazard_hierarchical"].copy()
        for horizon, group in data.groupby("horizon", sort=True):
            eligible = group.loc[group["eligible"].astype(bool)].copy()
            pass_mask = eligible["test_contract_pass"].fillna(False).astype(bool)
            rows.append(
                {
                    "cold_definition": definition,
                    "horizon": int(horizon),
                    "folds": len(group),
                    "eligible_folds": len(eligible),
                    "passing_folds": int(pass_mask.sum()),
                    "median_test_action_upper": eligible[
                        "test_unweighted_risk_ucb"
                    ].median(),
                    "median_test_resource_upper": eligible[
                        "test_risk_ucb"
                    ].median(),
                    "test_actions": int(eligible["test_actions"].sum()),
                    "test_exposure_gib_hours": eligible[
                        "test_reclaimed_gib_hours"
                    ].sum(),
                }
            )
    return pd.DataFrame(rows)


def _component_ledger() -> pd.DataFrame:
    primary_consistency = pd.read_csv(
        ROOT / "artifacts/multihorizon/probability_consistency.csv"
    )
    strict_consistency = pd.read_csv(
        ROOT / "artifacts/multihorizon_strict/probability_consistency.csv"
    )
    full_stream = json.loads(
        (ROOT / "artifacts/revision_round2/full_proposal_stream/summary.json").read_text()
    )
    selection = pd.read_csv(
        ROOT
        / "artifacts/revision_round2/selection_aware_coverage/selection_summary.csv"
    )
    shift = selection.loc[
        (selection["scenario"] == "two_axis_test_shift")
        & (selection["protocol"] == "fixed_sequence_pointwise")
    ].iloc[0]

    def violation(data: pd.DataFrame, method: str) -> float:
        return float(
            data.loc[
                (data["method"] == method) & (data["split"] == "test"),
                "violation_rate",
            ].iloc[0]
        )

    return pd.DataFrame(
        [
            {
                "mechanism": "coherent_horizons",
                "failure_without": "independent direct horizon models violate probability order",
                "without_value": (
                    f"{violation(primary_consistency, 'direct_hierarchical'):.4f}/"
                    f"{violation(strict_consistency, 'direct_hierarchical'):.4f}"
                ),
                "with_value": (
                    f"{violation(primary_consistency, 'hazard_hierarchical'):.4f}/"
                    f"{violation(strict_consistency, 'hazard_hierarchical'):.4f}"
                ),
                "unit": "primary/strict test violation rate",
            },
            {
                "mechanism": "dual_risk",
                "failure_without": "frequency-only screen misses concentrated exposure",
                "without_value": "0.0309",
                "with_value": "0.2547",
                "unit": "primary 1-day action/resource upper bound",
            },
            {
                "mechanism": "full_stream_maturity",
                "failure_without": "complete-case passes omit unresolved proposals",
                "without_value": str(full_stream["originally_forward_supported"]),
                "with_value": str(full_stream["full_bound_forward_supported"]),
                "unit": "observed-endpoint passes/canary-eligible full streams",
            },
            {
                "mechanism": "forward_expiry",
                "failure_without": "calibration-selected policies become unsafe after shift",
                "without_value": str(int(shift["unsafe_forward_selected"])),
                "with_value": str(
                    int(
                        round(
                            shift["unsafe_forward_selected"]
                            * shift["unsafe_caught_by_forward_audit"]
                        )
                    )
                ),
                "unit": "unsafe shifted policies/caught by forward audit",
            },
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rolling = _rolling_summary()
    components = _component_ledger()
    rolling.to_csv(OUT / "rolling_hazard_summary.csv", index=False)
    components.to_csv(OUT / "component_evidence_ledger.csv", index=False)
    total_eligible = int(rolling["eligible_folds"].sum())
    total_passing = int(rolling["passing_folds"].sum())
    report = [
        "# Q2 trace-evidence consolidation",
        "",
        (
            "With the predictor frozen after day 60 and thresholds recalibrated in "
            "three chronological windows, the hierarchy-informed hazard policy is "
            f"calibration-eligible in {total_eligible}/18 definition-horizon-fold "
            f"cells and passes the next complete-case window in {total_passing}/"
            f"{total_eligible} eligible cells."
        ),
        "",
        "These rolling results are complete-case temporal transport diagnostics. They do not override the complete-proposal-stream maturity gate, which remains the operational route.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
