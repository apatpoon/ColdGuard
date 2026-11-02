from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/q2_upgrade/mechanism_stress"
    )
    args = parser.parse_args()
    artifact_dir = ROOT / args.artifact_dir
    detail = pd.read_csv(artifact_dir / "mechanism_replications.csv")
    summary = pd.read_csv(artifact_dir / "mechanism_summary.csv")
    design = json.loads((artifact_dir / "design.json").read_text())
    replications = int(design["replications_per_scenario"])
    scenarios = list(design["scenarios"])
    protocols = list(design["protocols"])

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, value: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value})

    expected_rows = replications * len(scenarios) * len(protocols)
    check("detail_row_count", len(detail) == expected_rows, len(detail))
    check(
        "summary_row_count",
        len(summary) == len(scenarios) * len(protocols),
        len(summary),
    )
    counts = detail.groupby(["scenario", "protocol"]).size()
    check(
        "balanced_replications",
        bool((counts == replications).all()),
        {f"{scenario}/{protocol}": int(value) for (scenario, protocol), value in counts.items()},
    )

    def row(scenario: str, protocol: str) -> pd.Series:
        match = summary.loc[
            (summary["scenario"] == scenario)
            & (summary["protocol"] == protocol)
        ]
        if len(match) != 1:
            raise AssertionError(f"Missing summary row: {scenario}/{protocol}")
        return match.iloc[0]

    iid_full = row("iid", "coldguard_full")
    check(
        "iid_full_zero_false_promotions",
        int(iid_full["false_promotions"]) == 0,
        int(iid_full["false_promotions"]),
    )
    for scenario in ["abrupt_forward_shift", "informative_endpoint_loss"]:
        full = row(scenario, "coldguard_full")
        no_expiry = row(scenario, "no_forward_expiry")
        check(
            f"expiry_reduces_false_promotions_{scenario}",
            int(full["false_promotions"]) < int(no_expiry["false_promotions"]),
            {
                "full": int(full["false_promotions"]),
                "no_expiry": int(no_expiry["false_promotions"]),
            },
        )

    full_resource = row("resource_concentration", "coldguard_full")
    action_only = row("resource_concentration", "action_risk_only")
    check(
        "dual_risk_reduces_resource_false_promotions",
        int(full_resource["false_promotions"])
        < int(action_only["false_promotions"]),
        {
            "full": int(full_resource["false_promotions"]),
            "action_only": int(action_only["false_promotions"]),
        },
    )

    full_missing = row("informative_endpoint_loss", "coldguard_full")
    complete_case = row("informative_endpoint_loss", "complete_case_only")
    check(
        "maturity_gate_reduces_missingness_false_promotions",
        int(full_missing["false_promotions"])
        < int(complete_case["false_promotions"]),
        {
            "full": int(full_missing["false_promotions"]),
            "complete_case": int(complete_case["false_promotions"]),
        },
    )

    for column in [
        "selection_rate",
        "promotion_rate",
        "false_promotion_rate",
    ]:
        valid = summary[column].between(0.0, 1.0, inclusive="both").all()
        check(f"valid_{column}", bool(valid), [summary[column].min(), summary[column].max()])

    passed = all(item["passed"] for item in checks)
    result = {"passed": passed, "checks": checks}
    (artifact_dir / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        failures = [item for item in checks if not item["passed"]]
        raise SystemExit(f"Q2 mechanism verification failed: {failures}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
