from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def _one(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def _truth(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _rolling_summary(relative: str, method: str) -> dict[str, int]:
    frame = _read(relative)
    rows = frame.loc[frame["method"] == method]
    eligible = rows["eligible"].map(_truth)
    passed = rows["test_contract_pass"].map(_truth)
    return {
        "combinations": int(len(rows)),
        "eligible": int(eligible.sum()),
        "passed": int(passed.sum()),
    }


def _selector_summary(relative: str, method: str) -> dict[str, int]:
    frame = _read(relative)
    rows = frame.loc[frame["method"] == method]
    return {
        "combinations": int(len(rows)),
        "eligible": int(rows["eligible"].map(_truth).sum()),
        "passed": int(rows["test_contract_pass"].map(_truth).sum()),
    }


def main() -> None:
    original = _read("artifacts/operational_horizons/global_policies.csv")
    strict = _read("artifacts/operational_horizons_strict/global_policies.csv")
    original_consistency = _read(
        "artifacts/multihorizon/probability_consistency.csv"
    )
    strict_consistency = _read(
        "artifacts/multihorizon_strict/probability_consistency.csv"
    )
    strict_ltt = _read("artifacts/operational_horizons_strict/block_ltt.csv")
    original_windows = _read(
        "artifacts/calibration_window/calibration_window_results.csv"
    )
    strict_windows = _read(
        "artifacts/calibration_window_strict/calibration_window_results.csv"
    )
    strict_selector = _read(
        "artifacts/window_selector_strict/selected_window_results.csv"
    )

    hazard_consistency = pd.concat(
        [original_consistency, strict_consistency], ignore_index=True
    )
    hazard_consistency = hazard_consistency.loc[
        hazard_consistency["method"].str.startswith("hazard_")
    ]
    direct_consistency = pd.concat(
        [original_consistency, strict_consistency], ignore_index=True
    )
    direct_consistency = direct_consistency.loc[
        direct_consistency["method"].str.startswith("direct_")
        & (direct_consistency["split"] == "test")
    ]

    original_1d_hazard = _one(
        original, horizon=1, method="hazard_hierarchical"
    )
    original_1d_ttl = _one(original, horizon=1, method="ttl")
    original_1d_interval = _one(
        original, horizon=1, method="next_interval_hierarchical"
    )
    strict_3d_hazard = _one(strict, horizon=3, method="hazard_hierarchical")
    strict_3d_ttl = _one(strict, horizon=3, method="ttl")
    strict_3d_interval = _one(
        strict, horizon=3, method="next_interval_hierarchical"
    )
    strict_7d_hazard = _one(strict, horizon=7, method="hazard_hierarchical")
    strict_7d_ltt = _one(
        strict_ltt,
        horizon=7,
        block_days=1,
        mode="fixed_sequence",
    )
    original_late_3d_30 = _one(
        original_windows,
        fold="late",
        horizon=3,
        calibration_window_days=30,
        method="hazard_hierarchical",
    )
    original_late_3d_60 = _one(
        original_windows,
        fold="late",
        horizon=3,
        calibration_window_days=60,
        method="hazard_hierarchical",
    )
    strict_middle_3d_30 = _one(
        strict_windows,
        fold="middle",
        horizon=3,
        calibration_window_days=30,
        method="hazard_hierarchical",
    )
    strict_middle_3d_60 = _one(
        strict_windows,
        fold="middle",
        horizon=3,
        calibration_window_days=60,
        method="hazard_hierarchical",
    )
    strict_late_7d_30 = _one(
        strict_windows,
        fold="late",
        horizon=7,
        calibration_window_days=30,
        method="hazard_hierarchical",
    )
    strict_late_7d_60 = _one(
        strict_windows,
        fold="late",
        horizon=7,
        calibration_window_days=60,
        method="hazard_hierarchical",
    )
    strict_selected_middle_3d = _one(
        strict_selector,
        fold="middle",
        horizon=3,
        method="hazard_hierarchical",
    )
    strict_selected_late_7d = _one(
        strict_selector,
        fold="late",
        horizon=7,
        method="hazard_hierarchical",
    )

    original_1d_uplift = (
        float(original_1d_hazard["test_reclaimed_gib_hours"])
        / float(original_1d_ttl["test_reclaimed_gib_hours"])
        - 1.0
    )
    interval_yield_fraction = (
        float(original_1d_interval["test_reclaimed_gib_hours"])
        / float(original_1d_hazard["test_reclaimed_gib_hours"])
    )

    claims = {
        "schema_version": 1,
        "probability_consistency": {
            "hazard_max_violation_rate": float(
                hazard_consistency["violation_rate"].max()
            ),
            "direct_test_min_violation_rate": float(
                direct_consistency["violation_rate"].min()
            ),
            "direct_test_max_violation_rate": float(
                direct_consistency["violation_rate"].max()
            ),
        },
        "original_1d": {
            "hazard_pass": _truth(original_1d_hazard["test_dual_contract_pass"]),
            "hazard_reclaimed_gib_hours": float(
                original_1d_hazard["test_reclaimed_gib_hours"]
            ),
            "ttl_pass": _truth(original_1d_ttl["test_dual_contract_pass"]),
            "ttl_reclaimed_gib_hours": float(
                original_1d_ttl["test_reclaimed_gib_hours"]
            ),
            "hazard_uplift_over_ttl": original_1d_uplift,
            "next_interval_pass": _truth(
                original_1d_interval["test_dual_contract_pass"]
            ),
            "next_interval_reclaimed_gib_hours": float(
                original_1d_interval["test_reclaimed_gib_hours"]
            ),
            "next_interval_yield_fraction_of_hazard": interval_yield_fraction,
        },
        "strict_3d": {
            "hazard_eligible": _truth(strict_3d_hazard["eligible"]),
            "hazard_pass": _truth(strict_3d_hazard["test_dual_contract_pass"]),
            "hazard_reclaimed_gib_hours": float(
                strict_3d_hazard["test_reclaimed_gib_hours"]
            ),
            "ttl_eligible": _truth(strict_3d_ttl["eligible"]),
            "next_interval_eligible": _truth(strict_3d_interval["eligible"]),
        },
        "strict_7d": {
            "global_hazard_eligible": _truth(strict_7d_hazard["eligible"]),
            "global_hazard_pass": _truth(
                strict_7d_hazard["test_dual_contract_pass"]
            ),
            "global_hazard_action_risk_ucb": float(
                strict_7d_hazard["test_unweighted_risk_ucb"]
            ),
            "global_hazard_resource_risk_ucb": float(
                strict_7d_hazard["test_risk_ucb"]
            ),
            "daily_ltt_eligible": _truth(strict_7d_ltt["eligible"]),
            "daily_ltt_pass": _truth(strict_7d_ltt["test_block_contract_pass"]),
            "daily_ltt_action_risk_ucb": float(
                strict_7d_ltt["test_unweighted_risk_ucb"]
            ),
            "daily_ltt_resource_risk_ucb": float(
                strict_7d_ltt["test_risk_ucb"]
            ),
            "daily_ltt_reclaimed_gib_hours": float(
                strict_7d_ltt["test_reclaimed_gib_hours"]
            ),
        },
        "rolling_hazard": {
            "original": _rolling_summary(
                "artifacts/robustness/rolling_global.csv",
                "hazard_hierarchical",
            ),
            "strict": _rolling_summary(
                "artifacts/robustness_strict/rolling_global.csv",
                "hazard_hierarchical",
            ),
        },
        "rolling_next_interval": {
            "original": _rolling_summary(
                "artifacts/robustness/rolling_global.csv",
                "next_interval_hierarchical",
            ),
            "strict": _rolling_summary(
                "artifacts/robustness_strict/rolling_global.csv",
                "next_interval_hierarchical",
            ),
        },
        "calibration_window": {
            "original_late_3d_30": {
                "pass": _truth(original_late_3d_30["test_contract_pass"]),
                "reclaimed_gib_hours": float(
                    original_late_3d_30["test_reclaimed_gib_hours"]
                ),
            },
            "original_late_3d_60": {
                "pass": _truth(original_late_3d_60["test_contract_pass"]),
                "reclaimed_gib_hours": float(
                    original_late_3d_60["test_reclaimed_gib_hours"]
                ),
            },
            "strict_middle_3d_30": {
                "pass": _truth(strict_middle_3d_30["test_contract_pass"]),
                "resource_risk_ucb": float(strict_middle_3d_30["test_risk_ucb"]),
            },
            "strict_middle_3d_60": {
                "pass": _truth(strict_middle_3d_60["test_contract_pass"]),
                "action_risk_ucb": float(
                    strict_middle_3d_60["test_unweighted_risk_ucb"]
                ),
                "resource_risk_ucb": float(strict_middle_3d_60["test_risk_ucb"]),
            },
            "strict_late_7d_30_eligible": _truth(strict_late_7d_30["eligible"]),
            "strict_late_7d_60": {
                "eligible": _truth(strict_late_7d_60["eligible"]),
                "pass": _truth(strict_late_7d_60["test_contract_pass"]),
                "reclaimed_gib_hours": float(
                    strict_late_7d_60["test_reclaimed_gib_hours"]
                ),
            },
        },
        "window_selector": {
            "original_hazard": _selector_summary(
                "artifacts/window_selector/selected_window_results.csv",
                "hazard_hierarchical",
            ),
            "strict_hazard": _selector_summary(
                "artifacts/window_selector_strict/selected_window_results.csv",
                "hazard_hierarchical",
            ),
            "original_ttl": _selector_summary(
                "artifacts/window_selector/selected_window_results.csv",
                "ttl",
            ),
            "strict_ttl": _selector_summary(
                "artifacts/window_selector_strict/selected_window_results.csv",
                "ttl",
            ),
            "strict_middle_3d": {
                "selected_window_days": int(
                    strict_selected_middle_3d["selected_window_days"]
                ),
                "pass": _truth(strict_selected_middle_3d["test_contract_pass"]),
                "action_risk_ucb": float(
                    strict_selected_middle_3d["test_unweighted_risk_ucb"]
                ),
                "resource_risk_ucb": float(
                    strict_selected_middle_3d["test_risk_ucb"]
                ),
            },
            "strict_late_7d": {
                "selected_window_days": int(
                    strict_selected_late_7d["selected_window_days"]
                ),
                "pass": _truth(strict_selected_late_7d["test_contract_pass"]),
                "action_risk_ucb": float(
                    strict_selected_late_7d["test_unweighted_risk_ucb"]
                ),
                "resource_risk_ucb": float(
                    strict_selected_late_7d["test_risk_ucb"]
                ),
            },
        },
    }

    assert claims["probability_consistency"]["hazard_max_violation_rate"] == 0.0
    assert 0.26 <= claims["probability_consistency"]["direct_test_min_violation_rate"]
    assert claims["probability_consistency"]["direct_test_max_violation_rate"] <= 0.45
    assert claims["original_1d"]["hazard_pass"]
    assert claims["original_1d"]["ttl_pass"]
    assert 0.12 <= claims["original_1d"]["hazard_uplift_over_ttl"] <= 0.13
    assert 0.35 <= claims["original_1d"]["next_interval_yield_fraction_of_hazard"] <= 0.37
    assert claims["strict_3d"]["hazard_pass"]
    assert not claims["strict_3d"]["ttl_eligible"]
    assert not claims["strict_3d"]["next_interval_eligible"]
    assert claims["strict_7d"]["global_hazard_eligible"]
    assert not claims["strict_7d"]["global_hazard_pass"]
    assert claims["strict_7d"]["daily_ltt_pass"]
    assert claims["rolling_hazard"]["original"] == {
        "combinations": 9,
        "eligible": 7,
        "passed": 6,
    }
    assert claims["rolling_hazard"]["strict"] == {
        "combinations": 9,
        "eligible": 8,
        "passed": 5,
    }
    assert claims["rolling_next_interval"]["original"] == {
        "combinations": 9,
        "eligible": 1,
        "passed": 0,
    }
    assert claims["rolling_next_interval"]["strict"] == {
        "combinations": 9,
        "eligible": 0,
        "passed": 0,
    }
    assert claims["calibration_window"]["original_late_3d_30"]["pass"]
    assert claims["calibration_window"]["original_late_3d_60"]["pass"]
    assert (
        claims["calibration_window"]["original_late_3d_60"]["reclaimed_gib_hours"]
        > 1.6
        * claims["calibration_window"]["original_late_3d_30"]["reclaimed_gib_hours"]
    )
    assert not claims["calibration_window"]["strict_middle_3d_30"]["pass"]
    assert claims["calibration_window"]["strict_middle_3d_30"]["resource_risk_ucb"] > 0.10
    assert claims["calibration_window"]["strict_middle_3d_60"]["pass"]
    assert claims["calibration_window"]["strict_middle_3d_60"]["resource_risk_ucb"] < 0.04
    assert not claims["calibration_window"]["strict_late_7d_30_eligible"]
    assert claims["calibration_window"]["strict_late_7d_60"]["eligible"]
    assert claims["calibration_window"]["strict_late_7d_60"]["pass"]
    assert claims["window_selector"]["original_hazard"] == {
        "combinations": 9,
        "eligible": 7,
        "passed": 6,
    }
    assert claims["window_selector"]["strict_hazard"] == {
        "combinations": 9,
        "eligible": 9,
        "passed": 7,
    }
    assert claims["window_selector"]["original_ttl"] == {
        "combinations": 9,
        "eligible": 5,
        "passed": 4,
    }
    assert claims["window_selector"]["strict_ttl"] == {
        "combinations": 9,
        "eligible": 3,
        "passed": 1,
    }
    assert claims["window_selector"]["strict_middle_3d"]["selected_window_days"] == 60
    assert claims["window_selector"]["strict_middle_3d"]["pass"]
    assert claims["window_selector"]["strict_middle_3d"]["resource_risk_ucb"] < 0.04
    assert claims["window_selector"]["strict_late_7d"]["selected_window_days"] == 90
    assert claims["window_selector"]["strict_late_7d"]["pass"]
    assert np.isfinite(
        np.asarray(
            [
                claims["original_1d"]["hazard_reclaimed_gib_hours"],
                claims["strict_3d"]["hazard_reclaimed_gib_hours"],
                claims["strict_7d"]["daily_ltt_reclaimed_gib_hours"],
            ]
        )
    ).all()

    destination = ROOT / "artifacts" / "claim_verification.json"
    destination.write_text(
        json.dumps(claims, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
