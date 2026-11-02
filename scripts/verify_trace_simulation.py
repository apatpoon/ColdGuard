from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/trace_simulation"


def _assert_nonincreasing(frame: pd.DataFrame, order: str, metric: str) -> None:
    values = frame.sort_values(order)[metric].to_numpy(dtype=float)
    if np.any(np.diff(values) > 1e-8):
        raise AssertionError(f"{metric} increases with {order}: {values}")


def main() -> None:
    replications = pd.read_csv(ARTIFACTS / "replication_results.csv")
    summary = pd.read_csv(ARTIFACTS / "scenario_summary.csv")
    expected_replications = 2 * 3 * 6 * 3 * 3 * 4 * 10
    expected_summaries = 2 * 3 * 6 * 3 * 3 * 4
    assert len(replications) == expected_replications
    assert len(summary) == expected_summaries
    assert (replications["unfinished_reloads"] == 0).all()
    assert replications["actions"].ge(replications["reloads"]).all()

    retain = replications[replications["method"] == "always_retain"]
    assert (retain[["actions", "reloads", "simulated_released_gib_hours"]] == 0).all().all()
    oracle = replications[replications["method"] == "oracle"]
    assert (oracle["reloads"] == 0).all()

    policy_keys = ["cold_definition", "horizon", "method"]
    fixed_metrics = [
        "actions",
        "reloads",
        "simulated_released_gib_hours",
        "safe_released_gib_hours",
        "revival_exposure_gib_hours",
        "reload_volume_gib",
        "action_risk",
        "exposure_risk",
    ]
    for metric in fixed_metrics:
        assert replications.groupby(policy_keys)[metric].nunique(dropna=False).max() == 1

    active = replications[replications["reloads"] > 0]
    for _, group in active.groupby(
        policy_keys + ["seed", "load_slots", "arrival_spread_s"]
    ):
        _assert_nonincreasing(group, "bandwidth_gib_per_s", "p95_added_latency_s")
    for _, group in active.groupby(
        policy_keys + ["seed", "bandwidth_gib_per_s", "arrival_spread_s"]
    ):
        _assert_nonincreasing(group, "load_slots", "p95_queue_wait_s")
    for _, group in active.groupby(
        policy_keys + ["seed", "bandwidth_gib_per_s", "load_slots"]
    ):
        _assert_nonincreasing(group, "arrival_spread_s", "p95_queue_wait_s")

    canonical = summary[
        (summary["cold_definition"] == "sm_le_0.1_pct")
        & (summary["horizon"] == 3)
        & (summary["bandwidth_gib_per_s"] == 16)
        & (summary["load_slots"] == 2)
        & (summary["arrival_spread_s"] == 3_600)
    ].set_index("method")
    hazard = canonical.loc["hazard_hierarchical"]
    direct = canonical.loc["direct_monotone"]
    assert int(hazard["actions"]) == 184
    assert int(hazard["reloads"]) == 9
    assert np.isclose(hazard["simulated_released_gib_hours"], 1_718_281.875)
    assert hazard["p95_added_latency_s"] < direct["p95_added_latency_s"]
    assert hazard["reload_slo_violation_rate_30s"] < direct["reload_slo_violation_rate_30s"]
    burst = summary[
        (summary["cold_definition"] == "sm_le_1.0_pct")
        & (summary["horizon"] == 1)
        & (summary["method"] == "direct_monotone")
        & (summary["bandwidth_gib_per_s"] == 16)
        & (summary["arrival_spread_s"] == 10)
    ].set_index("load_slots")
    expected_burst = {1: 164.881983, 2: 52.213322, 4: 10.208454}
    for slots, expected in expected_burst.items():
        assert np.isclose(burst.loc[slots, "p95_queue_wait_s"], expected, atol=1e-5)
    checks = {
        "replication_rows": int(len(replications)),
        "scenario_rows": int(len(summary)),
        "all_reloads_finished": bool((replications["unfinished_reloads"] == 0).all()),
        "strict_3d_coldguard_actions": int(hazard["actions"]),
        "strict_3d_coldguard_reloads": int(hazard["reloads"]),
        "strict_3d_coldguard_released_gib_hours": float(hazard["simulated_released_gib_hours"]),
        "strict_3d_coldguard_p95_added_latency_s": float(hazard["p95_added_latency_s"]),
        "strict_3d_direct_p95_added_latency_s": float(direct["p95_added_latency_s"]),
        "strict_3d_coldguard_reload_slo_violation_rate_30s": float(hazard["reload_slo_violation_rate_30s"]),
        "strict_3d_direct_reload_slo_violation_rate_30s": float(direct["reload_slo_violation_rate_30s"]),
        "primary_1d_direct_10s_burst_p95_queue_wait_s": {
            str(slots): float(burst.loc[slots, "p95_queue_wait_s"])
            for slots in [1, 2, 4]
        },
        "all_checks_passed": True,
    }
    (ARTIFACTS / "claim_verification_trace_simulation.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print("trace simulation verification passed")


if __name__ == "__main__":
    main()
