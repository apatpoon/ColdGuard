from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/placement_reconstruction_sensitivity"


def main() -> None:
    structural = pd.read_csv(ARTIFACT / "structural_sensitivity.csv")
    daily = pd.read_csv(ARTIFACT / "daily_reconstruction_sensitivity.csv")
    audit = json.loads((ARTIFACT / "audit.json").read_text())
    assert len(structural) == 60
    assert len(daily) == 24
    assert audit["actions"] == 184
    assert audit["expected_active_action_days"] == 540
    assert audit["trace_observed_action_days"] == 540
    assert np.isclose(audit["trace_action_day_coverage"], 1.0)
    assert np.isclose(audit["noninteger_action_day_gpu_count_fraction"], 0.0)
    assert audit["floor_round_block_count_equal"]
    assert audit["round_ceil_block_count_equal"]
    assert max(audit["capacity_conservation"].values()) < 1e-9

    index = ["runtime_headroom", "template"]
    columns = [
        "released_gib_days",
        "global_pool_replica_days",
        "cluster_gpu_pool_replica_days",
        "block_packed_replica_days",
        "placement_retention",
    ]
    floor = structural.loc[structural["variant"].eq("floor_uniform")].set_index(index)[columns]
    rounded = structural.loc[structural["variant"].eq("round_uniform")].set_index(index)[columns]
    ceiling = structural.loc[structural["variant"].eq("ceil_uniform")].set_index(index)[columns]
    single = structural.loc[structural["variant"].eq("single_instance")].set_index(index)[columns]
    concentrated = structural.loc[structural["variant"].eq("round_concentrated")].set_index(index)[columns]
    pd.testing.assert_frame_equal(floor, rounded)
    pd.testing.assert_frame_equal(rounded, ceiling)
    pd.testing.assert_frame_equal(single, concentrated)

    canonical = rounded.loc[0.20]
    expected = {
        "7B": (4606, 3919),
        "13B": (2472, 1682),
        "32B": (997, 546),
        "70B": (447, 186),
    }
    for template, (global_count, packed_count) in expected.items():
        assert int(canonical.loc[template, "global_pool_replica_days"]) == global_count
        assert int(canonical.loc[template, "block_packed_replica_days"]) == packed_count

    repeated = daily.loc[
        daily["source"].eq("repeated_action_day") & np.isclose(daily["runtime_headroom"], 0.20)
    ].set_index("template")
    observed = daily.loc[
        daily["source"].eq("trace_daily_reconstructed") & np.isclose(daily["runtime_headroom"], 0.20)
    ].set_index("template")
    assert np.isclose(repeated.loc["7B", "released_gib_days"], 72296.514198, atol=5e-6)
    assert np.isclose(observed.loc["7B", "released_gib_days"], 71595.079432, atol=5e-6)
    expected_observed = {
        "7B": (4562, 3899),
        "13B": (2448, 1684),
        "32B": (986, 544),
        "70B": (442, 186),
    }
    for template, (global_count, packed_count) in expected_observed.items():
        assert int(observed.loc[template, "global_pool_replica_days"]) == global_count
        assert int(observed.loc[template, "block_packed_replica_days"]) == packed_count

    result = {
        "status": "PASS",
        "claims_checked": 31,
        "structural_rows": int(len(structural)),
        "daily_rows": int(len(daily)),
        "actions": 184,
        "active_action_days": 540,
        "daily_trace_coverage": 1.0,
    }
    (ARTIFACT / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
