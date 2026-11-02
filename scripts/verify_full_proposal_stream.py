from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/full_proposal_stream"


def _one(frame: pd.DataFrame, definition: str, horizon: int, method: str) -> pd.Series:
    rows = frame.loc[
        (frame["cold_definition"] == definition)
        & (frame["horizon"] == horizon)
        & (frame["method"] == method)
    ]
    assert len(rows) == 1, (definition, horizon, method, len(rows))
    return rows.iloc[0]


def main() -> None:
    policies = pd.read_csv(ARTIFACT / "policy_bounds.csv")
    curves = pd.read_csv(ARTIFACT / "tipping_curves.csv")
    actions = pd.read_parquet(ARTIFACT / "selected_actions.parquet")

    assert len(policies) == 31
    original_forward = policies.loc[
        policies["original_complete_case_route"] == "FORWARD_SUPPORTED"
    ]
    assert len(original_forward) == 19
    assert set(original_forward["full_stream_route"]) == {
        "SHADOW_ONLY_UNRESOLVED_BOUND"
    }
    assert policies["proposed_actions"].eq(
        policies["observed_actions"] + policies["unresolved_actions"]
    ).all()

    expected_grid = np.linspace(0.0, 1.0, 101)
    for key, group in curves.groupby(
        ["cold_definition", "horizon", "method"], observed=True
    ):
        assert len(group) == 101, key
        assert np.allclose(
            group.sort_values("unresolved_revival_fraction")[
                "unresolved_revival_fraction"
            ],
            expected_grid,
        )

    strict = _one(policies, "sm_le_0.1_pct", 3, "hazard_hierarchical")
    assert int(strict["proposed_actions"]) == 514
    assert int(strict["observed_actions"]) == 461
    assert int(strict["unresolved_actions"]) == 53
    assert bool(strict["observed_count_matches_complete_case"])
    assert np.isclose(strict["lower_action_upper"], 0.057964, atol=5e-6)
    assert np.isclose(strict["upper_action_upper"], 0.203791, atol=5e-6)
    assert np.isclose(strict["upper_resource_upper"], 0.231279, atol=5e-6)
    assert np.isclose(strict["dual_bound_first_crossing_fraction"], 0.33)

    primary = _one(policies, "sm_le_1.0_pct", 3, "hazard_hierarchical")
    assert int(primary["proposed_actions"]) == 811
    assert int(primary["observed_actions"]) == 737
    assert int(primary["unresolved_actions"]) == 74
    assert bool(primary["observed_count_matches_complete_case"])
    assert np.isclose(primary["upper_action_upper"], 0.162372, atol=5e-6)
    assert np.isclose(primary["upper_resource_upper"], 0.117051, atol=5e-6)
    assert np.isclose(primary["dual_bound_first_crossing_fraction"], 0.52)

    for row in policies.itertuples(index=False):
        curve = curves.loc[
            (curves["cold_definition"] == row.cold_definition)
            & (curves["horizon"] == row.horizon)
            & (curves["method"] == row.method)
        ].sort_values("unresolved_revival_fraction")
        lower = curve.iloc[0]
        upper = curve.iloc[-1]
        assert np.isclose(lower["action_upper"], row.lower_action_upper)
        assert np.isclose(lower["resource_upper"], row.lower_resource_upper)
        assert np.isclose(upper["action_upper"], row.upper_action_upper)
        assert np.isclose(upper["resource_upper"], row.upper_resource_upper)

    for horizon, group in actions.groupby("horizon", observed=True):
        assert int(group["day"].max()) <= 180 - int(horizon)

    result = {
        "status": "PASS",
        "claims_checked": 18,
        "policies": int(len(policies)),
        "curves": int(len(curves)),
        "selected_action_rows": int(len(actions)),
        "original_forward_supported": int(len(original_forward)),
        "retained_under_full_upper_bound": 0,
        "rerouted_to_shadow_only": int(len(original_forward)),
    }
    (ARTIFACT / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

