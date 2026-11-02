from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_TIERS = {
    "qwen_0_5b": 0.5,
    "qwen_1_5b": 1.5,
    "qwen_3b": 3.0,
    "qwen_7b": 7.0,
}


def nearest_tier(params_b: float) -> tuple[str, float]:
    model_id = min(MODEL_TIERS, key=lambda key: abs(MODEL_TIERS[key] - params_b))
    return model_id, MODEL_TIERS[model_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    selected = frame[
        frame["params_b"].between(0.5, 7.0, inclusive="both")
        & frame["horizon"].eq(3)
        & frame["cold_definition"].isin(["sm_le_1.0_pct", "sm_le_0.1_pct"])
        & frame["method"].isin(["hazard_hierarchical", "direct_monotone"])
        & frame["revives_within_horizon"].eq(1)
    ].copy()
    if selected.empty:
        raise ValueError("no in-domain revival events")

    group_keys = ["cold_definition", "action_id"]
    methods = (
        selected.groupby(group_keys, sort=True)["method"]
        .agg(lambda values: "+".join(sorted(set(values))))
        .rename("policy_membership")
    )
    unique = selected.sort_values(group_keys + ["method"]).drop_duplicates(group_keys)
    unique = unique.merge(methods, on=group_keys, validate="one_to_one")
    mapped = unique["params_b"].map(nearest_tier)
    unique["model_id"] = mapped.map(lambda value: value[0])
    unique["representative_params_b"] = mapped.map(lambda value: value[1])
    unique["mapping_absolute_error_b"] = (
        unique["params_b"] - unique["representative_params_b"]
    ).abs()
    unique["selected_by_coldguard"] = unique["policy_membership"].str.contains(
        "hazard_hierarchical", regex=False
    )
    unique["selected_by_direct"] = unique["policy_membership"].str.contains(
        "direct_monotone", regex=False
    )
    unique["trace_event_id"] = (
        unique["cold_definition"].astype(str) + ":" + unique["action_id"].astype(str)
    )
    columns = [
        "trace_event_id",
        "cold_definition",
        "definition_label",
        "action_id",
        "action_day",
        "revival_day",
        "params_b",
        "model_id",
        "representative_params_b",
        "mapping_absolute_error_b",
        "model_gib",
        "exposure_gib_hours",
        "policy_membership",
        "selected_by_coldguard",
        "selected_by_direct",
        "qualification_route_full_forward",
        "time_block",
    ]
    output = unique[columns].sort_values(["cold_definition", "action_day", "action_id"])
    if output["trace_event_id"].duplicated().any():
        raise ValueError("trace_event_id is not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        {
            "events": len(output),
            "coldguard_events": int(output["selected_by_coldguard"].sum()),
            "direct_events": int(output["selected_by_direct"].sum()),
            "shared_events": int(
                (output["selected_by_coldguard"] & output["selected_by_direct"]).sum()
            ),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()

