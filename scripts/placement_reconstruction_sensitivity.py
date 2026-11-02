from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from coldguard.placement import (
    ModelTemplate,
    evaluate_placement_day,
    reconstruct_virtual_blocks,
)
from run_placement_analysis import RAW_GLOB, TEMPLATE_SPECS, _load_actions, _sql_path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/placement_analysis"
OUTPUT = ROOT / "artifacts/revision_round2/placement_reconstruction_sensitivity"
HEADROOMS = (0.15, 0.20, 0.30)


def _canonical_actions() -> pd.DataFrame:
    actions = _load_actions()
    actions = actions.loc[
        actions["cold_definition"].eq("sm_le_0.1_pct")
        & actions["horizon"].eq(3)
        & actions["method"].eq("hazard_hierarchical")
    ].copy()
    actions["release_end_exclusive"] = actions["action_day"] + 3
    revival = actions["revives_within_horizon"].astype(bool)
    actions.loc[revival, "release_end_exclusive"] = np.minimum(
        actions.loc[revival, "release_end_exclusive"].to_numpy(int),
        actions.loc[revival, "revival_day"].to_numpy(int),
    )
    actions["release_end_exclusive"] = actions["release_end_exclusive"].astype(int)
    return actions


def _action_day_instances() -> pd.DataFrame:
    blocks = pd.read_parquet(SOURCE / "action_blocks.parquet")
    blocks = blocks.loc[blocks["cold_definition"].eq("sm_le_0.1_pct")].copy()
    columns = [
        "cold_definition",
        "action_id",
        "site_id",
        "instance_id",
        "cluster_id",
        "gpu_type",
        "instance_block_gib",
        "average_gpu_count",
    ]
    return blocks[columns].drop_duplicates(
        ["cold_definition", "action_id", "instance_id"]
    )


def _trace_daily_instances(actions: pd.DataFrame) -> pd.DataFrame:
    specs = actions[
        [
            "action_id",
            "action_day",
            "release_end_exclusive",
            "app_id",
            "model_id",
            "site_id",
            "pd_role",
        ]
    ].copy()
    connection = duckdb.connect()
    connection.register("action_specs", specs)
    try:
        result = connection.execute(
            f"""
            WITH segments AS (
              SELECT
                s.action_id,
                r.day,
                s.site_id,
                r.instance_id,
                r.cluster_id,
                r.gpu_type,
                r.gpu_count,
                r.gpu_hours,
                r.gpu_mem_alloc,
                LEAST(
                  CASE WHEN r.gpu_count > 0 THEN r.gpu_hours / r.gpu_count ELSE 0 END,
                  24.0
                ) AS observed_hours
              FROM read_parquet('{_sql_path(RAW_GLOB)}', union_by_name=true) AS r
              JOIN action_specs AS s
                ON r.app_id = s.app_id
               AND r.model_id = s.model_id
               AND r.site_id = s.site_id
               AND r.pd_role = s.pd_role
               AND r.day >= s.action_day
               AND r.day < s.release_end_exclusive
              WHERE r.workload = 'online'
                AND r.day BETWEEN 121 AND 152
            )
            SELECT
              action_id,
              day,
              ANY_VALUE(site_id) AS site_id,
              instance_id,
              ANY_VALUE(cluster_id) AS cluster_id,
              ANY_VALUE(gpu_type) AS gpu_type,
              SUM(gpu_mem_alloc * observed_hours) / 24.0 AS instance_block_gib,
              CASE WHEN SUM(observed_hours) > 0
                THEN SUM(gpu_hours) / SUM(observed_hours)
                ELSE 1.0
              END AS average_gpu_count
            FROM segments
            GROUP BY action_id, day, instance_id
            """
        ).fetchdf()
    finally:
        connection.close()
    result["cold_definition"] = "sm_le_0.1_pct"
    return result


def _templates() -> list[ModelTemplate]:
    return [
        ModelTemplate(name, params, 2.0, tp, runtime_headroom=headroom)
        for headroom in HEADROOMS
        for name, params, tp in TEMPLATE_SPECS
    ]


def _summarize_days(
    day_blocks: dict[int, pd.DataFrame], *, source: str, variant: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    empty = pd.DataFrame(columns=["site_id", "cluster_id", "gpu_type", "block_gib"])
    for template in _templates():
        day_results = []
        for day in range(121, 153):
            blocks = day_blocks.get(day, empty)
            day_results.append(evaluate_placement_day(blocks, template))
        rows.append(
            {
                "source": source,
                "variant": variant,
                "runtime_headroom": template.runtime_headroom,
                "template": template.name,
                "tensor_parallel_degree": template.tensor_parallel_degree,
                "footprint_gib": template.footprint_gib,
                "released_gib_days": float(sum(row["released_gib"] for row in day_results)),
                "global_pool_replica_days": int(sum(row["global_pool_replicas"] for row in day_results)),
                "cluster_gpu_pool_replica_days": int(sum(row["cluster_gpu_pool_replicas"] for row in day_results)),
                "block_packed_replica_days": int(sum(row["block_packed_replicas"] for row in day_results)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["placement_retention"] = np.where(
        frame["global_pool_replica_days"] > 0,
        frame["block_packed_replica_days"] / frame["global_pool_replica_days"],
        0.0,
    )
    return frame


def _repeated_action_day(
    actions: pd.DataFrame, blocks: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    result: dict[int, pd.DataFrame] = {}
    for day in range(121, 153):
        ids = actions.loc[
            (actions["action_day"] <= day)
            & (actions["release_end_exclusive"] > day),
            "action_id",
        ]
        result[day] = blocks.loc[blocks["action_id"].isin(ids)].copy()
    return result


def _observed_daily(blocks: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(day): group.copy() for day, group in blocks.groupby("day", sort=False)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    actions = _canonical_actions()
    action_day_instances = _action_day_instances()
    action_day_instances = action_day_instances.loc[
        action_day_instances["action_id"].isin(actions["action_id"])
    ].copy()

    variants = [
        ("single_instance", "single", "uniform"),
        ("floor_uniform", "floor", "uniform"),
        ("round_uniform", "round", "uniform"),
        ("ceil_uniform", "ceil", "uniform"),
        ("round_concentrated", "round", "concentrated"),
    ]
    structural = []
    reconstructed: dict[str, pd.DataFrame] = {}
    for name, count_rule, profile in variants:
        blocks = reconstruct_virtual_blocks(
            action_day_instances,
            count_rule=count_rule,
            allocation_profile=profile,
        )
        reconstructed[name] = blocks
        structural.append(
            _summarize_days(
                _repeated_action_day(actions, blocks),
                source="repeated_action_day",
                variant=name,
            )
        )
    structural_frame = pd.concat(structural, ignore_index=True)
    structural_frame.to_csv(OUTPUT / "structural_sensitivity.csv", index=False)

    trace_instances = _trace_daily_instances(actions)
    trace_blocks = reconstruct_virtual_blocks(
        trace_instances, count_rule="round", allocation_profile="uniform"
    )
    repeated_summary = structural_frame.loc[
        structural_frame["variant"].eq("round_uniform")
    ].copy()
    daily_summary = _summarize_days(
        _observed_daily(trace_blocks),
        source="trace_daily_reconstructed",
        variant="round_uniform",
    )
    daily_compare = pd.concat([repeated_summary, daily_summary], ignore_index=True)
    daily_compare.to_csv(OUTPUT / "daily_reconstruction_sensitivity.csv", index=False)
    trace_instances.to_parquet(OUTPUT / "trace_daily_instance_blocks.parquet", index=False)

    expected_action_days = int(
        (actions["release_end_exclusive"] - actions["action_day"]).sum()
    )
    observed_action_days = int(
        trace_instances[["action_id", "day"]].drop_duplicates().shape[0]
    )
    counts = action_day_instances["average_gpu_count"].to_numpy(float)
    audit = {
        "policy": "strict three-day ColdGuard hazard complete-case shadow set",
        "actions": int(len(actions)),
        "action_day_instances": int(len(action_day_instances)),
        "expected_active_action_days": expected_action_days,
        "trace_observed_action_days": observed_action_days,
        "trace_action_day_coverage": observed_action_days / expected_action_days,
        "trace_daily_instances": int(len(trace_instances)),
        "noninteger_action_day_gpu_count_fraction": float(
            (np.abs(counts - np.rint(counts)) > 1e-9).mean()
        ),
        "floor_round_block_count_equal": bool(
            len(reconstructed["floor_uniform"]) == len(reconstructed["round_uniform"])
        ),
        "round_ceil_block_count_equal": bool(
            len(reconstructed["round_uniform"]) == len(reconstructed["ceil_uniform"])
        ),
        "capacity_conservation": {
            name: float(abs(blocks["block_gib"].sum() - action_day_instances["instance_block_gib"].sum()))
            for name, blocks in reconstructed.items()
        },
    }
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT / "structural_sensitivity.csv")


if __name__ == "__main__":
    main()
