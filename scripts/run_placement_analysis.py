from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from coldguard.placement import ModelTemplate, evaluate_placement_day


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "artifacts/trace_simulation"
DEFAULT_OUTPUT = ROOT / "artifacts/placement_analysis"
RAW_GLOB = ROOT / "data/raw/instance_daily/part_*.parquet"
LANDMARKS = {
    "sm_le_1.0_pct": ROOT / "data/interim/cold_landmarks_all.parquet",
    "sm_le_0.1_pct": ROOT / "data/interim/cold_landmarks_all_strict.parquet",
}
METHODS = ("hazard_hierarchical", "direct_monotone")
TEMPLATE_SPECS = (
    ("7B", 7.0, 1),
    ("13B", 13.0, 1),
    ("32B", 32.0, 2),
    ("70B", 70.0, 4),
)
PRECISIONS = {"FP16/BF16": 2.0, "INT8": 1.0, "INT4": 0.5}
HEADROOMS = (0.15, 0.20, 0.30)


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _load_actions() -> pd.DataFrame:
    actions = pd.read_parquet(TRACE_DIR / "selected_actions.parquet")
    actions = actions.loc[
        actions["method"].isin(METHODS)
        & actions["qualification_route_full_forward"].eq("FORWARD_SUPPORTED")
    ].copy()
    joined: list[pd.DataFrame] = []
    extra_columns = [
        "episode_id",
        "day",
        "pd_role",
        "params_b",
        "instance_count",
        "cluster_count",
        "gpu_type_count",
        "avg_gpu_count",
    ]
    for definition, group in actions.groupby("cold_definition", sort=False):
        landmarks = pd.read_parquet(LANDMARKS[definition], columns=extra_columns)
        local = group.copy()
        local["episode_id_uint64"] = pd.to_numeric(
            local["episode_id"], errors="raise"
        ).astype("uint64")
        local = local.merge(
            landmarks,
            left_on=["episode_id_uint64", "action_day"],
            right_on=["episode_id", "day"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_landmark"),
        )
        if local["pd_role"].isna().any():
            raise ValueError(f"failed to attach physical metadata for {definition}")
        joined.append(local)
    result = pd.concat(joined, ignore_index=True)
    if result.duplicated(
        ["cold_definition", "horizon", "method", "action_id"]
    ).any():
        raise ValueError("policy action keys are not unique")
    return result


def _extract_instance_blocks(
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if not list(RAW_GLOB.parent.glob(RAW_GLOB.name)):
        raise FileNotFoundError(f"raw daily shards not found under {RAW_GLOB.parent}")
    specs = actions[
        [
            "cold_definition",
            "action_id",
            "action_day",
            "app_id",
            "model_id",
            "site_id",
            "pd_role",
            "model_gib",
        ]
    ].drop_duplicates(["cold_definition", "action_id"])
    if specs.duplicated(["cold_definition", "action_id"]).any():
        raise ValueError("physical action specifications are not unique")

    connection = duckdb.connect()
    connection.register("action_specs", specs)
    try:
        instance_blocks = connection.execute(
            f"""
            WITH segments AS (
              SELECT
                s.cold_definition,
                s.action_id,
                s.site_id,
                s.model_gib,
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
                ON r.day = s.action_day
               AND r.app_id = s.app_id
               AND r.model_id = s.model_id
               AND r.site_id = s.site_id
               AND r.pd_role = s.pd_role
              WHERE r.workload = 'online'
                AND r.day BETWEEN {int(actions['action_day'].min())}
                              AND {int(actions['action_day'].max())}
            ), instance_day AS (
              SELECT
                cold_definition,
                action_id,
                ANY_VALUE(site_id) AS site_id,
                ANY_VALUE(model_gib) AS model_gib,
                instance_id,
                ANY_VALUE(cluster_id) AS cluster_id,
                ANY_VALUE(gpu_type) AS gpu_type,
                SUM(gpu_mem_alloc * observed_hours) / 24.0 AS instance_block_gib,
                CASE WHEN SUM(observed_hours) > 0
                  THEN SUM(gpu_hours) / SUM(observed_hours)
                  ELSE 1.0
                END AS average_gpu_count
              FROM segments
              GROUP BY cold_definition, action_id, instance_id
            )
            SELECT * FROM instance_day
            """
        ).fetchdf()
    finally:
        connection.close()

    if instance_blocks.empty:
        raise ValueError("raw trace query returned no instance blocks")
    gpu_count = np.rint(instance_blocks["average_gpu_count"].to_numpy(float))
    if np.any(~np.isfinite(gpu_count)) or np.any(gpu_count < 1):
        raise ValueError("invalid average GPU count in raw trace blocks")
    instance_blocks["virtual_gpu_count"] = gpu_count.astype(np.int64)
    blocks = instance_blocks.loc[
        instance_blocks.index.repeat(instance_blocks["virtual_gpu_count"])
    ].copy()
    blocks["virtual_gpu_index"] = blocks.groupby(
        ["cold_definition", "action_id", "instance_id"], sort=False
    ).cumcount()
    blocks["block_gib"] = (
        blocks["instance_block_gib"] / blocks["virtual_gpu_count"]
    )
    blocks.reset_index(drop=True, inplace=True)

    observed = blocks.groupby(["cold_definition", "action_id"], sort=False)[
        "block_gib"
    ].sum()
    expected = specs.set_index(["cold_definition", "action_id"])["model_gib"]
    aligned = expected.to_frame("expected_gib").join(observed.rename("observed_gib"))
    missing_actions = int(aligned["observed_gib"].isna().sum())
    aligned["relative_error"] = (
        (aligned["observed_gib"] - aligned["expected_gib"]).abs()
        / aligned["expected_gib"].clip(lower=1e-12)
    )
    maximum_error = float(aligned["relative_error"].max())
    if missing_actions or maximum_error > 1e-9:
        raise ValueError(
            "raw block reconstruction failed conservation: "
            f"missing={missing_actions}, max_relative_error={maximum_error:g}"
        )
    audit: dict[str, float | int] = {
        "unique_actions": int(len(specs)),
        "instance_day_blocks": int(len(instance_blocks)),
        "virtual_gpu_blocks": int(len(blocks)),
        "missing_actions": missing_actions,
        "maximum_capacity_conservation_relative_error": maximum_error,
        "multi_gpu_instance_day_fraction": float(
            (instance_blocks["virtual_gpu_count"] > 1).mean()
        ),
        "median_virtual_gpu_block_gib": float(blocks["block_gib"].median()),
        "p90_virtual_gpu_block_gib": float(blocks["block_gib"].quantile(0.90)),
    }
    return blocks, audit


def _template_grid() -> list[tuple[str, ModelTemplate]]:
    result: list[tuple[str, ModelTemplate]] = []
    for precision, bytes_per_parameter in PRECISIONS.items():
        for headroom in HEADROOMS:
            for name, parameters, tensor_parallel in TEMPLATE_SPECS:
                result.append(
                    (
                        precision,
                        ModelTemplate(
                            name=name,
                            parameters_billion=parameters,
                            bytes_per_parameter=bytes_per_parameter,
                            tensor_parallel_degree=tensor_parallel,
                            runtime_headroom=headroom,
                        ),
                    )
                )
    return result


def _evaluate(
    actions: pd.DataFrame, blocks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    templates = _template_grid()
    scenario_columns = [
        "cold_definition",
        "definition_label",
        "horizon",
        "method",
        "method_label",
    ]
    for keys, scenario_actions in actions.groupby(scenario_columns, sort=True):
        metadata = dict(zip(scenario_columns, keys))
        scenario_actions = scenario_actions.copy()
        horizon = int(metadata["horizon"])
        scenario_actions["release_end_exclusive"] = (
            scenario_actions["action_day"] + horizon
        ).astype(int)
        revives = scenario_actions["revives_within_horizon"].astype(bool)
        scenario_actions.loc[revives, "release_end_exclusive"] = np.minimum(
            scenario_actions.loc[revives, "release_end_exclusive"].to_numpy(int),
            scenario_actions.loc[revives, "revival_day"].to_numpy(int),
        )
        scenario_blocks = blocks.merge(
            scenario_actions[
                ["cold_definition", "action_id", "action_day", "release_end_exclusive"]
            ],
            on=["cold_definition", "action_id"],
            how="inner",
            validate="many_to_one",
        )
        if scenario_blocks["action_id"].nunique() != len(scenario_actions):
            raise ValueError("not every selected action has reconstructed placement blocks")

        day_start = 121
        day_end_exclusive = 150 + horizon
        for day in range(day_start, day_end_exclusive):
            active = scenario_blocks.loc[
                (scenario_blocks["action_day"] <= day)
                & (scenario_blocks["release_end_exclusive"] > day)
            ]
            for precision, template in templates:
                result = evaluate_placement_day(active, template)
                rows.append(
                    {
                        **metadata,
                        "day": day,
                        "precision": precision,
                        "bytes_per_parameter": template.bytes_per_parameter,
                        "runtime_headroom": template.runtime_headroom,
                        "template": template.name,
                        "parameters_billion": template.parameters_billion,
                        "tensor_parallel_degree": template.tensor_parallel_degree,
                        "footprint_gib": template.footprint_gib,
                        "rank_gib": template.rank_gib,
                        **result,
                    }
                )
    daily = pd.DataFrame(rows)
    group_columns = scenario_columns + [
        "precision",
        "bytes_per_parameter",
        "runtime_headroom",
        "template",
        "parameters_billion",
        "tensor_parallel_degree",
        "footprint_gib",
        "rank_gib",
    ]
    sum_columns = [
        "released_gib",
        "global_pool_replicas",
        "site_gpu_pool_replicas",
        "cluster_gpu_pool_replicas",
        "block_packed_replicas",
        "block_packed_used_gib",
    ]
    summary = daily.groupby(group_columns, as_index=False, sort=True)[sum_columns].sum()
    day_metrics = daily.groupby(group_columns, as_index=False, sort=True).agg(
        analysis_days=("day", "nunique"),
        days_with_released_capacity=("released_gib", lambda value: int((value > 0).sum())),
        days_with_block_feasible_replica=(
            "block_packed_replicas",
            lambda value: int((value > 0).sum()),
        ),
        peak_released_gib=("released_gib", "max"),
        peak_block_packed_replicas=("block_packed_replicas", "max"),
    )
    summary = summary.merge(day_metrics, on=group_columns, validate="one_to_one")
    scenario_counts = actions.groupby(scenario_columns, as_index=False, sort=True).agg(
        actions=("action_id", "nunique"),
        reloads=("revives_within_horizon", "sum"),
    )
    summary = summary.merge(scenario_counts, on=scenario_columns, validate="many_to_one")
    summary["placement_retention"] = np.where(
        summary["global_pool_replicas"] > 0,
        summary["block_packed_replicas"] / summary["global_pool_replicas"],
        0.0,
    )
    summary["site_locality_retention"] = np.where(
        summary["global_pool_replicas"] > 0,
        summary["site_gpu_pool_replicas"] / summary["global_pool_replicas"],
        0.0,
    )
    summary["cluster_locality_retention"] = np.where(
        summary["global_pool_replicas"] > 0,
        summary["cluster_gpu_pool_replicas"] / summary["global_pool_replicas"],
        0.0,
    )
    summary["block_packed_utilization"] = np.where(
        summary["released_gib"] > 0,
        summary["block_packed_used_gib"] / summary["released_gib"],
        0.0,
    )
    summary.rename(
        columns={
            "released_gib": "released_gib_days",
            "global_pool_replicas": "global_pool_replica_days",
            "site_gpu_pool_replicas": "site_gpu_pool_replica_days",
            "cluster_gpu_pool_replicas": "cluster_gpu_pool_replica_days",
            "block_packed_replicas": "block_packed_replica_days",
            "block_packed_used_gib": "block_packed_used_gib_days",
        },
        inplace=True,
    )
    return daily, summary


def _markdown_table(frame: pd.DataFrame) -> str:
    header = [
        "Definition",
        "Method",
        "Template",
        "Global",
        "Site+GPU",
        "Cluster+GPU",
        "Block-packed",
        "Retention",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.definition_label),
                    str(row.method_label),
                    str(row.template),
                    f"{int(row.global_pool_replica_days):,}",
                    f"{int(row.site_gpu_pool_replica_days):,}",
                    f"{int(row.cluster_gpu_pool_replica_days):,}",
                    f"{int(row.block_packed_replica_days):,}",
                    f"{100 * float(row.placement_retention):.1f}%",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, audit: dict[str, object], output_dir: Path) -> None:
    canonical = summary.loc[
        (summary["horizon"] == 3)
        & summary["precision"].eq("FP16/BF16")
        & np.isclose(summary["runtime_headroom"], 0.20)
    ].sort_values(["cold_definition", "method", "parameters_billion"])
    lines = [
        "# Placement-aware deployment boundary",
        "",
        "This deterministic trace replay converts the selected cold-model allocations into instance-level, time-averaged virtual GPU blocks. It reports nested model-residency bounds: global pooling, same-site and same-GPU-type pooling, same-cluster and same-GPU-type pooling, and block-constrained placement with distinct blocks for tensor-parallel ranks.",
        "",
        "## Canonical FP16/BF16 scenario",
        "",
        "Horizon: 3 days; runtime headroom: 20%; templates: 7B/TP1, 13B/TP1, 32B/TP2, and 70B/TP4. Values are deployable replica-days, not request throughput.",
        "",
        _markdown_table(canonical),
        "",
        "## Reconstruction audit",
        "",
        f"- Unique selected action footprints reconstructed: {int(audit['unique_actions']):,}",
        f"- Instance-day blocks: {int(audit['instance_day_blocks']):,}",
        f"- Virtual GPU blocks after multi-GPU splitting: {int(audit['virtual_gpu_blocks']):,}",
        f"- Maximum memory-conservation relative error: {float(audit['maximum_capacity_conservation_relative_error']):.3e}",
        "",
        "## Interpretation boundary",
        "",
        "The public trace has no physical GPU identifier or within-day segment timestamps. Each instance block is therefore its allocated-memory exposure divided by 24 hours and multi-GPU instances are split evenly across their rounded average GPU count. The result is a daily average placement envelope: it is stricter than a fully fungible logical pool, but it is not proof of instantaneous production placement. Footprints cover model weights plus scenario headroom; they do not model KV-cache demand, kernels, communication topology, or admission-control overhead. The factorial precision/headroom grid is a structural sensitivity analysis and has no sampling confidence interval.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    actions = _load_actions()
    blocks, audit = _extract_instance_blocks(actions)
    daily, summary = _evaluate(actions, blocks)
    blocks.to_parquet(output_dir / "action_blocks.parquet", index=False)
    daily.to_csv(output_dir / "daily_results.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    (output_dir / "reconstruction_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    design = {
        "schema_version": 1,
        "purpose": "Bound how much reclaimed memory remains deployable after locality and block constraints",
        "experimental_unit": "one deterministic policy-definition-horizon trace replay",
        "time_unit": "daily average",
        "policy_methods": list(METHODS),
        "cold_definitions": sorted(LANDMARKS),
        "horizons_days": [1, 3, 7],
        "templates": [
            {"name": name, "parameters_billion": value, "tensor_parallel_degree": tp}
            for name, value, tp in TEMPLATE_SPECS
        ],
        "precision_bytes_per_parameter": PRECISIONS,
        "runtime_headroom": list(HEADROOMS),
        "bounds": [
            "global fungible pool",
            "site and GPU-type pool",
            "cluster and GPU-type pool",
            "virtual-GPU block placement with distinct tensor-parallel ranks",
        ],
        "uncertainty": "factorial structural sensitivity only; deterministic replays are not independent statistical replications",
    }
    (output_dir / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    _write_report(summary, audit, output_dir)
    print(output_dir / "scenario_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output_dir)
