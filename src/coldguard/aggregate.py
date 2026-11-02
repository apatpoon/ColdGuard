from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from .config import load_config, require_keys, resolve_project_path


ALLOWED_KEYS = {
    "app_id",
    "model_id",
    "base_id",
    "cluster_id",
    "site_id",
    "gpu_type",
    "pd_role",
}


def _sql_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def aggregate_daily(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(config, "paths", ["raw_daily_glob", "deployment_daily", "artifacts_dir"])
    require_keys(config, "data", ["workload", "day_min", "day_max", "logical_keys"])
    require_keys(
        config,
        "cold_definition",
        ["sm_util_max", "zero_fraction_min", "min_observed_hours", "min_mem_gib_hours"],
    )

    raw_glob = resolve_project_path(project_root, config["paths"]["raw_daily_glob"])
    output = resolve_project_path(project_root, config["paths"]["deployment_daily"])
    artifacts = resolve_project_path(project_root, config["paths"]["artifacts_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    compute = config.get("compute", {})
    temp_directory = resolve_project_path(
        project_root, compute.get("duckdb_temp_directory", "data/interim/duckdb_tmp")
    )
    temp_directory.mkdir(parents=True, exist_ok=True)

    matched = sorted(raw_glob.parent.glob(raw_glob.name))
    if not matched:
        raise FileNotFoundError(
            f"No Parquet shards matched {raw_glob}. Run: bash scripts/download_data.sh daily"
        )

    logical_keys = list(config["data"]["logical_keys"])
    invalid = [key for key in logical_keys if key not in ALLOWED_KEYS]
    if invalid:
        raise ValueError(f"Unsupported logical deployment keys: {invalid}")
    if "app_id" not in logical_keys or "model_id" not in logical_keys:
        raise ValueError("logical_keys must include app_id and model_id")

    key_sql = ", ".join(logical_keys)
    workload = _sql_string(config["data"]["workload"])
    day_min = int(config["data"]["day_min"])
    day_max = int(config["data"]["day_max"])
    cold = config["cold_definition"]

    temp_output = output.with_suffix(".tmp.parquet")
    temp_output.unlink(missing_ok=True)

    query = f"""
    COPY (
      WITH segments AS (
        SELECT
          day,
          instance_id,
          app_id,
          model_id,
          base_id,
          cluster_id,
          site_id,
          gpu_type,
          workload,
          model_arch,
          base_arch,
          variant_arch,
          pd_role,
          params_b,
          gpu_count,
          gpu_hours,
          sm_util,
          gpu_mem_alloc,
          samples,
          zero_samples,
          CASE
            WHEN gpu_count > 0 AND gpu_hours >= 0
              THEN LEAST(gpu_hours / gpu_count, 24.0)
            ELSE 0.0
          END AS observed_hours
        FROM read_parquet('{_sql_string(raw_glob)}', union_by_name=true)
        WHERE workload = '{workload}'
          AND day BETWEEN {day_min} AND {day_max}
      ),
      instance_day AS (
        SELECT
          day,
          instance_id,
          {key_sql},
          any_value(base_id) AS base_id_feature,
          any_value(cluster_id) AS cluster_id_feature,
          any_value(gpu_type) AS gpu_type_feature,
          any_value(model_arch) AS model_arch,
          any_value(base_arch) AS base_arch,
          any_value(variant_arch) AS variant_arch,
          MAX(params_b) AS params_b,
          SUM(gpu_hours) AS gpu_hours,
          SUM(observed_hours) AS observed_hours,
          SUM(COALESCE(sm_util, 0.0) * COALESCE(gpu_hours, 0.0)) AS sm_numerator,
          SUM(COALESCE(zero_samples, 0)) AS zero_samples,
          SUM(COALESCE(samples, 0)) AS samples,
          SUM(COALESCE(gpu_mem_alloc, 0.0) * observed_hours) AS mem_gib_hours
        FROM segments
        GROUP BY day, instance_id, {key_sql}
      ),
      deployment_day AS (
        SELECT
          day,
          {key_sql},
          any_value(base_id_feature) AS base_id,
          any_value(cluster_id_feature) AS cluster_id,
          any_value(gpu_type_feature) AS gpu_type,
          any_value(model_arch) AS model_arch,
          any_value(base_arch) AS base_arch,
          any_value(variant_arch) AS variant_arch,
          MAX(params_b) AS params_b,
          COUNT(DISTINCT instance_id) AS instance_count,
          COUNT(DISTINCT cluster_id_feature) AS cluster_count,
          COUNT(DISTINCT gpu_type_feature) AS gpu_type_count,
          SUM(gpu_hours) AS gpu_hours,
          SUM(observed_hours) AS observed_hours,
          CASE WHEN SUM(gpu_hours) > 0
            THEN SUM(sm_numerator) / SUM(gpu_hours)
            ELSE 0.0 END AS sm_util,
          CASE WHEN SUM(samples) > 0
            THEN SUM(zero_samples)::DOUBLE / SUM(samples)
            ELSE NULL END AS zero_fraction,
          SUM(mem_gib_hours) AS mem_gib_hours,
          SUM(mem_gib_hours) / 24.0 AS avg_mem_gib_calendar,
          CASE WHEN SUM(observed_hours) > 0
            THEN SUM(gpu_hours) / SUM(observed_hours)
            ELSE 0.0 END AS avg_gpu_count
        FROM instance_day
        GROUP BY day, {key_sql}
      )
      SELECT
        *,
        (
          sm_util <= {float(cold['sm_util_max'])}
          AND COALESCE(zero_fraction, 0.0) >= {float(cold['zero_fraction_min'])}
          AND observed_hours >= {float(cold['min_observed_hours'])}
          AND mem_gib_hours >= {float(cold['min_mem_gib_hours'])}
        ) AS is_cold
      FROM deployment_day
      ORDER BY {key_sql}, day
    ) TO '{_sql_string(temp_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    connection = duckdb.connect()
    try:
        memory_limit = str(compute.get("duckdb_memory_limit", "8GB")).replace("'", "")
        threads = max(1, int(compute.get("duckdb_threads", 4)))
        connection.execute(f"SET memory_limit='{memory_limit}'")
        connection.execute(f"SET temp_directory='{_sql_string(temp_directory)}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET threads TO {threads}")
        connection.execute(query)
        stats = connection.execute(
            f"""
            SELECT
              COUNT(*) AS deployment_days,
              COUNT(DISTINCT app_id) AS applications,
              COUNT(DISTINCT model_id) AS models,
              SUM(is_cold::INTEGER) AS cold_days,
              SUM(mem_gib_hours) AS total_mem_gib_hours,
              SUM(CASE WHEN is_cold THEN mem_gib_hours ELSE 0 END) AS cold_mem_gib_hours,
              MIN(day) AS min_day,
              MAX(day) AS max_day
            FROM read_parquet('{_sql_string(temp_output)}')
            """
        ).fetchdf().iloc[0].to_dict()
    finally:
        connection.close()

    temp_output.replace(output)
    audit_path = artifacts / "aggregation_audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump({key: _json_value(value) for key, value in stats.items()}, handle, indent=2)
    return output


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value

