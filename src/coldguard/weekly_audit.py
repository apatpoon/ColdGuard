from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .config import load_config, require_keys, resolve_project_path


def _sql_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def audit_weekly_labels(config_path: str | Path) -> Path:
    """Validate daily cold labels against the released five-minute week.

    The weekly trace does not expose `zero_samples`, so its proxy uses the
    fraction of source-weighted five-minute windows whose mean SM utilization
    is below a configurable threshold. Agreement is diagnostic rather than a
    claim that the two definitions are identical.
    """
    config, project_root = load_config(config_path)
    require_keys(
        config,
        "paths",
        ["raw_daily_glob", "raw_weekly_glob", "deployment_daily", "artifacts_dir"],
    )
    require_keys(config, "data", ["logical_keys", "workload"])

    raw_daily = resolve_project_path(project_root, config["paths"]["raw_daily_glob"])
    raw_weekly = resolve_project_path(project_root, config["paths"]["raw_weekly_glob"])
    deployment_daily = resolve_project_path(project_root, config["paths"]["deployment_daily"])
    output = resolve_project_path(
        project_root,
        config["paths"].get(
            "weekly_label_comparison", "data/interim/weekly_label_comparison.parquet"
        ),
    )
    artifacts = resolve_project_path(project_root, config["paths"]["artifacts_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    for path in [deployment_daily]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not list(raw_daily.parent.glob(raw_daily.name)):
        raise FileNotFoundError(raw_daily)
    if not list(raw_weekly.parent.glob(raw_weekly.name)):
        raise FileNotFoundError(raw_weekly)

    keys = list(config["data"]["logical_keys"])
    key_sql = ", ".join(keys)
    workload = str(config["data"]["workload"]).replace("'", "''")
    compute = config.get("compute", {})
    temp_directory = resolve_project_path(
        project_root, compute.get("duckdb_temp_directory", "data/interim/duckdb_tmp")
    )
    temp_directory.mkdir(parents=True, exist_ok=True)
    memory_limit = str(compute.get("duckdb_memory_limit", "8GB")).replace("'", "")
    threads = max(1, int(compute.get("duckdb_threads", 4)))

    temporary = output.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET temp_directory='{_sql_string(temp_directory)}'")
        con.execute(f"SET threads={threads}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(
            f"""
            COPY (
              WITH daily_map AS (
                SELECT
                  day,
                  instance_id,
                  cluster_id,
                  any_value(app_id) AS app_id,
                  any_value(model_id) AS model_id,
                  any_value(site_id) AS site_id,
                  any_value(pd_role) AS pd_role,
                  COUNT(DISTINCT concat_ws('|', app_id, model_id, site_id, pd_role)) AS map_variants
                FROM read_parquet('{_sql_string(raw_daily)}', union_by_name=true)
                WHERE workload = '{workload}' AND day BETWEEN 140 AND 146
                GROUP BY day, instance_id, cluster_id
              ),
              weekly_instance AS (
                SELECT
                  floor(time_s / 86400)::BIGINT AS day,
                  instance_id,
                  cluster_id,
                  SUM(COALESCE(sm_util, 0.0) * COALESCE(samples, 0)) /
                    NULLIF(SUM(COALESCE(samples, 0)), 0) AS weekly_sm_util,
                  SUM(CASE WHEN COALESCE(sm_util, 0.0) <= 1.0 THEN COALESCE(samples, 0) ELSE 0 END)::DOUBLE /
                    NULLIF(SUM(COALESCE(samples, 0)), 0) AS weekly_low_fraction_1pct,
                  SUM(CASE WHEN COALESCE(sm_util, 0.0) <= 0.1 THEN COALESCE(samples, 0) ELSE 0 END)::DOUBLE /
                    NULLIF(SUM(COALESCE(samples, 0)), 0) AS weekly_low_fraction_01pct,
                  COUNT(*) AS five_min_bins,
                  SUM(COALESCE(samples, 0)) AS source_samples
                FROM read_parquet('{_sql_string(raw_weekly)}', union_by_name=true)
                GROUP BY day, instance_id, cluster_id
              ),
              weekly_enriched AS (
                SELECT w.*, m.app_id, m.model_id, m.site_id, m.pd_role
                FROM weekly_instance w
                JOIN daily_map m USING (day, instance_id, cluster_id)
                WHERE m.map_variants = 1
              ),
              weekly_deployment AS (
                SELECT
                  day,
                  {key_sql},
                  SUM(weekly_sm_util * source_samples) / NULLIF(SUM(source_samples), 0) AS weekly_sm_util,
                  SUM(weekly_low_fraction_1pct * source_samples) / NULLIF(SUM(source_samples), 0) AS weekly_low_fraction_1pct,
                  SUM(weekly_low_fraction_01pct * source_samples) / NULLIF(SUM(source_samples), 0) AS weekly_low_fraction_01pct,
                  SUM(five_min_bins) AS five_min_bins,
                  SUM(source_samples) AS source_samples,
                  COUNT(DISTINCT instance_id) AS weekly_instances
                FROM weekly_enriched
                GROUP BY day, {key_sql}
              )
              SELECT
                d.day,
                {', '.join([f'd.{key}' for key in keys])},
                d.is_cold AS daily_is_cold,
                d.sm_util AS daily_sm_util,
                d.zero_fraction AS daily_zero_fraction,
                d.mem_gib_hours,
                w.weekly_sm_util,
                w.weekly_low_fraction_1pct,
                w.weekly_low_fraction_01pct,
                w.five_min_bins,
                w.source_samples,
                w.weekly_instances
              FROM read_parquet('{_sql_string(deployment_daily)}') d
              JOIN weekly_deployment w
                ON d.day = w.day
               AND {' AND '.join([f'd.{key} IS NOT DISTINCT FROM w.{key}' for key in keys])}
              WHERE d.day BETWEEN 140 AND 146
            ) TO '{_sql_string(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()
    temporary.replace(output)

    frame = pd.read_parquet(output)
    sensitivity_rows: list[dict[str, Any]] = []
    for sm_threshold in [0.1, 0.5, 1.0, 2.0]:
        fraction_column = (
            "weekly_low_fraction_01pct" if sm_threshold == 0.1 else "weekly_low_fraction_1pct"
        )
        for fraction_threshold in [0.80, 0.90, 0.95]:
            weekly_cold = (
                (frame["weekly_sm_util"] <= sm_threshold)
                & (frame[fraction_column] >= fraction_threshold)
            )
            sensitivity_rows.append(
                {
                    "sm_threshold": sm_threshold,
                    "low_fraction_threshold": fraction_threshold,
                    **_classification_metrics(frame["daily_is_cold"].astype(bool), weekly_cold),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(artifacts / "weekly_label_sensitivity.csv", index=False)

    preferred = sensitivity.loc[
        (sensitivity["sm_threshold"] == 1.0)
        & (sensitivity["low_fraction_threshold"] == 0.90)
    ].iloc[0].to_dict()
    audit = {
        "overlap_deployment_days": int(len(frame)),
        "overlap_deployments": int(frame[keys].drop_duplicates().shape[0]),
        "day_range": [int(frame["day"].min()), int(frame["day"].max())],
        "daily_cold_prevalence": float(frame["daily_is_cold"].mean()),
        "weekly_sm_pearson": _safe_corr(frame["daily_sm_util"], frame["weekly_sm_util"], "pearson"),
        "weekly_sm_spearman": _safe_corr(frame["daily_sm_util"], frame["weekly_sm_util"], "spearman"),
        "preferred_proxy": {key: _json_value(value) for key, value in preferred.items()},
        "limitation": (
            "Weekly data contain five-minute mean SM utilization but not the daily table's exact zero_samples. "
            "The weekly cold label is therefore a proxy used for consistency auditing."
        ),
    }
    report_path = artifacts / "weekly_label_audit.json"
    report_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return report_path


def _classification_metrics(reference: pd.Series, candidate: pd.Series) -> dict[str, float | int]:
    reference = reference.to_numpy(bool)
    candidate = candidate.to_numpy(bool)
    tp = int(np.sum(reference & candidate))
    tn = int(np.sum(~reference & ~candidate))
    fp = int(np.sum(~reference & candidate))
    fn = int(np.sum(reference & ~candidate))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "agreement": (tp + tn) / max(1, len(reference)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "weekly_cold_prevalence": float(candidate.mean()),
    }


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    value = left.corr(right, method=method)
    return None if pd.isna(value) else float(value)


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value

