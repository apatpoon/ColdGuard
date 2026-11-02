from __future__ import annotations

import json
from pathlib import Path

import duckdb

from .config import load_config, require_keys, resolve_project_path


def _sql_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def build_episodes(config_path: str | Path) -> tuple[Path, Path]:
    config, project_root = load_config(config_path)
    require_keys(config, "paths", ["deployment_daily", "episodes", "landmarks", "artifacts_dir"])
    require_keys(config, "data", ["logical_keys"])
    require_keys(config, "label", ["horizon_days"])

    daily = resolve_project_path(project_root, config["paths"]["deployment_daily"])
    episodes = resolve_project_path(project_root, config["paths"]["episodes"])
    landmarks = resolve_project_path(project_root, config["paths"]["landmarks"])
    landmarks_all = resolve_project_path(
        project_root,
        config["paths"].get("landmarks_all", "data/interim/cold_landmarks_all.parquet"),
    )
    artifacts = resolve_project_path(project_root, config["paths"]["artifacts_dir"])
    if not daily.exists():
        raise FileNotFoundError(f"Missing deployment-day table: {daily}")
    episodes.parent.mkdir(parents=True, exist_ok=True)
    landmarks.parent.mkdir(parents=True, exist_ok=True)
    landmarks_all.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    compute = config.get("compute", {})
    temp_directory = resolve_project_path(
        project_root, compute.get("duckdb_temp_directory", "data/interim/duckdb_tmp")
    )
    temp_directory.mkdir(parents=True, exist_ok=True)

    keys = list(config["data"]["logical_keys"])
    key_sql = ", ".join(keys)
    partition_sql = ", ".join(keys)
    join_sql = " AND ".join([f"d.{key} IS NOT DISTINCT FROM e.{key}" for key in keys])
    horizon = int(config["label"]["horizon_days"])
    if horizon < 1:
        raise ValueError("label.horizon_days must be positive")

    episodes_tmp = episodes.with_suffix(".tmp.parquet")
    landmarks_tmp = landmarks.with_suffix(".tmp.parquet")
    landmarks_all_tmp = landmarks_all.with_suffix(".tmp.parquet")
    episodes_tmp.unlink(missing_ok=True)
    landmarks_tmp.unlink(missing_ok=True)
    landmarks_all_tmp.unlink(missing_ok=True)

    connection = duckdb.connect()
    try:
        memory_limit = str(compute.get("duckdb_memory_limit", "8GB")).replace("'", "")
        threads = max(1, int(compute.get("duckdb_threads", 4)))
        connection.execute(f"SET memory_limit='{memory_limit}'")
        connection.execute(f"SET temp_directory='{_sql_string(temp_directory)}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET threads TO {threads}")
        daily_sql = _sql_string(daily)

        common_ctes = f"""
        WITH ordered AS (
          SELECT
            *,
            LAG(day) OVER (PARTITION BY {partition_sql} ORDER BY day) AS previous_day,
            LAG(is_cold) OVER (PARTITION BY {partition_sql} ORDER BY day) AS previous_is_cold,
            MAX(CASE WHEN NOT is_cold THEN day END) OVER (
              PARTITION BY {partition_sql} ORDER BY day
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS previous_active_day,
            SUM(CASE WHEN NOT is_cold THEN 1 ELSE 0 END) OVER (
              PARTITION BY {partition_sql} ORDER BY day
              RANGE BETWEEN 30 PRECEDING AND 1 PRECEDING
            ) AS prior_active_days_30
          FROM read_parquet('{daily_sql}')
        ),
        flagged AS (
          SELECT
            *,
            CASE
              WHEN is_cold AND (
                previous_day IS NULL OR previous_day <> day - 1 OR NOT COALESCE(previous_is_cold, false)
              ) THEN 1 ELSE 0
            END AS starts_cold_episode
          FROM ordered
        ),
        tagged AS (
          SELECT
            *,
            SUM(starts_cold_episode) OVER (
              PARTITION BY {partition_sql} ORDER BY day ROWS UNBOUNDED PRECEDING
            ) AS episode_sequence
          FROM flagged
        ),
        cold_rows AS (
          SELECT * FROM tagged WHERE is_cold
        ),
        episode_base AS (
          SELECT
            {key_sql},
            episode_sequence,
            MIN(day) AS start_day,
            MAX(day) AS end_cold_day,
            COUNT(*) AS observed_cold_days,
            SUM(mem_gib_hours) AS cold_mem_gib_hours,
            AVG(mem_gib_hours) AS mean_daily_mem_gib_hours,
            MAX(params_b) AS params_b,
            any_value(base_id) AS base_id,
            any_value(cluster_id) AS cluster_id,
            any_value(gpu_type) AS gpu_type,
            any_value(model_arch) AS model_arch,
            any_value(base_arch) AS base_arch,
            any_value(variant_arch) AS variant_arch
          FROM cold_rows
          GROUP BY {key_sql}, episode_sequence
        ),
        episode_enriched AS (
          SELECT
            e.*,
            (d.day IS NOT NULL AND NOT d.is_cold) AS reactivated,
            CASE WHEN d.day IS NOT NULL AND NOT d.is_cold THEN d.day END AS event_day,
            CASE
              WHEN d.day IS NOT NULL AND NOT d.is_cold THEN 'reactivated'
              ELSE 'right_censored_or_gap'
            END AS episode_end_type,
            hash(concat_ws('|', {', '.join([f"CAST(e.{key} AS VARCHAR)" for key in keys])}, CAST(e.episode_sequence AS VARCHAR))) AS episode_id
          FROM episode_base e
          LEFT JOIN read_parquet('{daily_sql}') d
            ON {join_sql} AND d.day = e.end_cold_day + 1
        )
        """

        connection.execute(
            f"""
            COPY (
              {common_ctes}
              SELECT * FROM episode_enriched ORDER BY {key_sql}, episode_sequence
            ) TO '{_sql_string(episodes_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        connection.execute(
            f"""
            COPY (
              {common_ctes},
              landmark_candidates AS (
                SELECT
                  c.day,
                  {', '.join([f'c.{key}' for key in keys])},
                  e.episode_id,
                  e.episode_sequence,
                  e.start_day,
                  e.end_cold_day,
                  e.reactivated,
                  e.event_day,
                  c.day - e.start_day + 1 AS cold_age,
                  e.episode_sequence - 1 AS prior_cold_episodes,
                  CASE
                    WHEN c.previous_active_day IS NULL THEN NULL
                    ELSE c.day - c.previous_active_day
                  END AS days_since_last_active,
                  COALESCE(c.prior_active_days_30, 0) AS prior_active_days_30,
                  c.base_id,
                  c.cluster_id,
                  c.gpu_type,
                  c.model_arch,
                  c.base_arch,
                  c.variant_arch,
                  c.params_b,
                  c.instance_count,
                  c.cluster_count,
                  c.gpu_type_count,
                  c.gpu_hours,
                  c.observed_hours,
                  c.sm_util,
                  c.zero_fraction,
                  c.mem_gib_hours,
                  SUM(c.mem_gib_hours) OVER (
                    PARTITION BY e.episode_id ORDER BY c.day
                    ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                  ) AS remaining_mem_gib_hours,
                  c.avg_mem_gib_calendar,
                  c.avg_gpu_count,
                  CASE
                    WHEN e.event_day IS NOT NULL AND e.event_day <= c.day + {horizon} THEN 1
                    ELSE 0
                  END AS reactivates_within_horizon,
                  CASE
                    WHEN e.event_day = c.day + 1 THEN 1 ELSE 0
                  END AS event_next_day,
                  (
                    (e.event_day IS NOT NULL AND e.event_day <= c.day + {horizon})
                    OR e.end_cold_day >= c.day + {horizon}
                  ) AS has_full_horizon
                FROM cold_rows c
                JOIN episode_enriched e
                  ON { ' AND '.join([f'c.{key} IS NOT DISTINCT FROM e.{key}' for key in keys]) }
                 AND c.episode_sequence = e.episode_sequence
              )
              SELECT *
              FROM landmark_candidates
              ORDER BY {key_sql}, day
            ) TO '{_sql_string(landmarks_all_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        connection.execute(
            f"""
            COPY (
              {common_ctes},
              landmark_candidates AS (
                SELECT
                  c.day,
                  {', '.join([f'c.{key}' for key in keys])},
                  e.episode_id,
                  e.episode_sequence,
                  e.start_day,
                  e.end_cold_day,
                  e.reactivated,
                  e.event_day,
                  c.day - e.start_day + 1 AS cold_age,
                  e.episode_sequence - 1 AS prior_cold_episodes,
                  CASE
                    WHEN c.previous_active_day IS NULL THEN NULL
                    ELSE c.day - c.previous_active_day
                  END AS days_since_last_active,
                  COALESCE(c.prior_active_days_30, 0) AS prior_active_days_30,
                  c.base_id,
                  c.cluster_id,
                  c.gpu_type,
                  c.model_arch,
                  c.base_arch,
                  c.variant_arch,
                  c.params_b,
                  c.instance_count,
                  c.cluster_count,
                  c.gpu_type_count,
                  c.gpu_hours,
                  c.observed_hours,
                  c.sm_util,
                  c.zero_fraction,
                  c.mem_gib_hours,
                  SUM(c.mem_gib_hours) OVER (
                    PARTITION BY e.episode_id ORDER BY c.day
                    ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                  ) AS remaining_mem_gib_hours,
                  c.avg_mem_gib_calendar,
                  c.avg_gpu_count,
                  CASE
                    WHEN e.event_day IS NOT NULL AND e.event_day <= c.day + {horizon} THEN 1
                    ELSE 0
                  END AS reactivates_within_horizon,
                  CASE
                    WHEN e.event_day = c.day + 1 THEN 1 ELSE 0
                  END AS event_next_day,
                  (
                    (e.event_day IS NOT NULL AND e.event_day <= c.day + {horizon})
                    OR e.end_cold_day >= c.day + {horizon}
                  ) AS has_full_horizon
                FROM cold_rows c
                JOIN episode_enriched e
                  ON { ' AND '.join([f'c.{key} IS NOT DISTINCT FROM e.{key}' for key in keys]) }
                 AND c.episode_sequence = e.episode_sequence
              )
              SELECT *
              FROM landmark_candidates
              WHERE has_full_horizon
              ORDER BY {key_sql}, day
            ) TO '{_sql_string(landmarks_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        stats = connection.execute(
            f"""
            SELECT
              (SELECT COUNT(*) FROM read_parquet('{_sql_string(episodes_tmp)}')) AS episodes,
              (SELECT SUM(reactivated::INTEGER) FROM read_parquet('{_sql_string(episodes_tmp)}')) AS reactivated_episodes,
              (SELECT COUNT(*) FROM read_parquet('{_sql_string(landmarks_tmp)}')) AS eligible_landmarks,
              (SELECT SUM(reactivates_within_horizon) FROM read_parquet('{_sql_string(landmarks_tmp)}')) AS positive_landmarks,
              (SELECT SUM(mem_gib_hours) FROM read_parquet('{_sql_string(landmarks_tmp)}')) AS eligible_mem_gib_hours,
              (SELECT MIN(day) FROM read_parquet('{_sql_string(landmarks_tmp)}')) AS min_day,
              (SELECT MAX(day) FROM read_parquet('{_sql_string(landmarks_tmp)}')) AS max_day
            """
        ).fetchdf().iloc[0].to_dict()
    finally:
        connection.close()

    episodes_tmp.replace(episodes)
    landmarks_tmp.replace(landmarks)
    landmarks_all_tmp.replace(landmarks_all)
    with (artifacts / "episode_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({key: _json_value(value) for key, value in stats.items()}, handle, indent=2)
    return episodes, landmarks


def _json_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value

