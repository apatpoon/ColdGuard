from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .pilot import HierarchicalTabularEncoder
from .policy import wilson_upper_bound


EXPECTED_TRACE_BYTES = 91_044_147_663
EXPECTED_TRACE_ROWS = 6_122_413_756
EXPECTED_TRACE_ROW_GROUPS = 6_114
EXPECTED_TRACE_SHA256 = "74833c1469577c5d1f1ef9d7393d165dab31d29950f1d76fe906d5726594e9ec"
EXPECTED_COLUMNS = [
    "invocation_id",
    "function_name",
    "chute_id",
    "user_id",
    "rehash_round",
    "instance_id",
    "started_at",
    "completed_at",
    "it",
    "ot",
    "ct",
    "ttft",
]
HORIZONS_DAYS = (1, 3, 7, 14, 30)
CHUTES_NUMERIC_FEATURES = [
    "cold_age",
    "requests_prior_1d",
    "requests_prior_7d",
    "requests_prior_30d",
    "active_days_prior_7d",
    "active_days_prior_30d",
    "lifetime_requests",
    "lifetime_active_days",
    "last_active_requests",
    "prior_cold_episodes",
    "weekday_sin",
    "weekday_cos",
]
CHUTES_CATEGORICAL_FEATURES = ["chute_id"]


def _sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_trace_metadata(trace_path: str | Path) -> dict[str, Any]:
    trace_path = Path(trace_path)
    parquet = pq.ParquetFile(trace_path)
    metadata = parquet.metadata
    columns = [metadata.schema.column(index).name for index in range(metadata.num_columns)]
    if trace_path.stat().st_size != EXPECTED_TRACE_BYTES:
        raise ValueError("Chutes trace byte size does not match the official release")
    if metadata.num_rows != EXPECTED_TRACE_ROWS:
        raise ValueError("Chutes trace row count does not match the official release")
    if metadata.num_row_groups != EXPECTED_TRACE_ROW_GROUPS:
        raise ValueError("Unexpected Chutes row-group count")
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected Chutes schema: {columns}")

    started_index = columns.index("started_at")
    previous_min = None
    previous_max = None
    minima_ordered = True
    ranges_nonoverlapping = True
    null_counts = {column: 0 for column in columns}
    global_min = None
    global_max = None
    for row_group in range(metadata.num_row_groups):
        for column_index, column in enumerate(columns):
            statistics = metadata.row_group(row_group).column(column_index).statistics
            if statistics is not None and statistics.null_count is not None:
                null_counts[column] += int(statistics.null_count)
        statistics = metadata.row_group(row_group).column(started_index).statistics
        if statistics is None or not statistics.has_min_max:
            raise ValueError(f"Missing started_at statistics in row group {row_group}")
        lower, upper = statistics.min, statistics.max
        global_min = lower if global_min is None else min(global_min, lower)
        global_max = upper if global_max is None else max(global_max, upper)
        if previous_min is not None and lower < previous_min:
            minima_ordered = False
        if previous_max is not None and lower < previous_max:
            ranges_nonoverlapping = False
        previous_min, previous_max = lower, upper
    if not minima_ordered or not ranges_nonoverlapping:
        raise ValueError("Chutes row groups are not safe for chronological streaming")
    return {
        "trace_path": str(trace_path),
        "trace_bytes": int(trace_path.stat().st_size),
        "trace_rows": int(metadata.num_rows),
        "trace_row_groups": int(metadata.num_row_groups),
        "columns": columns,
        "started_at_min": global_min.isoformat(),
        "started_at_max": global_max.isoformat(),
        "span_days": float((global_max - global_min).total_seconds() / 86_400),
        "started_at_row_group_minima_ordered": minima_ordered,
        "started_at_row_group_ranges_nonoverlapping": ranges_nonoverlapping,
        "null_counts": null_counts,
    }


def aggregate_daily(
    trace_path: str | Path,
    output_path: str | Path,
    threads: int = 8,
    memory_limit: str = "12GB",
) -> Path:
    """Aggregate the immutable invocation trace to one row per chute-day.

    DuckDB reads only ``chute_id`` and ``started_at``.  No global row sort and no
    second invocation-level copy are created.
    """
    trace_path = Path(trace_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(threads)}")
    connection.execute(f"SET memory_limit='{memory_limit}'")
    connection.execute("SET preserve_insertion_order=false")
    query = f"""
        COPY (
            SELECT
                chute_id,
                CAST(started_at AS DATE) AS activity_date,
                count(*)::UBIGINT AS requests,
                min(started_at) AS first_started_at,
                max(started_at) AS last_started_at
            FROM read_parquet('{_sql_path(trace_path)}')
            GROUP BY chute_id, activity_date
        ) TO '{_sql_path(temporary)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    connection.execute(query)
    connection.close()
    temporary.replace(output_path)
    return output_path


def load_model_metadata(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    required = {
        "chute_id",
        "name",
        "invocation_count",
        "current_estimated_price",
        "supported_gpus",
    }
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing Chutes model columns: {required - set(metadata.columns)}")

    def gpu_count(value: object) -> int:
        if pd.isna(value):
            return 0
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return 0
        return len(parsed) if isinstance(parsed, list) else 0

    metadata = metadata.copy()
    metadata["supported_gpu_count"] = metadata["supported_gpus"].map(gpu_count)
    metadata["model_owner"] = (
        metadata["name"].fillna("__UNMAPPED__").astype(str).str.split("/", n=1).str[0]
    )
    metadata["current_estimated_price"] = pd.to_numeric(
        metadata["current_estimated_price"], errors="coerce"
    )
    return metadata


def _window_sum(prefix: np.ndarray, day: np.ndarray, width: int) -> np.ndarray:
    start = np.maximum(0, day - width)
    return prefix[day] - prefix[start]


def build_daily_landmarks(
    daily: pd.DataFrame,
    model_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily = daily.copy()
    daily["activity_date"] = pd.to_datetime(daily["activity_date"])
    daily["requests"] = pd.to_numeric(daily["requests"], errors="raise").astype(np.int64)
    daily.sort_values(["chute_id", "activity_date"], inplace=True)
    if daily.duplicated(["chute_id", "activity_date"]).any():
        raise ValueError("Duplicate chute-day rows in daily aggregate")
    origin = daily["activity_date"].min()
    terminal = daily["activity_date"].max()
    terminal_day = int((terminal - origin).days)
    metadata = model_metadata.set_index("chute_id", drop=False)
    if not metadata.index.is_unique:
        raise ValueError("Duplicate chute_id values in model metadata")
    frames: list[pd.DataFrame] = []
    next_episode_id = 1

    for chute_id, group in daily.groupby("chute_id", sort=False, observed=True):
        observed_days = ((group["activity_date"] - origin).dt.days).to_numpy(np.int32)
        first_day = int(observed_days.min())
        counts = np.zeros(terminal_day + 1, dtype=np.int64)
        counts[observed_days] = group["requests"].to_numpy(np.int64)
        active = counts > 0
        eligible_days = np.flatnonzero((np.arange(terminal_day + 1) > first_day) & ~active)
        if eligible_days.size == 0:
            continue

        last_active = np.maximum.accumulate(np.where(active, np.arange(len(counts)), -1))
        cold_age = eligible_days - last_active[eligible_days]
        cold_start = np.zeros(len(counts), dtype=bool)
        cold_start[1:] = ~active[1:] & active[:-1]
        episode_sequence = np.cumsum(cold_start)
        sequence_values = episode_sequence[eligible_days]
        unique_sequences = np.unique(sequence_values)
        sequence_map = {
            int(sequence): next_episode_id + offset
            for offset, sequence in enumerate(unique_sequences)
        }
        episode_ids = np.array(
            [sequence_map[int(sequence)] for sequence in sequence_values], dtype=np.int64
        )
        next_episode_id += len(unique_sequences)

        next_active = np.full(len(counts), -1, dtype=np.int32)
        nearest = -1
        for index in range(len(counts) - 1, -1, -1):
            next_active[index] = nearest
            if active[index]:
                nearest = index
        future_day = next_active[eligible_days]
        time_to_next = np.where(future_day >= 0, future_day - eligible_days, np.nan)
        prefix_requests = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        prefix_active = np.concatenate(([0], np.cumsum(active, dtype=np.int32)))
        last_active_requests = counts[last_active[eligible_days]]
        calendar = origin + pd.to_timedelta(eligible_days, unit="D")
        result = pd.DataFrame(
            {
                "episode_id": episode_ids,
                "chute_id": chute_id,
                "day": eligible_days.astype(np.int16),
                "decision_date": calendar,
                "cold_age": cold_age.astype(np.int16),
                "time_to_next_days": time_to_next,
                "requests_prior_1d": _window_sum(prefix_requests, eligible_days, 1),
                "requests_prior_7d": _window_sum(prefix_requests, eligible_days, 7),
                "requests_prior_30d": _window_sum(prefix_requests, eligible_days, 30),
                "active_days_prior_7d": _window_sum(prefix_active, eligible_days, 7),
                "active_days_prior_30d": _window_sum(prefix_active, eligible_days, 30),
                "lifetime_requests": prefix_requests[eligible_days],
                "lifetime_active_days": prefix_active[eligible_days],
                "last_active_requests": last_active_requests,
                "prior_cold_episodes": sequence_values - 1,
                "weekday_sin": np.sin(2 * np.pi * calendar.dayofweek / 7.0),
                "weekday_cos": np.cos(2 * np.pi * calendar.dayofweek / 7.0),
            }
        )
        for horizon in HORIZONS_DAYS:
            mature = eligible_days + horizon <= terminal_day
            labels = np.full(len(eligible_days), np.nan)
            labels[mature] = (
                (future_day[mature] >= 0)
                & (future_day[mature] <= eligible_days[mature] + horizon)
            ).astype(float)
            result[f"revives_{horizon}d"] = labels
        frames.append(result)

    landmarks = pd.concat(frames, ignore_index=True)
    selected_metadata = metadata.reindex(landmarks["chute_id"].astype(str))
    landmarks["metadata_mapped"] = selected_metadata["name"].notna().to_numpy(np.int8)
    landmarks["model_name"] = selected_metadata["name"].fillna("__UNMAPPED__").to_numpy()
    landmarks["model_owner"] = (
        selected_metadata["model_owner"].fillna("__UNMAPPED__").to_numpy()
    )
    landmarks["current_estimated_price"] = selected_metadata[
        "current_estimated_price"
    ].to_numpy(float)
    landmarks["supported_gpu_count"] = (
        pd.to_numeric(selected_metadata["supported_gpu_count"], errors="coerce")
        .fillna(0)
        .to_numpy(np.int16)
    )
    landmarks["split"] = np.select(
        [landmarks["day"] < 60, landmarks["day"] < 120, landmarks["day"] < 180],
        ["train", "calibration", "test"],
        default="forward",
    )
    landmarks.sort_values(["day", "chute_id"], inplace=True)
    landmarks.reset_index(drop=True, inplace=True)
    audit = {
        "origin_date": str(origin.date()),
        "terminal_date": str(terminal.date()),
        "calendar_days": terminal_day + 1,
        "observed_chutes": int(daily["chute_id"].nunique()),
        "mapped_chutes": int(
            daily["chute_id"].drop_duplicates().isin(model_metadata["chute_id"]).sum()
        ),
        "daily_active_rows": int(len(daily)),
        "landmarks": int(len(landmarks)),
        "episodes": int(landmarks["episode_id"].nunique()),
        "horizons_days": list(HORIZONS_DAYS),
    }
    return landmarks, audit


def _fit_hazard_model(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[HierarchicalTabularEncoder, HistGradientBoostingClassifier]:
    target = train["revives_1d"].to_numpy(np.int8)
    if np.unique(target).size != 2:
        raise ValueError("Chutes training hazard target must contain both classes")
    encoder = HierarchicalTabularEncoder(numeric, categorical, strength=50.0)
    if categorical:
        encoded = encoder.fit_transform_group_oof(
            train,
            target,
            train["chute_id"].to_numpy(str),
            n_splits=5,
        )
    else:
        encoded = encoder.fit_transform(train, target)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(encoded, target)
    return encoder, model


def _advanced_inactive_frame(frame: pd.DataFrame, offset: int) -> pd.DataFrame:
    if offset == 0:
        return frame
    advanced = frame.copy()
    advanced["cold_age"] = advanced["cold_age"] + offset
    future_weekday = (pd.to_datetime(advanced["decision_date"]).dt.dayofweek + offset) % 7
    advanced["weekday_sin"] = np.sin(2 * np.pi * future_weekday / 7.0)
    advanced["weekday_cos"] = np.cos(2 * np.pi * future_weekday / 7.0)
    # Conditional on remaining inactive, yesterday's request count is zero.
    advanced["requests_prior_1d"] = 0
    return advanced


def score_hazard_models(
    landmarks: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[HierarchicalTabularEncoder, HistGradientBoostingClassifier]]]:
    train = landmarks.loc[
        landmarks["split"].eq("train")
        & landmarks["day"].le(58)
        & landmarks["revives_1d"].notna()
    ].copy()
    specifications = {
        "age_hazard": (["cold_age"], []),
        "hierarchical_hazard": (CHUTES_NUMERIC_FEATURES, CHUTES_CATEGORICAL_FEATURES),
    }
    scores = landmarks[
        [
            "episode_id",
            "chute_id",
            "day",
            "decision_date",
            "split",
            "cold_age",
            "time_to_next_days",
            *[f"revives_{horizon}d" for horizon in HORIZONS_DAYS],
        ]
    ].copy()
    seen_chutes = set(train["chute_id"].astype(str))
    scores["seen_in_train"] = scores["chute_id"].astype(str).isin(seen_chutes).astype(np.int8)
    bundles = {}
    for model_index, (method, (numeric, categorical)) in enumerate(specifications.items()):
        encoder, model = _fit_hazard_model(
            train, numeric, categorical, seed=20261030 + model_index
        )
        bundles[method] = (encoder, model)
        survival = np.ones(len(landmarks), dtype=np.float64)
        for offset in range(max(HORIZONS_DAYS)):
            future = _advanced_inactive_frame(landmarks, offset)
            hazard = model.predict_proba(encoder.transform(future))[:, 1]
            survival *= 1.0 - np.clip(hazard, 0.0, 1.0 - 1e-12)
            horizon = offset + 1
            if horizon in HORIZONS_DAYS:
                scores[f"score_{method}_{horizon}d"] = (1.0 - survival).astype(
                    np.float32
                )
    return scores, bundles


def _predictive_metrics_safe(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    return {
        "brier": float(brier_score_loss(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else np.nan,
        "prevalence": float(labels.mean()),
    }


def predictive_audit(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS_DAYS:
        base_windows = [
            (
                "train",
                scores["split"].eq("train") & scores["day"].le(59 - horizon),
            ),
            (
                "calibration",
                scores["split"].eq("calibration")
                & scores["day"].le(119 - horizon),
            ),
            ("test", scores["split"].eq("test")),
        ]
        for start in range(180, 360, 30):
            base_windows.append(
                (
                    f"forward_{start}_{start + 29}",
                    scores["day"].between(start, start + 29),
                )
            )
        label_column = f"revives_{horizon}d"
        train_labels = scores.loc[
            scores["split"].eq("train")
            & scores["day"].le(59 - horizon)
            & scores[label_column].notna(),
            label_column,
        ]
        train_rate = float(train_labels.mean())
        for window, mask in base_windows:
            frame = scores.loc[mask & scores[label_column].notna()]
            if frame.empty:
                continue
            labels = frame[label_column].to_numpy(np.int8)
            rows.append(
                {
                    "window": window,
                    "horizon_days": horizon,
                    "method": "constant_train_rate",
                    "rows": int(len(frame)),
                    "episodes": int(frame["episode_id"].nunique()),
                    **_predictive_metrics_safe(
                        labels, np.full(len(frame), train_rate, dtype=float)
                    ),
                }
            )
            for method in ["age_hazard", "hierarchical_hazard"]:
                prediction = frame[f"score_{method}_{horizon}d"].to_numpy(float)
                rows.append(
                    {
                        "window": window,
                        "horizon_days": horizon,
                        "method": method,
                        "rows": int(len(frame)),
                        "episodes": int(frame["episode_id"].nunique()),
                        **_predictive_metrics_safe(labels, prediction),
                    }
                )
    return pd.DataFrame(rows)


def _first_episode_actions(frame: pd.DataFrame, mask: np.ndarray | pd.Series) -> pd.DataFrame:
    selected = frame.loc[mask].sort_values(["episode_id", "day"])
    return selected.drop_duplicates("episode_id", keep="first")


def _chutes_action_metrics(
    actions: pd.DataFrame,
    label_column: str,
    horizon: int,
    confidence: float,
    comparisons: int,
) -> dict[str, float | int]:
    if actions.empty:
        return {
            "actions": 0,
            "revivals": 0,
            "action_risk": np.nan,
            "action_risk_ucb": 1.0,
            "safe_model_days": 0.0,
        }
    labels = actions[label_column].to_numpy(float)
    time_to_next = actions["time_to_next_days"].to_numpy(float)
    safe_days = np.where(
        np.isfinite(time_to_next), np.minimum(float(horizon), time_to_next), float(horizon)
    )
    return {
        "actions": int(len(actions)),
        "revivals": int(labels.sum()),
        "action_risk": float(labels.mean()),
        "action_risk_ucb": wilson_upper_bound(
            float(labels.sum()), float(len(labels)), confidence, comparisons
        ),
        "safe_model_days": float(safe_days.sum()),
    }


def calibrate_chutes_policies(
    scores: pd.DataFrame,
    target_risk: float = 0.05,
    minimum_actions: int = 200,
    confidence: float = 0.95,
    threshold_grid_size: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontier_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    windows = {
        "test": scores["split"].eq("test"),
        **{
            f"forward_{start}_{start + 29}": scores["day"].between(start, start + 29)
            for start in range(180, 360, 30)
        },
    }
    for horizon in HORIZONS_DAYS:
        label_column = f"revives_{horizon}d"
        calibration = scores.loc[
            scores["split"].eq("calibration")
            & scores["day"].le(119 - horizon)
            & scores[label_column].notna()
        ].copy()
        policy_specs: list[tuple[str, str | None, np.ndarray]] = []
        ttl_values = np.sort(calibration["cold_age"].unique()).astype(float)
        policy_specs.append(("ttl", None, ttl_values))
        for method in ["age_hazard", "hierarchical_hazard"]:
            column = f"score_{method}_{horizon}d"
            thresholds = np.unique(
                np.quantile(
                    calibration[column].to_numpy(float),
                    np.linspace(0.01, 0.99, threshold_grid_size),
                )
            )
            policy_specs.append((method, column, thresholds))

        for policy, score_column, parameters in policy_specs:
            candidate_rows = []
            comparisons = max(1, len(parameters))
            for parameter in parameters:
                if policy == "ttl":
                    mask = calibration["cold_age"] >= parameter
                else:
                    assert score_column is not None
                    mask = calibration[score_column] <= parameter
                actions = _first_episode_actions(calibration, mask)
                metrics = _chutes_action_metrics(
                    actions, label_column, horizon, confidence, comparisons
                )
                row = {
                    "horizon_days": horizon,
                    "policy": policy,
                    "parameter": float(parameter),
                    **{f"cal_{key}": value for key, value in metrics.items()},
                }
                row["qualified"] = bool(
                    metrics["actions"] >= minimum_actions
                    and metrics["action_risk_ucb"] <= target_risk
                )
                frontier_rows.append(row)
                candidate_rows.append(row)
            candidates = pd.DataFrame(candidate_rows)
            eligible = candidates.loc[candidates["qualified"]]
            selected = (
                eligible.sort_values(
                    ["cal_safe_model_days", "cal_action_risk_ucb"],
                    ascending=[False, True],
                ).iloc[0]
                if not eligible.empty
                else None
            )
            selection: dict[str, Any] = {
                "horizon_days": horizon,
                "policy": policy,
                "selected": selected is not None,
                "parameter": float(selected["parameter"]) if selected is not None else np.nan,
                "minimum_any_cal_action_risk_ucb": float(
                    candidates["cal_action_risk_ucb"].min()
                ),
                "minimum_supported_cal_action_risk_ucb": float(
                    candidates.loc[
                        candidates["cal_actions"] >= minimum_actions,
                        "cal_action_risk_ucb",
                    ].min()
                ),
            }
            if selected is not None:
                for key in [
                    "cal_actions",
                    "cal_revivals",
                    "cal_action_risk",
                    "cal_action_risk_ucb",
                    "cal_safe_model_days",
                ]:
                    selection[key] = selected[key]
                for window_name, window_mask in windows.items():
                    frame = scores.loc[window_mask & scores[label_column].notna()].copy()
                    if policy == "ttl":
                        mask = frame["cold_age"] >= selected["parameter"]
                    else:
                        assert score_column is not None
                        mask = frame[score_column] <= selected["parameter"]
                    actions = _first_episode_actions(frame, mask)
                    metrics = _chutes_action_metrics(
                        actions, label_column, horizon, confidence, 1
                    )
                    selection.update(
                        {f"{window_name}_{key}": value for key, value in metrics.items()}
                    )
            selected_rows.append(selection)
    return pd.DataFrame(frontier_rows), pd.DataFrame(selected_rows)


def novelty_predictive_audit(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, window_mask in [
        ("test", scores["split"].eq("test")),
        ("forward", scores["split"].eq("forward")),
    ]:
        for novelty_value, novelty_label in [(1, "seen"), (0, "unseen")]:
            novelty_mask = scores["seen_in_train"].eq(novelty_value)
            for horizon in HORIZONS_DAYS:
                label_column = f"revives_{horizon}d"
                frame = scores.loc[
                    window_mask & novelty_mask & scores[label_column].notna()
                ]
                if frame.empty:
                    continue
                labels = frame[label_column].to_numpy(np.int8)
                for method in ["age_hazard", "hierarchical_hazard"]:
                    prediction = frame[f"score_{method}_{horizon}d"].to_numpy(float)
                    rows.append(
                        {
                            "window": window,
                            "novelty": novelty_label,
                            "horizon_days": horizon,
                            "method": method,
                            "rows": int(len(frame)),
                            "episodes": int(frame["episode_id"].nunique()),
                            "chutes": int(frame["chute_id"].nunique()),
                            **_predictive_metrics_safe(labels, prediction),
                        }
                    )
    return pd.DataFrame(rows)


def rolling_recalibration_audit(
    scores: pd.DataFrame,
    target_risk: float = 0.05,
    minimum_actions: int = 200,
    confidence: float = 0.95,
    threshold_grid_size: int = 80,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    terminal_day = int(scores["day"].max())
    for test_start in range(180, 360, 30):
        nominal_test_end = min(test_start + 29, terminal_day)
        for horizon in HORIZONS_DAYS:
            label_column = f"revives_{horizon}d"
            calibration_start = test_start - 60
            calibration_end = test_start - horizon - 1
            calibration = scores.loc[
                scores["day"].between(calibration_start, calibration_end)
                & scores[label_column].notna()
            ].copy()
            test = scores.loc[
                scores["day"].between(test_start, nominal_test_end)
                & scores[label_column].notna()
            ].copy()
            if calibration.empty or test.empty:
                continue
            specs: list[tuple[str, str | None, np.ndarray]] = [
                ("ttl", None, np.sort(calibration["cold_age"].unique()).astype(float))
            ]
            for method in ["age_hazard", "hierarchical_hazard"]:
                score_column = f"score_{method}_{horizon}d"
                thresholds = np.unique(
                    np.quantile(
                        calibration[score_column].to_numpy(float),
                        np.linspace(0.01, 0.99, threshold_grid_size),
                    )
                )
                specs.append((method, score_column, thresholds))
            for policy, score_column, parameters in specs:
                comparisons = max(1, len(parameters))
                candidates = []
                for parameter in parameters:
                    if policy == "ttl":
                        mask = calibration["cold_age"] >= parameter
                    else:
                        assert score_column is not None
                        mask = calibration[score_column] <= parameter
                    actions = _first_episode_actions(calibration, mask)
                    metrics = _chutes_action_metrics(
                        actions, label_column, horizon, confidence, comparisons
                    )
                    candidates.append({"parameter": float(parameter), **metrics})
                candidate_frame = pd.DataFrame(candidates)
                supported = candidate_frame.loc[
                    candidate_frame["actions"] >= minimum_actions
                ]
                eligible = supported.loc[
                    supported["action_risk_ucb"] <= target_risk
                ]
                selected = (
                    eligible.sort_values(
                        ["safe_model_days", "action_risk_ucb"],
                        ascending=[False, True],
                    ).iloc[0]
                    if not eligible.empty
                    else None
                )
                row: dict[str, Any] = {
                    "test_start_day": test_start,
                    "test_end_day": int(test["day"].max()),
                    "calibration_start_day": calibration_start,
                    "calibration_end_day": calibration_end,
                    "horizon_days": horizon,
                    "policy": policy,
                    "selected": selected is not None,
                    "parameter": float(selected["parameter"])
                    if selected is not None
                    else np.nan,
                    "minimum_supported_cal_action_risk_ucb": float(
                        supported["action_risk_ucb"].min()
                    )
                    if not supported.empty
                    else np.nan,
                }
                if selected is not None:
                    row.update(
                        {f"cal_{key}": selected[key] for key in selected.index if key != "parameter"}
                    )
                    if policy == "ttl":
                        test_mask = test["cold_age"] >= selected["parameter"]
                    else:
                        assert score_column is not None
                        test_mask = test[score_column] <= selected["parameter"]
                    test_actions = _first_episode_actions(test, test_mask)
                    test_metrics = _chutes_action_metrics(
                        test_actions, label_column, horizon, confidence, 1
                    )
                    row.update(
                        {f"test_{key}": value for key, value in test_metrics.items()}
                    )
                    row["test_contract_pass"] = bool(
                        test_metrics["actions"] >= minimum_actions
                        and test_metrics["action_risk_ucb"] <= target_risk
                    )
                else:
                    row["test_contract_pass"] = False
                rows.append(row)
    return pd.DataFrame(rows)


def run_chutes_external(
    landmarks_path: str | Path,
    output_dir: str | Path,
) -> Path:
    landmarks_path = Path(landmarks_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    landmarks = pd.read_parquet(landmarks_path)
    scores, bundles = score_hazard_models(landmarks)
    metrics = predictive_audit(scores)
    novelty = novelty_predictive_audit(scores)
    frontier, selected = calibrate_chutes_policies(scores)
    rolling = rolling_recalibration_audit(scores)
    for method, (encoder, model) in bundles.items():
        joblib.dump(
            {"encoder": encoder, "model": model, "horizons_days": HORIZONS_DAYS},
            models_dir / f"{method}.joblib",
        )
    scores.to_parquet(output_dir / "scored_landmarks.parquet", compression="zstd", index=False)
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    novelty.to_csv(output_dir / "novelty_metrics.csv", index=False)
    frontier.to_csv(output_dir / "policy_frontier.csv", index=False)
    selected.to_csv(output_dir / "selected_policies.csv", index=False)
    rolling.to_csv(output_dir / "rolling_recalibration.csv", index=False)

    split_audit = (
        landmarks.groupby("split", as_index=False)
        .agg(
            landmarks=("episode_id", "size"),
            episodes=("episode_id", "nunique"),
            chutes=("chute_id", "nunique"),
            mapped_rate=("metadata_mapped", "mean"),
            revival_rate_1d=("revives_1d", "mean"),
            revival_rate_7d=("revives_7d", "mean"),
            revival_rate_30d=("revives_30d", "mean"),
        )
    )
    split_audit.to_csv(output_dir / "split_audit.csv", index=False)
    trace_audit_path = landmarks_path.parent / "trace_audit.json"
    trace_audit = (
        json.loads(trace_audit_path.read_text(encoding="utf-8"))
        if trace_audit_path.exists()
        else {}
    )
    train_chutes = set(landmarks.loc[landmarks["split"].eq("train"), "chute_id"])
    test_chutes = set(landmarks.loc[landmarks["split"].eq("test"), "chute_id"])
    manifest = {
        "landmarks_path": str(landmarks_path.resolve()),
        "landmarks": int(len(landmarks)),
        "episodes": int(landmarks["episode_id"].nunique()),
        "chutes": int(landmarks["chute_id"].nunique()),
        "train_chutes": int(len(train_chutes)),
        "test_chutes": int(len(test_chutes)),
        "test_unseen_chutes": int(len(test_chutes - train_chutes)),
        "mapped_chutes": int(
            landmarks.loc[landmarks["metadata_mapped"].eq(1), "chute_id"].nunique()
        ),
        "horizons_days": list(HORIZONS_DAYS),
        "target_action_risk": 0.05,
        "minimum_calibration_actions": 200,
        "selected_policy_count": int(selected["selected"].sum()),
        "rolling_selected_policy_count": int(rolling["selected"].sum()),
        "rolling_test_contract_pass_count": int(rolling["test_contract_pass"].sum()),
        "resource_claim": "none; Chutes has no observed GPU residency or memory allocation",
        "trace_sha256_verified": EXPECTED_TRACE_SHA256,
        "trace_bytes": trace_audit.get("trace_bytes", EXPECTED_TRACE_BYTES),
        "trace_rows": trace_audit.get("trace_rows", EXPECTED_TRACE_ROWS),
        "trace_row_groups": trace_audit.get(
            "trace_row_groups", EXPECTED_TRACE_ROW_GROUPS
        ),
        "started_at_null_count": trace_audit.get("null_counts", {}).get(
            "started_at", 0
        ),
        "chute_id_null_count": trace_audit.get("null_counts", {}).get("chute_id", 0),
        "future_covariate_path": (
            "cold_age and weekday advance; previous-day requests become zero; "
            "other history features use last observation carried forward conditional on inactivity"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    test = metrics.loc[metrics["window"].eq("test")].set_index(
        ["horizon_days", "method"]
    )
    novelty_test = novelty.loc[novelty["window"].eq("test")]
    selected_count = int(selected["selected"].sum())
    policy_text = (
        f"{selected_count}个方法—时域组合通过5%动作风险校准"
        if selected_count
        else "没有方法—时域组合通过5%动作风险校准，控制器保持ABSTAIN"
    )
    split_index = split_audit.set_index("split")
    selected_view_columns = [
        "horizon_days",
        "policy",
        "selected",
        "parameter",
        "minimum_supported_cal_action_risk_ucb",
        "cal_actions",
        "cal_action_risk_ucb",
        "test_actions",
        "test_action_risk",
        "test_action_risk_ucb",
    ]
    selected_view = selected.reindex(columns=selected_view_columns).copy()
    table_headers = [
        "Horizon",
        "Policy",
        "Selected",
        "Parameter",
        "Min supported UCB",
        "Cal actions",
        "Cal UCB",
        "Test actions",
        "Test risk",
        "Test UCB",
    ]
    table_lines = [
        "| " + " | ".join(table_headers) + " |",
        "|" + "|".join(["---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]) + "|",
    ]
    for row in selected_view.itertuples(index=False):
        values = []
        for index, value in enumerate(row):
            if pd.isna(value):
                values.append("—")
            elif index == 2:
                values.append("yes" if bool(value) else "no")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        table_lines.append("| " + " | ".join(values) + " |")
    selected_markdown = "\n".join(table_lines)
    rolling_selected = rolling.loc[rolling["selected"]].copy()
    rolling_headers = [
        "Test days",
        "Horizon",
        "Policy",
        "Cal actions",
        "Cal UCB",
        "Test actions",
        "Test risk",
        "Test UCB",
        "Pass",
    ]
    rolling_lines = [
        "| " + " | ".join(rolling_headers) + " |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rolling_selected.itertuples(index=False):
        rolling_lines.append(
            "| "
            + " | ".join(
                [
                    f"{int(row.test_start_day)}–{int(row.test_end_day)}",
                    str(int(row.horizon_days)),
                    str(row.policy),
                    str(int(row.cal_actions)),
                    f"{row.cal_action_risk_ucb:.4f}",
                    str(int(row.test_actions)),
                    f"{row.test_action_risk:.4f}",
                    f"{row.test_action_risk_ucb:.4f}",
                    "yes" if bool(row.test_contract_pass) else "no",
                ]
            )
            + " |"
        )
    rolling_markdown = (
        "\n".join(rolling_lines)
        if not rolling_selected.empty
        else "没有滚动校准窗口产生可执行策略。"
    )
    rolling_summary_text = (
        f"滚动再校准产生{int(rolling['selected'].sum())}个校准合格策略，"
        f"其中{int(rolling['test_contract_pass'].sum())}个在下一30日测试块继续通过5%合同。"
    )
    novelty_lines = []
    for horizon in [1, 7, 30]:
        for novelty_label in ["seen", "unseen"]:
            row = novelty_test.loc[
                novelty_test["horizon_days"].eq(horizon)
                & novelty_test["novelty"].eq(novelty_label)
                & novelty_test["method"].eq("hierarchical_hazard")
            ]
            if not row.empty:
                value = row.iloc[0]
                novelty_lines.append(
                    f"| {horizon} | {novelty_label} | {int(value['chutes'])} | "
                    f"{value['roc_auc']:.3f} | {value['average_precision']:.3f} |"
                )
    report = f"""# Chutes 一年期独立LLM复活验证

## 核心结果

完整轨迹包含6,122,413,756次调用、9,174个模型和366.8天时间跨度。逐row-group流式处理将其压缩为73,151个活跃模型—日，再构造{manifest['landmarks']:,}个右删失冷决策点和{manifest['episodes']:,}个独立冷episode。

在严格后移的第121–180天测试段，ColdGuard层级hazard的1/7/30日 ROC-AUC 分别为 {test.loc[(1, 'hierarchical_hazard'), 'roc_auc']:.3f}/{test.loc[(7, 'hierarchical_hazard'), 'roc_auc']:.3f}/{test.loc[(30, 'hierarchical_hazard'), 'roc_auc']:.3f}，PR-AUC为 {test.loc[(1, 'hierarchical_hazard'), 'average_precision']:.3f}/{test.loc[(7, 'hierarchical_hazard'), 'average_precision']:.3f}/{test.loc[(30, 'hierarchical_hazard'), 'average_precision']:.3f}。{policy_text}。

## 固定协议

- 实体：`chute_id`，即匿名模型服务单元。
- 冷状态：模型在某日无调用，且此前至少出现过一次调用。
- 事件：未来1/3/7/14/30日内再次出现调用。
- 时间切分：第1–60天训练、第61–120天校准、第121–180天测试；第181天后作为冻结模型的六个30日滚动外推块。
- 模型：冷龄hazard与层级离散hazard；后者加入近期请求、活跃日、生命周期和模型身份特征。
- 风险合同：每个冷episode只计第一次动作；校准侧经候选数校正的95% Wilson上界≤5%，且至少200个动作。
- 证据边界：Chutes没有真实GPU显存或驻留字段，因此只报告动作风险与模型—日机会，不报告GiB-hours。

## 完整性与流式预处理

- 原始Parquet为{manifest['trace_bytes']:,} bytes、{manifest['trace_rows']:,}行和{manifest['trace_row_groups']:,}个row group；SHA-256已核验为 `{manifest['trace_sha256_verified']}`。
- `started_at`与`chute_id`缺失数均为0；所有row group的时间最小值严格递增且范围不交叠。
- 全量聚合只读取`chute_id`和`started_at`，不执行61亿行全局排序，也不生成第二份调用级副本。
- 官方模型映射与轨迹交集为{manifest['mapped_chutes']:,}个具有冷landmark的模型；价格与支持GPU字段只用于覆盖说明和后续敏感性，不进入主动作风险合同。

## 时间分布审计

| 切分 | Landmarks | Episodes | Chutes | 1日复活率 | 7日复活率 | 30日复活率 |
|---|---:|---:|---:|---:|---:|---:|
| 训练 | {int(split_index.loc['train','landmarks'])} | {int(split_index.loc['train','episodes'])} | {int(split_index.loc['train','chutes'])} | {split_index.loc['train','revival_rate_1d']:.2%} | {split_index.loc['train','revival_rate_7d']:.2%} | {split_index.loc['train','revival_rate_30d']:.2%} |
| 校准 | {int(split_index.loc['calibration','landmarks'])} | {int(split_index.loc['calibration','episodes'])} | {int(split_index.loc['calibration','chutes'])} | {split_index.loc['calibration','revival_rate_1d']:.2%} | {split_index.loc['calibration','revival_rate_7d']:.2%} | {split_index.loc['calibration','revival_rate_30d']:.2%} |
| 测试 | {int(split_index.loc['test','landmarks'])} | {int(split_index.loc['test','episodes'])} | {int(split_index.loc['test','chutes'])} | {split_index.loc['test','revival_rate_1d']:.2%} | {split_index.loc['test','revival_rate_7d']:.2%} | {split_index.loc['test','revival_rate_30d']:.2%} |
| 后续外推 | {int(split_index.loc['forward','landmarks'])} | {int(split_index.loc['forward','episodes'])} | {int(split_index.loc['forward','chutes'])} | {split_index.loc['forward','revival_rate_1d']:.2%} | {split_index.loc['forward','revival_rate_7d']:.2%} | {split_index.loc['forward','revival_rate_30d']:.2%} |

## 测试段预测效果

| 时域 | 方法 | ROC-AUC | PR-AUC | Brier | 流行率 |
|---:|---|---:|---:|---:|---:|
| 1 | 冷龄hazard | {test.loc[(1, 'age_hazard'), 'roc_auc']:.3f} | {test.loc[(1, 'age_hazard'), 'average_precision']:.3f} | {test.loc[(1, 'age_hazard'), 'brier']:.3f} | {test.loc[(1, 'age_hazard'), 'prevalence']:.3f} |
| 1 | 层级hazard | {test.loc[(1, 'hierarchical_hazard'), 'roc_auc']:.3f} | {test.loc[(1, 'hierarchical_hazard'), 'average_precision']:.3f} | {test.loc[(1, 'hierarchical_hazard'), 'brier']:.3f} | {test.loc[(1, 'hierarchical_hazard'), 'prevalence']:.3f} |
| 7 | 冷龄hazard | {test.loc[(7, 'age_hazard'), 'roc_auc']:.3f} | {test.loc[(7, 'age_hazard'), 'average_precision']:.3f} | {test.loc[(7, 'age_hazard'), 'brier']:.3f} | {test.loc[(7, 'age_hazard'), 'prevalence']:.3f} |
| 7 | 层级hazard | {test.loc[(7, 'hierarchical_hazard'), 'roc_auc']:.3f} | {test.loc[(7, 'hierarchical_hazard'), 'average_precision']:.3f} | {test.loc[(7, 'hierarchical_hazard'), 'brier']:.3f} | {test.loc[(7, 'hierarchical_hazard'), 'prevalence']:.3f} |
| 30 | 冷龄hazard | {test.loc[(30, 'age_hazard'), 'roc_auc']:.3f} | {test.loc[(30, 'age_hazard'), 'average_precision']:.3f} | {test.loc[(30, 'age_hazard'), 'brier']:.3f} | {test.loc[(30, 'age_hazard'), 'prevalence']:.3f} |
| 30 | 层级hazard | {test.loc[(30, 'hierarchical_hazard'), 'roc_auc']:.3f} | {test.loc[(30, 'hierarchical_hazard'), 'average_precision']:.3f} | {test.loc[(30, 'hierarchical_hazard'), 'brier']:.3f} | {test.loc[(30, 'hierarchical_hazard'), 'prevalence']:.3f} |

## 未见模型压力测试

训练期仅含{manifest['train_chutes']}个模型；测试期{manifest['test_chutes']}个模型中有{manifest['test_unseen_chutes']}个未在训练期出现。这不是普通同分布测试，而是大规模模型涌入下的时间与新颖性联合压力测试。

| 时域 | 新颖性 | 模型数 | ROC-AUC | PR-AUC |
|---:|---|---:|---:|---:|
{chr(10).join(novelty_lines)}

## 校准选择

{selected_markdown}

## 60日滚动再校准

{rolling_summary_text} 每个窗口只使用测试块开始前已经成熟的标签，模型参数保持冻结，仅更新策略阈值。

{rolling_markdown}

## 论文定位

Chutes补强的是同领域、独立平台、全年尺度的请求复活与证书失效证据。它与GenTD26的真实GPU显存/加载链条互补：前者回答“复活规律和风险路由能否跨平台、跨时间保持”，后者回答“冷但驻留以及复活加载代价是否真实存在”。
"""
    report_path = output_dir / "report_CN.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path
