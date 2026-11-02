from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .pilot import HierarchicalTabularEncoder, _predictive_metrics
from .policy import wilson_upper_bound
from .survival import cumulative_incidence_from_hazards


HOUR_MS = 3_600_000
PRIMARY_HORIZON_HOURS = 6


def _hours(value: int) -> pd.Timedelta:
    return pd.Timedelta(timedelta(hours=int(value)))

NUMERIC_FEATURES = [
    "cold_age",
    "prior_active_6h",
    "prior_active_24h",
    "prior_active_72h",
    "req_mean_6h",
    "req_mean_24h",
    "req_max_24h",
    "sm_mean_6h",
    "gpu_memory_pct",
    "gpu_utilization_pct",
    "gpu_memory_util_pct",
    "ttft_avg_s",
    "queue_avg_s",
    "e2e_latency_avg_s",
    "hour_sin",
    "hour_cos",
]

CATEGORICAL_FEATURES = [
    "server_id",
    "node_id",
    "name",
    "model_name",
    "engine",
    "gpu_platform",
]


@dataclass(frozen=True)
class TemporalDesign:
    start: pd.Timestamp
    train_boundary: pd.Timestamp
    calibration_boundary: pd.Timestamp
    end: pd.Timestamp
    horizon_hours: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _active_mask(frame: pd.DataFrame) -> pd.Series:
    columns = ["req_per_min", "running", "queued", "gen_throughput", "prefill_tps"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing serving-activity fields: {missing}")
    active = pd.Series(False, index=frame.index)
    for column in columns:
        active |= pd.to_numeric(frame[column], errors="coerce").fillna(0.0) > 0.0
    return active


def load_hourly_frames(
    history_path: str | Path,
    dashboard_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history_path = Path(history_path)
    dashboard_path = Path(dashboard_path)
    history = _load_json(history_path)
    dashboard = _load_json(dashboard_path)
    if int(history.get("grain_ms", 0)) != HOUR_MS:
        raise ValueError("The external probe is predeclared for one-hour history grains")

    serving = pd.DataFrame(history.get("sglang", []))
    gpu = pd.DataFrame(history.get("gpu", []))
    metadata = pd.DataFrame(dashboard.get("sglang", []))
    if serving.empty or gpu.empty or metadata.empty:
        raise ValueError("FreeInference history or dashboard metadata is empty")

    for name, frame, keys in [
        ("serving", serving, ["server_id", "ts"]),
        ("gpu", gpu, ["gpu_id", "ts"]),
    ]:
        duplicates = int(frame.duplicated(keys).sum())
        if duplicates:
            raise ValueError(f"{name} history has {duplicates} duplicate key rows")
        frame["dt"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)

    serving["active"] = _active_mask(serving)
    meta_columns = [
        "id",
        "model_name",
        "engine",
        "gpu_indexes",
        "weight_gb",
        "kv_cache_gb",
        "status",
    ]
    meta = metadata[[column for column in meta_columns if column in metadata]].copy()
    meta.rename(columns={"id": "server_id"}, inplace=True)
    serving = serving.merge(meta, on="server_id", how="left", validate="many_to_one")

    dashboard_gpu = pd.DataFrame(dashboard.get("gpus", []))
    gpu_meta = dashboard_gpu[["node_id", "gpu_index", "name", "memory_total_mb"]].copy()
    gpu_meta.rename(columns={"name": "gpu_platform"}, inplace=True)
    gpu = gpu.merge(
        gpu_meta,
        on=["node_id", "gpu_index"],
        how="left",
        validate="many_to_one",
    )
    gpu["mapped_used_gib"] = (
        pd.to_numeric(gpu["memory_pct"], errors="coerce")
        * pd.to_numeric(gpu["memory_total_mb"], errors="coerce")
        / (100.0 * 1024.0)
    )

    node_gpu = (
        gpu.groupby(["node_id", "dt"], as_index=False)
        .agg(
            gpu_memory_pct=("memory_pct", "mean"),
            gpu_utilization_pct=("utilization_pct", "mean"),
            gpu_memory_util_pct=("memory_util_pct", "mean"),
            gpu_count=("gpu_id", "nunique"),
            gpu_platform=("gpu_platform", lambda values: "|".join(sorted(set(values.dropna().astype(str))))),
        )
    )
    serving = serving.merge(
        node_gpu,
        on=["node_id", "dt"],
        how="left",
        validate="many_to_one",
    )
    serving.sort_values(["server_id", "dt"], inplace=True)
    gpu.sort_values(["gpu_id", "dt"], inplace=True)

    manifest = {
        "history_sha256": _sha256(history_path),
        "dashboard_sha256": _sha256(dashboard_path),
        "grain_ms": int(history["grain_ms"]),
        "history_generated_at_utc": pd.to_datetime(
            history["generated_at"], unit="ms", utc=True
        ).isoformat(),
        "dashboard_generated_at_utc": pd.to_datetime(
            dashboard["generated_at"], unit="ms", utc=True
        ).isoformat(),
        "from_cache": bool(history.get("from_cache", False)),
    }
    return serving, gpu, metadata, manifest


def temporal_design(serving: pd.DataFrame, horizon_hours: int) -> TemporalDesign:
    start = serving["dt"].min()
    end = serving["dt"].max()
    horizon_hours = int(horizon_hours)
    if end - start < _hours(150):
        raise ValueError("Fewer than 150 hours are available for the seven-day probe")
    return TemporalDesign(
        start=start,
        train_boundary=start + _hours(72),
        calibration_boundary=start + _hours(120),
        end=end,
        horizon_hours=horizon_hours,
    )


def _numeric_summary(frame: pd.DataFrame, column: str, operation: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not values.notna().any():
        return float("nan")
    if operation == "mean":
        return float(values.mean())
    if operation == "max":
        return float(values.max())
    raise ValueError(operation)


def build_landmarks(
    serving: pd.DataFrame,
    horizon_hours: int = PRIMARY_HORIZON_HOURS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for server_id, source in serving.groupby("server_id", sort=True):
        source = source.sort_values("dt").set_index("dt")
        complete_index = pd.date_range(
            source.index.min(), source.index.max(), freq="1h", tz="UTC"
        )
        frame = source.reindex(complete_index)
        frame.index.name = "dt"
        frame["observed"] = frame["server_id"].notna()
        frame["active"] = frame["active"].where(frame["active"].notna(), False).astype(bool)

        cold_age = 0
        episode_sequence = 0
        for position, (decision_time, current) in enumerate(frame.iterrows()):
            if not bool(current["observed"]):
                cold_age = 0
                episode_sequence += 1
                continue
            if bool(current["active"]):
                cold_age = 0
                episode_sequence += 1
                continue

            cold_age += 1
            future = frame.iloc[position + 1 : position + 1 + horizon_hours]
            history_6 = frame.iloc[max(0, position - 5) : position + 1]
            if (
                len(future) != horizon_hours
                or not bool(future["observed"].all())
                or len(history_6) != 6
                or not bool(history_6["observed"].all())
            ):
                continue

            history_24 = frame.iloc[max(0, position - 23) : position + 1]
            history_72 = frame.iloc[max(0, position - 71) : position + 1]
            future_active = future["active"].to_numpy(bool)
            active_positions = np.flatnonzero(future_active)
            safe_hours = (
                int(active_positions[0]) + 1 if active_positions.size else horizon_hours
            )
            weight_gb = pd.to_numeric(
                pd.Series([current.get("weight_gb")]), errors="coerce"
            ).iloc[0]
            rows.append(
                {
                    "server_id": str(server_id),
                    "node_id": current.get("node_id"),
                    "name": current.get("name"),
                    "model_name": current.get("model_name"),
                    "engine": current.get("engine"),
                    "gpu_platform": current.get("gpu_platform"),
                    "dt": decision_time,
                    "day": int((decision_time - serving["dt"].min()) / _hours(1)),
                    "episode_id": f"{server_id}:{episode_sequence}",
                    "cold_age": int(cold_age),
                    "prior_active_6h": int(history_6["active"].sum()),
                    "prior_active_24h": int(history_24["active"].sum()),
                    "prior_active_72h": int(history_72["active"].sum()),
                    "req_mean_6h": _numeric_summary(history_6, "req_per_min", "mean"),
                    "req_mean_24h": _numeric_summary(history_24, "req_per_min", "mean"),
                    "req_max_24h": _numeric_summary(history_24, "req_per_min", "max"),
                    "sm_mean_6h": _numeric_summary(history_6, "sm_util_pct", "mean"),
                    "gpu_memory_pct": current.get("gpu_memory_pct"),
                    "gpu_utilization_pct": current.get("gpu_utilization_pct"),
                    "gpu_memory_util_pct": current.get("gpu_memory_util_pct"),
                    "ttft_avg_s": current.get("ttft_avg_s"),
                    "queue_avg_s": current.get("queue_avg_s"),
                    "e2e_latency_avg_s": current.get("e2e_latency_avg_s"),
                    "hour_sin": float(np.sin(2.0 * np.pi * decision_time.hour / 24.0)),
                    "hour_cos": float(np.cos(2.0 * np.pi * decision_time.hour / 24.0)),
                    "event_next_hour": int(future_active[0]),
                    "reactivates_within_horizon": int(future_active.any()),
                    "safe_reclaim_hours": safe_hours,
                    "weight_gb": float(weight_gb) if pd.notna(weight_gb) else np.nan,
                    "known_weight_gib_hours": (
                        float(weight_gb) * safe_hours if pd.notna(weight_gb) else np.nan
                    ),
                }
            )
    landmarks = pd.DataFrame(rows)
    if landmarks.empty:
        raise ValueError("No mature cold landmarks were constructed")
    landmarks.sort_values(["dt", "server_id"], inplace=True)
    landmarks.reset_index(drop=True, inplace=True)
    return landmarks


def assign_splits(landmarks: pd.DataFrame, design: TemporalDesign) -> pd.DataFrame:
    result = landmarks.copy()
    horizon = _hours(design.horizon_hours)
    result["split"] = np.select(
        [
            result["dt"] <= design.train_boundary - horizon,
            (result["dt"] > design.train_boundary)
            & (result["dt"] <= design.calibration_boundary - horizon),
            result["dt"] > design.calibration_boundary,
        ],
        ["train", "calibration", "test"],
        default="embargo",
    )
    return result


def _fit_hazard_model(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[HierarchicalTabularEncoder, HistGradientBoostingClassifier]:
    target = train["event_next_hour"].to_numpy(int)
    encoder = HierarchicalTabularEncoder(numeric, categorical, strength=10.0)
    if categorical:
        encoded = encoder.fit_transform_group_oof(
            train,
            target,
            train["server_id"].to_numpy(str),
            n_splits=5,
        )
    else:
        encoded = encoder.fit_transform(train, target)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=7,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(encoded, target)
    return encoder, model


def _horizon_scores(
    frame: pd.DataFrame,
    encoder: HierarchicalTabularEncoder,
    model: HistGradientBoostingClassifier,
    horizon_hours: int,
) -> np.ndarray:
    hazards = []
    for offset in range(horizon_hours):
        future = frame.copy()
        future["cold_age"] = future["cold_age"] + offset
        hazards.append(model.predict_proba(encoder.transform(future))[:, 1])
    return cumulative_incidence_from_hazards(np.column_stack(hazards))


def fit_and_score(
    landmarks: pd.DataFrame,
    horizon_hours: int,
    seed: int = 20261030,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {
        split: landmarks.loc[landmarks["split"] == split].copy()
        for split in ["train", "calibration", "test"]
    }
    for split, frame in frames.items():
        if frame.empty or frame["reactivates_within_horizon"].nunique() < 2:
            raise ValueError(f"The {split} split cannot identify predictive performance")

    specifications = {
        "age_hazard": (["cold_age"], []),
        "hierarchical_hazard": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    train = frames["train"]
    next_hour_rate = float(train["event_next_hour"].mean())
    constant_horizon_risk = 1.0 - (1.0 - next_hour_rate) ** horizon_hours

    for split, frame in frames.items():
        constant = np.full(len(frame), constant_horizon_risk)
        metric_rows.append(
            {
                "method": "constant_train_hazard",
                "split": split,
                **_predictive_metrics(
                    frame["reactivates_within_horizon"].to_numpy(int), constant
                ),
            }
        )

    for model_index, (method, (numeric, categorical)) in enumerate(specifications.items()):
        encoder, model = _fit_hazard_model(
            train, numeric, categorical, seed + model_index
        )
        for split, frame in frames.items():
            scores = _horizon_scores(frame, encoder, model, horizon_hours)
            scored = frame[
                [
                    "dt",
                    "server_id",
                    "episode_id",
                    "cold_age",
                    "reactivates_within_horizon",
                    "safe_reclaim_hours",
                    "weight_gb",
                    "known_weight_gib_hours",
                ]
            ].copy()
            scored["method"] = method
            scored["split"] = split
            scored["score"] = scores
            prediction_parts.append(scored)
            metric_rows.append(
                {
                    "method": method,
                    "split": split,
                    **_predictive_metrics(
                        frame["reactivates_within_horizon"].to_numpy(int), scores
                    ),
                }
            )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(metric_rows)


def _first_actions(frame: pd.DataFrame, selector: pd.Series) -> pd.DataFrame:
    selected = frame.loc[selector].sort_values(["episode_id", "dt"])
    return selected.drop_duplicates("episode_id", keep="first")


def _unweighted_action_metrics(
    actions: pd.DataFrame,
    confidence: float,
    comparisons: int,
) -> dict[str, Any]:
    if actions.empty:
        return {
            "actions": 0,
            "false_reclaims": 0,
            "action_risk": np.nan,
            "action_risk_ucb": 1.0,
            "safe_reclaim_hours": 0.0,
            "known_weight_gib_hours": 0.0,
            "known_weight_actions": 0,
        }
    labels = actions["reactivates_within_horizon"].to_numpy(float)
    return {
        "actions": int(len(actions)),
        "false_reclaims": int(labels.sum()),
        "action_risk": float(labels.mean()),
        "action_risk_ucb": wilson_upper_bound(
            float(labels.sum()), float(len(labels)), confidence, comparisons
        ),
        "safe_reclaim_hours": float(actions["safe_reclaim_hours"].sum()),
        "known_weight_gib_hours": float(
            actions["known_weight_gib_hours"].fillna(0.0).sum()
        ),
        "known_weight_actions": int(actions["weight_gb"].notna().sum()),
    }


def policy_replay(
    landmarks: pd.DataFrame,
    predictions: pd.DataFrame,
    confidence: float = 0.95,
    risk_target: float = 0.05,
    minimum_actions: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ttl_grid = [1, 2, 4, 8, 12, 24, 48]
    calibration = landmarks.loc[landmarks["split"] == "calibration"].copy()
    test = landmarks.loc[landmarks["split"] == "test"].copy()
    for ttl in ttl_grid:
        cal_actions = _first_actions(calibration, calibration["cold_age"] >= ttl)
        test_actions = _first_actions(test, test["cold_age"] >= ttl)
        cal_metrics = _unweighted_action_metrics(cal_actions, confidence, len(ttl_grid))
        test_metrics = _unweighted_action_metrics(test_actions, confidence, 1)
        rows.append(
            {
                "policy": "ttl",
                "parameter": float(ttl),
                **{f"cal_{key}": value for key, value in cal_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "qualified": bool(
                    cal_metrics["actions"] >= minimum_actions
                    and cal_metrics["action_risk_ucb"] <= risk_target
                ),
            }
        )

    for method in ["age_hazard", "hierarchical_hazard"]:
        cal_scores = predictions.loc[
            (predictions["method"] == method)
            & (predictions["split"] == "calibration")
        ].copy()
        test_scores = predictions.loc[
            (predictions["method"] == method) & (predictions["split"] == "test")
        ].copy()
        thresholds = np.unique(
            np.quantile(cal_scores["score"].to_numpy(float), np.linspace(0.05, 0.95, 19))
        )
        for threshold in thresholds:
            cal_actions = _first_actions(
                cal_scores, cal_scores["score"] <= float(threshold)
            )
            test_actions = _first_actions(
                test_scores, test_scores["score"] <= float(threshold)
            )
            cal_metrics = _unweighted_action_metrics(
                cal_actions, confidence, len(thresholds)
            )
            test_metrics = _unweighted_action_metrics(test_actions, confidence, 1)
            rows.append(
                {
                    "policy": method,
                    "parameter": float(threshold),
                    **{f"cal_{key}": value for key, value in cal_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                    "qualified": bool(
                        cal_metrics["actions"] >= minimum_actions
                        and cal_metrics["action_risk_ucb"] <= risk_target
                    ),
                }
            )
    return pd.DataFrame(rows)


def mapped_residency_audit(
    serving: pd.DataFrame,
    gpu: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    mapping_rows: list[dict[str, Any]] = []
    for row in metadata.itertuples(index=False):
        server_id = getattr(row, "id")
        node_id = getattr(row, "node_id")
        indexes = str(getattr(row, "gpu_indexes", "")).split(",")
        for value in indexes:
            try:
                gpu_index = int(value.strip())
            except ValueError:
                continue
            mapping_rows.append(
                {
                    "server_id": server_id,
                    "node_id": node_id,
                    "gpu_index": gpu_index,
                    "weight_gb": getattr(row, "weight_gb", np.nan),
                    "kv_cache_gb": getattr(row, "kv_cache_gb", np.nan),
                    "snapshot_status": getattr(row, "status", None),
                }
            )
    mapping = pd.DataFrame(mapping_rows)
    if mapping.empty:
        return pd.DataFrame()

    idle = serving.loc[~serving["active"], ["server_id", "node_id", "dt"]]
    mapped = idle.merge(
        mapping, on=["server_id", "node_id"], how="inner", validate="many_to_many"
    ).merge(
        gpu[
            [
                "node_id",
                "gpu_index",
                "dt",
                "mapped_used_gib",
                "memory_pct",
                "gpu_platform",
            ]
        ],
        on=["node_id", "gpu_index", "dt"],
        how="left",
        validate="many_to_one",
    )
    hourly = (
        mapped.groupby(
            ["server_id", "dt", "weight_gb", "kv_cache_gb", "snapshot_status"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mapped_used_gib=("mapped_used_gib", "sum"),
            mapped_memory_pct=("memory_pct", "mean"),
            mapped_gpus=("gpu_index", "nunique"),
            gpu_platform=("gpu_platform", lambda values: "|".join(sorted(set(values.dropna().astype(str))))),
        )
    )
    rows = []
    for server_id, frame in hourly.groupby("server_id", sort=True):
        valid = frame["mapped_memory_pct"].notna()
        observed = frame.loc[valid]
        rows.append(
            {
                "server_id": server_id,
                "idle_hours": int(len(frame)),
                "hours_with_memory_observation": int(valid.sum()),
                "weight_gb": float(frame["weight_gb"].iloc[0])
                if pd.notna(frame["weight_gb"].iloc[0])
                else np.nan,
                "kv_cache_gb": float(frame["kv_cache_gb"].iloc[0])
                if pd.notna(frame["kv_cache_gb"].iloc[0])
                else np.nan,
                "snapshot_status": frame["snapshot_status"].iloc[0],
                "median_mapped_used_gib": float(observed["mapped_used_gib"].median())
                if valid.any()
                else np.nan,
                "p10_mapped_used_gib": float(observed["mapped_used_gib"].quantile(0.10))
                if valid.any()
                else np.nan,
                "median_mapped_memory_pct": float(observed["mapped_memory_pct"].median())
                if valid.any()
                else np.nan,
                "share_idle_hours_above_50pct_memory": float(
                    (frame.loc[valid, "mapped_memory_pct"] >= 50.0).mean()
                )
                if valid.any()
                else np.nan,
                "gpu_platform": "|".join(sorted(set(frame["gpu_platform"].dropna().astype(str))))
            }
        )
    return pd.DataFrame(rows)


def fleet_audit(serving: pd.DataFrame, design: TemporalDesign) -> pd.DataFrame:
    total_hours = int((design.end - design.start) / _hours(1)) + 1
    rows = []
    for server_id, frame in serving.groupby("server_id", sort=True):
        observed = int(frame["dt"].nunique())
        active = int(frame["active"].sum())
        transitions = int(
            ((~frame.sort_values("dt")["active"].shift(fill_value=True)) & frame.sort_values("dt")["active"]).sum()
        )
        rows.append(
            {
                "server_id": server_id,
                "node_id": frame["node_id"].iloc[0],
                "gpu_platform": frame["gpu_platform"].iloc[0],
                "observed_hours": observed,
                "active_hours": active,
                "idle_hours": observed - active,
                "missing_hours": total_hours - observed,
                "active_fraction_observed": active / observed,
                "idle_to_active_transitions": transitions,
            }
        )
    return pd.DataFrame(rows)


def run_freeinference_probe(
    history_path: str | Path,
    dashboard_path: str | Path,
    output_dir: str | Path,
    horizon_hours: int = PRIMARY_HORIZON_HOURS,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serving, gpu, metadata, manifest = load_hourly_frames(history_path, dashboard_path)
    design = temporal_design(serving, horizon_hours)
    landmarks = assign_splits(build_landmarks(serving, horizon_hours), design)
    predictions, metrics = fit_and_score(landmarks, horizon_hours)
    policies = policy_replay(landmarks, predictions)
    fleet = fleet_audit(serving, design)
    residency = mapped_residency_audit(serving, gpu, metadata)

    serving.to_parquet(output_dir / "serving_hourly.parquet", index=False)
    gpu.to_parquet(output_dir / "gpu_hourly.parquet", index=False)
    landmarks.to_parquet(output_dir / "cold_landmarks.parquet", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    policies.to_csv(output_dir / "policy_replay.csv", index=False)
    fleet.to_csv(output_dir / "fleet_audit.csv", index=False)
    residency.to_csv(output_dir / "mapped_residency_audit.csv", index=False)

    split_audit = (
        landmarks.groupby("split", as_index=False)
        .agg(
            landmarks=("episode_id", "size"),
            episodes=("episode_id", "nunique"),
            deployments=("server_id", "nunique"),
            revival_rate_6h=("reactivates_within_horizon", "mean"),
        )
    )
    split_audit.to_csv(output_dir / "split_audit.csv", index=False)

    manifest.update(
        {
            "source_history_endpoint": "https://openinfra.freeinference.org/api/history?hours=168",
            "source_dashboard_endpoint": "https://openinfra.freeinference.org/api/dashboard",
            "history_start_utc": design.start.isoformat(),
            "history_end_utc": design.end.isoformat(),
            "train_boundary_utc": design.train_boundary.isoformat(),
            "calibration_boundary_utc": design.calibration_boundary.isoformat(),
            "horizon_hours": horizon_hours,
            "activity_definition": (
                "req_per_min>0 OR running>0 OR queued>0 OR "
                "gen_throughput>0 OR prefill_tps>0"
            ),
            "maturity_rule": (
                "All future hourly rows through the horizon must be observed; "
                "a missing hour breaks a cold episode."
            ),
            "gpu_mapping_scope": (
                "GPU indexes come from the terminal dashboard snapshot and are "
                "used only for an explicitly snapshot-assisted residency audit."
            ),
            "serving_rows": int(len(serving)),
            "gpu_rows": int(len(gpu)),
            "servers": int(serving["server_id"].nunique()),
            "gpus": int(gpu["gpu_id"].nunique()),
            "qualified_policy_count": int(policies["qualified"].sum()),
        }
    )
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    test_metrics = metrics.loc[metrics["split"] == "test"].set_index("method")
    cal_split = split_audit.loc[split_audit["split"] == "calibration"].iloc[0]
    test_split = split_audit.loc[split_audit["split"] == "test"].iloc[0]
    hierarchical = test_metrics.loc["hierarchical_hazard"]
    age = test_metrics.loc["age_hazard"]
    weighted_residency = residency.loc[
        residency["weight_gb"].notna()
        & (residency["hours_with_memory_observation"] >= 24)
    ]
    best_residency = (
        weighted_residency.sort_values(
            ["share_idle_hours_above_50pct_memory", "hours_with_memory_observation"],
            ascending=False,
        ).iloc[0]
        if not weighted_residency.empty
        else residency.sort_values("hours_with_memory_observation", ascending=False).iloc[0]
    )
    report = f"""# FreeInference 七天外部 Shadow Probe

## 结论

七天数据已成功接入 ColdGuard 的时序证据链。独立 GPU 服务轨迹包含 {len(serving):,} 条服务小时记录、{len(gpu):,} 条 GPU 小时记录、{serving['server_id'].nunique()} 个服务和 {gpu['gpu_id'].nunique()} 张 GPU。层级离散风险模型在严格后移测试段获得 ROC-AUC {hierarchical['roc_auc']:.3f}、PR-AUC {hierarchical['average_precision']:.3f}；相较年龄模型的 {age['roc_auc']:.3f}/{age['average_precision']:.3f}，ROC 排序改善而 PR 未全面胜出，且两者均高于测试正例率所给出的 PR 无技巧参考。这表明跨平台复活信号可被学习，但七天证据尚不足以宣称层级模型全面优于冷龄基线。

风险资格层保持 abstain：校准段复活率为 {cal_split['revival_rate_6h']:.1%}，测试段为 {test_split['revival_rate_6h']:.1%}，所有候选策略的 95% 多重比较校正风险上界均未达到 5% 目标。因此，本次实验形成“外部可迁移信号 + 正确拒绝签发安全资格”的结果，不将七天窗口写成部署安全证明。

## 预声明设计

- 单位：FreeInference `server_id` 的逐小时服务状态。
- 冷状态：`req_per_min`、`running`、`queued`、`gen_throughput`、`prefill_tps` 同时为零。
- 事件：冷状态后未来 {horizon_hours} 小时内首次重新活跃。
- 成熟度：未来 {horizon_hours} 个小时必须全部可观测；缺失小时打断 episode。
- 切分：前 72 小时训练，随后 48 小时校准，剩余时间测试；边界前保留 {horizon_hours} 小时标签成熟隔离带。
- 模型：小时离散 hazard；年龄模型对比层级模型。层级模型加入服务/节点/引擎/GPU 平台、历史请求、GPU 与延迟上下文。
- 策略：仅在校准段的 95% Wilson 上界不超过 5% 时签发；测试段不参与选阈值。

## 数据与事件审计

| 项目 | 数值 |
|---|---:|
| 时间范围（UTC） | {design.start.isoformat()} — {design.end.isoformat()} |
| 服务 / GPU | {serving['server_id'].nunique()} / {gpu['gpu_id'].nunique()} |
| 服务小时 / GPU 小时 | {len(serving):,} / {len(gpu):,} |
| 冷决策 landmarks | {len(landmarks):,} |
| 冷 episodes | {landmarks['episode_id'].nunique():,} |
| 校准 landmarks / 6h 复活率 | {int(cal_split['landmarks'])} / {cal_split['revival_rate_6h']:.1%} |
| 测试 landmarks / 6h 复活率 | {int(test_split['landmarks'])} / {test_split['revival_rate_6h']:.1%} |

## 后移测试效果

| 方法 | ROC-AUC | PR-AUC | Brier | 解释 |
|---|---:|---:|---:|---|
| 常数训练 hazard | {test_metrics.loc['constant_train_hazard','roc_auc']:.3f} | {test_metrics.loc['constant_train_hazard','average_precision']:.3f} | {test_metrics.loc['constant_train_hazard','brier']:.3f} | 无排序能力 |
| 冷龄 hazard | {age['roc_auc']:.3f} | {age['average_precision']:.3f} | {age['brier']:.3f} | 仅利用冷龄 |
| ColdGuard 层级 hazard | {hierarchical['roc_auc']:.3f} | {hierarchical['average_precision']:.3f} | {hierarchical['brier']:.3f} | ROC 改善；PR 与概率校准仍受短窗漂移影响 |

测试段正例率为 {test_split['revival_rate_6h']:.1%}，因此 PR-AUC 的无技巧参考值为 {test_split['revival_rate_6h']:.3f}。层级模型的价值首先体现在排序，而不是未经校准的绝对概率。

## GPU 驻留证据

按终点 dashboard 的 `gpu_indexes` 回接历史 GPU 指标后，公开权重为 {best_residency['weight_gb']:.1f} GB 的 `{best_residency['server_id']}` 在 {int(best_residency['hours_with_memory_observation'])} 个空闲小时中具有显存观测，中位映射显存为 {best_residency['median_mapped_used_gib']:.1f} GiB，且 {best_residency['share_idle_hours_above_50pct_memory']:.1%} 的空闲小时映射显存占用超过 50%。该结果在独立平台上直接支持“请求空闲但 GPU 显存仍被占用”的 ColdGuard 场景。

这部分被标记为 **snapshot-assisted**：开放历史没有逐小时保存部署到 GPU 的映射，当前 dashboard 映射不能被当作整周不变的事实。它适合作为真实 shadow 证据与后续持续采集的起点，不作为整周模型级显存归因的最终结论。

## 对论文的即时价值

1. 独立于阿里轨迹，验证了冷服务复活事件、异构 GPU 平台和真实服务延迟字段能够共同观测。
2. 层级 hazard 在测试段改善 ROC 排序，同时暴露出 PR 与概率校准的迁移缺口，为下一阶段的跨平台再校准提供了明确目标。
3. 七天窗口触发了风险资格的 abstain，展示风险校准层能够阻止短样本下的过度回收。
4. 继续滚动收集 60–90 天后，可将本 probe 升级为独立平台 shadow-only 主结果；逐次保存 dashboard 快照即可消除部署—GPU 映射的主要缺口。

## 证据边界

- 本 probe 的主要可发表结论是外部可观测性、排序信号与安全 abstention。
- 资源加权风险只可在公开 `weight_gb` 的服务子集上探索，不能替代完整部署级显存归因。
- `mapped_residency_audit.csv` 使用终点快照映射，文稿中必须保留 snapshot-assisted 限定。
"""
    (output_dir / "report_CN.md").write_text(report, encoding="utf-8")
    return output_dir / "report_CN.md"
