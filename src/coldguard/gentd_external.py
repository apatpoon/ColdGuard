from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .pilot import HierarchicalTabularEncoder, _predictive_metrics
from .policy import effective_sample_size, wilson_upper_bound


HORIZON_SECONDS = 3600
MIN_COLD_SECONDS = 1800
MAX_GAP_SECONDS = 120
GPU_DUTY_THRESHOLD = 1.0
MEMORY_THRESHOLD_GIB = 10.0

NUMERIC_FEATURES = [
    "cold_age_min",
    "memory_gib",
    "memory_mean_30",
    "memory_change_30",
    "duty_mean_5",
    "duty_mean_30",
    "active_fraction_30",
    "active_fraction_60",
    "prior_cold_episodes",
    "hour_sin",
    "hour_cos",
]
CATEGORICAL_FEATURES = ["container_ip"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_aligned_gpu(extracted_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    extracted_root = Path(extracted_root)
    duty_path = extracted_root / "pod_gpu_duty_cycle_anon.csv"
    memory_path = extracted_root / "pod_gpu_memory_used_bytes_anon.csv"
    duty = pd.read_csv(duty_path).rename(columns={"value": "gpu_duty"})
    memory = pd.read_csv(memory_path).rename(columns={"value": "gpu_memory_bytes"})
    for name, frame in [("duty", duty), ("memory", memory)]:
        if frame.duplicated(["container_ip", "timestamp_anon"]).any():
            raise ValueError(f"Duplicate {name} keys")
    aligned = duty.merge(
        memory,
        on=["container_ip", "timestamp_anon"],
        how="inner",
        validate="one_to_one",
    )
    aligned["memory_gib"] = aligned["gpu_memory_bytes"] / (1024.0**3)
    aligned.sort_values(["container_ip", "timestamp_anon"], inplace=True)
    aligned.reset_index(drop=True, inplace=True)
    manifest = {
        "duty_rows": int(len(duty)),
        "memory_rows": int(len(memory)),
        "aligned_rows": int(len(aligned)),
        "containers": int(aligned["container_ip"].nunique()),
        "start_timestamp_anon": float(aligned["timestamp_anon"].min()),
        "end_timestamp_anon": float(aligned["timestamp_anon"].max()),
        "span_hours": float(
            (aligned["timestamp_anon"].max() - aligned["timestamp_anon"].min()) / 3600.0
        ),
        "duty_sha256": _sha256(duty_path),
        "memory_sha256": _sha256(memory_path),
    }
    return aligned, manifest


def _rolling_mean(values: np.ndarray, index: int, width: int) -> float:
    start = max(0, index - width + 1)
    return float(np.nanmean(values[start : index + 1]))


def build_gentd_landmarks(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for container, group in aligned.groupby("container_ip", sort=True):
        group = group.sort_values("timestamp_anon").reset_index(drop=True)
        timestamps = group["timestamp_anon"].to_numpy(float)
        duty = group["gpu_duty"].to_numpy(float)
        memory = group["memory_gib"].to_numpy(float)
        cold = (duty <= GPU_DUTY_THRESHOLD) & (memory >= MEMORY_THRESHOLD_GIB)
        observed_steps = np.diff(timestamps)
        observed_steps = observed_steps[
            (observed_steps > 0) & (observed_steps <= MAX_GAP_SECONDS)
        ]
        sampling_step = float(np.median(observed_steps)) if observed_steps.size else 60.0
        episode_sequence = 0
        episode_start_index: int | None = None
        last_decision_timestamp = -np.inf

        for index in range(len(group)):
            gap_from_previous = (
                timestamps[index] - timestamps[index - 1] if index > 0 else np.inf
            )
            starts_episode = bool(
                cold[index]
                and (
                    index == 0
                    or not cold[index - 1]
                    or gap_from_previous > MAX_GAP_SECONDS
                )
            )
            if starts_episode:
                episode_sequence += 1
                episode_start_index = index
                last_decision_timestamp = -np.inf
            if not cold[index]:
                episode_start_index = None
                continue
            assert episode_start_index is not None
            age_seconds = (
                timestamps[index] - timestamps[episode_start_index] + sampling_step
            )
            if age_seconds < MIN_COLD_SECONDS:
                continue
            # Use roughly five-minute decision grains to limit landmark dependence.
            if timestamps[index] - last_decision_timestamp < 300.0:
                continue

            horizon_end = timestamps[index] + HORIZON_SECONDS
            end_position = int(np.searchsorted(timestamps, horizon_end, side="right"))
            future_end = min(len(group), end_position)
            future_timestamps = timestamps[index + 1 : future_end]
            future_duty = duty[index + 1 : future_end]
            if future_timestamps.size == 0:
                continue
            gap_series = np.diff(np.concatenate(([timestamps[index]], future_timestamps)))
            break_positions = np.flatnonzero(gap_series > MAX_GAP_SECONDS)
            valid_length = int(break_positions[0]) if break_positions.size else len(future_timestamps)
            valid_times = future_timestamps[:valid_length]
            valid_duty = future_duty[:valid_length]
            active_positions = np.flatnonzero(valid_duty > GPU_DUTY_THRESHOLD)
            if active_positions.size:
                revival_timestamp = float(valid_times[active_positions[0]])
                reactivates = int(revival_timestamp <= horizon_end)
                mature = True
            else:
                revival_timestamp = np.nan
                reactivates = 0
                mature = bool(
                    valid_times.size
                    and valid_times[-1] >= horizon_end - MAX_GAP_SECONDS
                )
            if not mature:
                continue
            last_decision_timestamp = timestamps[index]

            safe_seconds = (
                min(HORIZON_SECONDS, revival_timestamp - timestamps[index])
                if reactivates
                else HORIZON_SECONDS
            )
            timestamp = pd.to_datetime(timestamps[index], unit="s", utc=True)
            row = {
                "episode_id": f"{container}|{episode_sequence}",
                "container_ip": container,
                "cold_start_timestamp": float(timestamps[episode_start_index]),
                "decision_timestamp": float(timestamps[index]),
                "decision_time": timestamp,
                "cold_age_min": float(age_seconds / 60.0),
                "reactivates_within_horizon": reactivates,
                "revival_timestamp": revival_timestamp,
                "memory_gib": float(memory[index]),
                "potential_reclaim_gib_hours": float(memory[index]),
                "safe_reclaim_gib_hours": float(memory[index] * safe_seconds / 3600.0),
                "memory_mean_30": _rolling_mean(memory, index, 30),
                "memory_change_30": float(memory[index] - memory[max(0, index - 29)]),
                "duty_mean_5": _rolling_mean(duty, index, 5),
                "duty_mean_30": _rolling_mean(duty, index, 30),
                "active_fraction_30": _rolling_mean(
                    (duty > GPU_DUTY_THRESHOLD).astype(float), index, 30
                ),
                "active_fraction_60": _rolling_mean(
                    (duty > GPU_DUTY_THRESHOLD).astype(float), index, 60
                ),
                "prior_cold_episodes": int(episode_sequence - 1),
                "hour_sin": float(np.sin(2 * np.pi * timestamp.hour / 24.0)),
                "hour_cos": float(np.cos(2 * np.pi * timestamp.hour / 24.0)),
            }
            rows.append(row)

    landmarks = pd.DataFrame(rows)
    if landmarks.empty:
        raise ValueError("No mature GenTD26 landmarks were constructed")
    landmarks.sort_values(["decision_timestamp", "container_ip"], inplace=True)
    landmarks.reset_index(drop=True, inplace=True)
    episodes = (
        landmarks.groupby(["episode_id", "container_ip"], as_index=False)
        .agg(
            cold_start_timestamp=("cold_start_timestamp", "min"),
            first_eligible_timestamp=("decision_timestamp", "min"),
            revival_timestamp=("revival_timestamp", "min"),
            reactivates_within_horizon=("reactivates_within_horizon", "max"),
        )
    )
    return landmarks, episodes


def assign_splits(landmarks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = landmarks.copy()
    start = float(result["decision_timestamp"].min())
    end = float(result["decision_timestamp"].max())
    span = end - start
    train_boundary = start + 0.50 * span
    calibration_boundary = start + 0.75 * span
    result["split"] = np.select(
        [
            result["decision_timestamp"] <= train_boundary - HORIZON_SECONDS,
            (result["decision_timestamp"] > train_boundary)
            & (
                result["decision_timestamp"]
                <= calibration_boundary - HORIZON_SECONDS
            ),
            result["decision_timestamp"] > calibration_boundary,
        ],
        ["train", "calibration", "test"],
        default="embargo",
    )
    operational = result.loc[result["split"].isin(["train", "calibration", "test"])]
    cross_split = (
        operational.groupby("episode_id")["split"].nunique().loc[lambda values: values > 1]
    )
    return result, {
        "train_boundary_timestamp_anon": train_boundary,
        "calibration_boundary_timestamp_anon": calibration_boundary,
        "horizon_seconds": HORIZON_SECONDS,
        "minimum_cold_seconds": MIN_COLD_SECONDS,
        "gpu_duty_threshold": GPU_DUTY_THRESHOLD,
        "memory_threshold_gib": MEMORY_THRESHOLD_GIB,
        "cross_split_episode_count": int(len(cross_split)),
    }


def _fit_model(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[HierarchicalTabularEncoder, HistGradientBoostingClassifier]:
    target = train["reactivates_within_horizon"].to_numpy(int)
    encoder = HierarchicalTabularEncoder(numeric, categorical, strength=40.0)
    if categorical:
        encoded = encoder.fit_transform_group_oof(
            train,
            target,
            train["container_ip"].to_numpy(str),
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


def fit_predictions(landmarks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {
        split: landmarks.loc[landmarks["split"] == split].copy()
        for split in ["train", "calibration", "test"]
    }
    for split, frame in frames.items():
        if frame.empty or frame["reactivates_within_horizon"].nunique() < 2:
            raise ValueError(f"{split} split is not identifiable")
    specifications = {
        "cold_age": (["cold_age_min"], []),
        "hierarchical": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    prevalence = float(frames["train"]["reactivates_within_horizon"].mean())
    for split, frame in frames.items():
        metric_rows.append(
            {
                "method": "constant_train_rate",
                "split": split,
                **_predictive_metrics(
                    frame["reactivates_within_horizon"].to_numpy(int),
                    np.full(len(frame), prevalence),
                ),
            }
        )
    for index, (method, (numeric, categorical)) in enumerate(specifications.items()):
        encoder, model = _fit_model(frames["train"], numeric, categorical, 20261030 + index)
        for split, frame in frames.items():
            scores = model.predict_proba(encoder.transform(frame))[:, 1]
            scored = frame[
                [
                    "episode_id",
                    "container_ip",
                    "decision_time",
                    "cold_age_min",
                    "reactivates_within_horizon",
                    "memory_gib",
                    "potential_reclaim_gib_hours",
                    "safe_reclaim_gib_hours",
                ]
            ].copy()
            scored["method"] = method
            scored["split"] = split
            scored["score"] = scores
            prediction_rows.append(scored)
            metric_rows.append(
                {
                    "method": method,
                    "split": split,
                    **_predictive_metrics(
                        frame["reactivates_within_horizon"].to_numpy(int), scores
                    ),
                }
            )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows)


def _first_actions(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    selected = frame.loc[mask].sort_values(["episode_id", "decision_time"])
    return selected.drop_duplicates("episode_id", keep="first")


def _action_metrics(
    actions: pd.DataFrame, confidence: float, comparisons: int
) -> dict[str, Any]:
    if actions.empty:
        return {
            "actions": 0,
            "false_reclaims": 0,
            "action_risk": np.nan,
            "action_risk_ucb": 1.0,
            "resource_risk": np.nan,
            "resource_risk_ucb": 1.0,
            "reclaimed_gib_hours": 0.0,
        }
    labels = actions["reactivates_within_horizon"].to_numpy(float)
    weights = actions["potential_reclaim_gib_hours"].to_numpy(float)
    weighted_risk = float(np.dot(labels, weights) / weights.sum())
    effective = effective_sample_size(weights)
    return {
        "actions": int(len(actions)),
        "false_reclaims": int(labels.sum()),
        "action_risk": float(labels.mean()),
        "action_risk_ucb": wilson_upper_bound(
            float(labels.sum()), float(len(labels)), confidence, comparisons
        ),
        "resource_risk": weighted_risk,
        "resource_risk_ucb": wilson_upper_bound(
            weighted_risk * effective, effective, confidence, comparisons
        ),
        "reclaimed_gib_hours": float(actions["safe_reclaim_gib_hours"].sum()),
    }


def replay_policies(
    landmarks: pd.DataFrame,
    predictions: pd.DataFrame,
    target_risk: float = 0.05,
    minimum_actions: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calibration = landmarks.loc[landmarks["split"] == "calibration"].copy()
    test = landmarks.loc[landmarks["split"] == "test"].copy()
    ttl_grid = [30, 60, 120, 240]
    for ttl in ttl_grid:
        cal = _first_actions(calibration, calibration["cold_age_min"] >= ttl)
        tst = _first_actions(test, test["cold_age_min"] >= ttl)
        cm = _action_metrics(cal, 0.95, len(ttl_grid))
        tm = _action_metrics(tst, 0.95, 1)
        rows.append(
            {
                "policy": "ttl",
                "parameter": float(ttl),
                **{f"cal_{key}": value for key, value in cm.items()},
                **{f"test_{key}": value for key, value in tm.items()},
                "qualified": bool(
                    cm["actions"] >= minimum_actions
                    and cm["action_risk_ucb"] <= target_risk
                    and cm["resource_risk_ucb"] <= target_risk
                ),
            }
        )
    for method in ["cold_age", "hierarchical"]:
        cal_frame = predictions.loc[
            (predictions["split"] == "calibration") & (predictions["method"] == method)
        ].copy()
        test_frame = predictions.loc[
            (predictions["split"] == "test") & (predictions["method"] == method)
        ].copy()
        thresholds = np.unique(
            np.quantile(cal_frame["score"], np.linspace(0.02, 0.98, 25))
        )
        for threshold in thresholds:
            cal = _first_actions(cal_frame, cal_frame["score"] <= threshold)
            tst = _first_actions(test_frame, test_frame["score"] <= threshold)
            cm = _action_metrics(cal, 0.95, len(thresholds))
            tm = _action_metrics(tst, 0.95, 1)
            rows.append(
                {
                    "policy": method,
                    "parameter": float(threshold),
                    **{f"cal_{key}": value for key, value in cm.items()},
                    **{f"test_{key}": value for key, value in tm.items()},
                    "qualified": bool(
                        cm["actions"] >= minimum_actions
                        and cm["action_risk_ucb"] <= target_risk
                        and cm["resource_risk_ucb"] <= target_risk
                    ),
                }
            )
    return pd.DataFrame(rows)


def loading_consequence_audit(
    episodes: pd.DataFrame, extracted_root: str | Path
) -> pd.DataFrame:
    extracted_root = Path(extracted_root)
    event_frames = {}
    for name, filename in [
        ("pipeline", "pipeline_update_latency_anon.csv"),
        ("base_model", "basemodel_update_latency_anon.csv"),
    ]:
        frame = pd.read_csv(extracted_root / filename).sort_values(
            ["container_ip", "timestamp_anon"]
        )
        event_frames[name] = frame

    rows = []
    revived = episodes.loc[episodes["revival_timestamp"].notna()].copy()
    for episode in revived.itertuples(index=False):
        row: dict[str, Any] = {
            "episode_id": episode.episode_id,
            "container_ip": episode.container_ip,
            "revival_timestamp": episode.revival_timestamp,
        }
        for name, events in event_frames.items():
            candidates = events.loc[
                (events["container_ip"] == episode.container_ip)
                & (events["timestamp_anon"] >= episode.revival_timestamp - 120)
                & (events["timestamp_anon"] <= episode.revival_timestamp + 300)
            ].copy()
            if candidates.empty:
                row[f"{name}_event"] = 0
                row[f"{name}_latency_ms"] = np.nan
            else:
                nearest = candidates.iloc[
                    np.argmin(np.abs(candidates["timestamp_anon"] - episode.revival_timestamp))
                ]
                row[f"{name}_event"] = 1
                row[f"{name}_latency_ms"] = float(nearest["value"])
            placebo_timestamp = episode.revival_timestamp - 1800
            placebo = events.loc[
                (events["container_ip"] == episode.container_ip)
                & (events["timestamp_anon"] >= placebo_timestamp - 120)
                & (events["timestamp_anon"] <= placebo_timestamp + 300)
            ]
            row[f"{name}_placebo_event"] = int(not placebo.empty)
        rows.append(row)
    return pd.DataFrame(rows)


def run_gentd_external(extracted_root: str | Path, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned, manifest = load_aligned_gpu(extracted_root)
    landmarks, episodes = build_gentd_landmarks(aligned)
    landmarks, design = assign_splits(landmarks)
    predictions, metrics = fit_predictions(landmarks)
    policies = replay_policies(landmarks, predictions)
    loading = loading_consequence_audit(episodes, extracted_root)
    loading_summary_rows = []
    for event_type in ["pipeline", "base_model"]:
        actual = loading[f"{event_type}_event"].astype(bool)
        placebo = loading[f"{event_type}_placebo_event"].astype(bool)
        loading_summary_rows.append(
            {
                "event_type": event_type,
                "revived_episodes": int(len(loading)),
                "actual_window_events": int(actual.sum()),
                "actual_window_rate": float(actual.mean()),
                "placebo_window_events": int(placebo.sum()),
                "placebo_window_rate": float(placebo.mean()),
                "actual_only_pairs": int((actual & ~placebo).sum()),
                "placebo_only_pairs": int((~actual & placebo).sum()),
            }
        )
    loading_summary = pd.DataFrame(loading_summary_rows)

    landmarks.to_parquet(output_dir / "cold_landmarks.parquet", index=False)
    episodes.to_parquet(output_dir / "cold_episodes.parquet", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    policies.to_csv(output_dir / "policy_replay.csv", index=False)
    loading.to_csv(output_dir / "loading_consequence.csv", index=False)
    loading_summary.to_csv(output_dir / "loading_consequence_summary.csv", index=False)
    split_audit = (
        landmarks.groupby("split", as_index=False)
        .agg(
            landmarks=("episode_id", "size"),
            episodes=("episode_id", "nunique"),
            containers=("container_ip", "nunique"),
            revival_rate_1h=("reactivates_within_horizon", "mean"),
            median_memory_gib=("memory_gib", "median"),
        )
    )
    split_audit.to_csv(output_dir / "split_audit.csv", index=False)

    manifest.update(design)
    manifest["landmarks"] = int(len(landmarks))
    manifest["episodes"] = int(landmarks["episode_id"].nunique())
    manifest["qualified_policy_count"] = int(policies["qualified"].sum())
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    test = metrics.loc[metrics["split"] == "test"].set_index("method")
    age = test.loc["cold_age"]
    hierarchy = test.loc["hierarchical"]
    split_index = split_audit.set_index("split")
    pipeline_rate = float(loading["pipeline_event"].mean()) if not loading.empty else np.nan
    base_rate = float(loading["base_model_event"].mean()) if not loading.empty else np.nan
    pipeline_placebo_rate = (
        float(loading["pipeline_placebo_event"].mean()) if not loading.empty else np.nan
    )
    base_placebo_rate = (
        float(loading["base_model_placebo_event"].mean()) if not loading.empty else np.nan
    )
    pipeline_latency = float(loading["pipeline_latency_ms"].median())
    base_latency = float(loading["base_model_latency_ms"].median())
    qualification = (
        f"{int(policies['qualified'].sum())} 个策略候选获得资格"
        if policies["qualified"].any()
        else "没有策略候选满足双5%风险合同，保持ABSTAIN"
    )
    report = f"""# GenTD26 真实显存冷驻留外部验证

## 结论

GenTD26提供 {manifest['aligned_rows']:,} 条精确对齐的容器级GPU duty-cycle与显存记录，覆盖 {manifest['containers']} 个容器和 {manifest['span_hours']:.1f} 小时。采用“GPU duty不超过1%、显存至少10 GiB、连续冷30分钟”的固定定义，共构造 {manifest['landmarks']:,} 个成熟决策landmark，覆盖 {manifest['episodes']:,} 个冷episode。

严格后移测试段上，ColdGuard层级模型取得 ROC-AUC {hierarchy['roc_auc']:.3f}、PR-AUC {hierarchy['average_precision']:.3f}，冷龄基线为 {age['roc_auc']:.3f}/{age['average_precision']:.3f}。{qualification}。

## 固定协议

- 单位：匿名Stable Diffusion服务容器 `container_ip`。
- 冷状态：GPU duty-cycle ≤1%，同时实际显存 ≥10 GiB。
- 最小冷龄：30分钟；决策以约5分钟粒度生成。
- 事件：决策后1小时内GPU duty-cycle重新超过1%。
- 时间切分：前50%训练、中间25%校准、最后25%测试，并保留1小时标签成熟隔离带。共有{manifest['cross_split_episode_count']}个长episode在边界后继续出现；其早期标签均在边界前成熟，因此保留为顺序在线观测，并单独记录而不宣称实体冷启动泛化。
- 双风险合同：动作风险和显存加权风险的95%校准上界均不超过5%，至少30个独立episode动作。

## 数据审计

| 切分 | Landmarks | Episodes | 容器 | 1h复活率 | 中位显存GiB |
|---|---:|---:|---:|---:|---:|
| 训练 | {int(split_index.loc['train','landmarks'])} | {int(split_index.loc['train','episodes'])} | {int(split_index.loc['train','containers'])} | {split_index.loc['train','revival_rate_1h']:.1%} | {split_index.loc['train','median_memory_gib']:.1f} |
| 校准 | {int(split_index.loc['calibration','landmarks'])} | {int(split_index.loc['calibration','episodes'])} | {int(split_index.loc['calibration','containers'])} | {split_index.loc['calibration','revival_rate_1h']:.1%} | {split_index.loc['calibration','median_memory_gib']:.1f} |
| 测试 | {int(split_index.loc['test','landmarks'])} | {int(split_index.loc['test','episodes'])} | {int(split_index.loc['test','containers'])} | {split_index.loc['test','revival_rate_1h']:.1%} | {split_index.loc['test','median_memory_gib']:.1f} |

## 测试段预测效果

| 方法 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| 常数训练率 | {test.loc['constant_train_rate','roc_auc']:.3f} | {test.loc['constant_train_rate','average_precision']:.3f} | {test.loc['constant_train_rate','brier']:.3f} |
| 冷龄 | {age['roc_auc']:.3f} | {age['average_precision']:.3f} | {age['brier']:.3f} |
| ColdGuard层级模型 | {hierarchy['roc_auc']:.3f} | {hierarchy['average_precision']:.3f} | {hierarchy['brier']:.3f} |

## 加载代价证据

在复活时间前2分钟至后5分钟的观测窗口内，{pipeline_rate:.1%} 的复活episode匹配到Pipeline更新事件，其中位加载延迟为 {pipeline_latency:.0f} ms；{base_rate:.1%} 匹配到Base Model更新事件，其中位加载延迟为 {base_latency:.0f} ms。相同容器、相同窗口宽度但向前平移30分钟的配对安慰剂率分别只有 {pipeline_placebo_rate:.1%} 和 {base_placebo_rate:.1%}。这仍是观测关联而非回收导致的因果效应，但表明加载事件高度集中在复活邻域，而不是容器背景事件率造成的假象。

## 证据定位

该数据集来自Stable Diffusion而非LLM，且只有约一天，因此不承担长期漂移结论。它的论文价值是跨生成式AI工作负载验证“零计算但显存驻留—随后复活—伴随模型加载代价”这一ColdGuard机制链条。
"""
    (output_dir / "report_CN.md").write_text(report, encoding="utf-8")
    return output_dir / "report_CN.md"
