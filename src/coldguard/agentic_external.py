from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .pilot import HierarchicalTabularEncoder, _predictive_metrics
from .policy import wilson_upper_bound


HOUR_MS = 3_600_000
HORIZON_HOURS = 6
DECISION_AGES_HOURS = (1, 6, 12, 24)

NUMERIC_FEATURES = [
    "cold_age_h",
    "previous_gap_h",
    "requests_prior_1h",
    "requests_prior_6h",
    "requests_prior_24h",
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
    "ttft_ms",
    "n_messages",
    "hour_sin",
    "hour_cos",
]
CATEGORICAL_FEATURES = ["model", "provider"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requests(release_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    release_root = Path(release_root)
    files = sorted(release_root.glob("*/requests.parquet"))
    if len(files) != 16:
        raise ValueError(f"Expected 16 weekly request files, found {len(files)}")
    pattern = str(release_root / "*" / "requests.parquet")
    connection = duckdb.connect()
    query = f"""
        SELECT
            session_id,
            step,
            start_ms,
            end_ms,
            ttft_ms,
            model,
            provider,
            prompt_tokens,
            completion_tokens,
            cache_read_tokens,
            n_messages
        FROM read_parquet('{pattern}', union_by_name=true)
        WHERE model IS NOT NULL
          AND provider IS NOT NULL
          AND start_ms IS NOT NULL
          AND end_ms IS NOT NULL
        ORDER BY model, provider, start_ms, end_ms
    """
    requests = connection.execute(query).fetchdf()
    connection.close()
    if requests.empty:
        raise ValueError("No requests loaded")
    if (requests["end_ms"] < requests["start_ms"]).any():
        raise ValueError("Found requests ending before they start")
    manifest = {
        "weekly_files": len(files),
        "weekly_sha256": {str(path.relative_to(release_root)): _sha256(path) for path in files},
        "request_rows": int(len(requests)),
        "models": int(requests["model"].nunique()),
        "providers": int(requests["provider"].nunique()),
        "model_provider_groups": int(
            requests[["model", "provider"]].drop_duplicates().shape[0]
        ),
        "start_utc": pd.to_datetime(requests["start_ms"].min(), unit="ms", utc=True).isoformat(),
        "end_utc": pd.to_datetime(requests["start_ms"].max(), unit="ms", utc=True).isoformat(),
    }
    return requests, manifest


def _recent_count(times: np.ndarray, index: int, window_ms: int) -> int:
    lower = np.searchsorted(times, times[index] - window_ms, side="left")
    return int(index - lower + 1)


def build_gap_landmarks(
    requests: pd.DataFrame,
    horizon_hours: int = HORIZON_HOURS,
    decision_ages_hours: tuple[int, ...] = DECISION_AGES_HOURS,
) -> pd.DataFrame:
    global_end = int(requests["start_ms"].max())
    horizon_ms = int(horizon_hours * HOUR_MS)
    rows: list[dict[str, Any]] = []
    for (model, provider), group in requests.groupby(["model", "provider"], sort=True):
        group = group.sort_values(["start_ms", "end_ms"]).reset_index(drop=True)
        starts = group["start_ms"].to_numpy(np.int64)
        ends = group["end_ms"].to_numpy(np.int64)
        previous_gap = np.full(len(group), np.nan)
        if len(group) > 1:
            previous_gap[1:] = np.maximum(0, starts[1:] - ends[:-1]) / HOUR_MS

        for index in range(len(group)):
            gap_start = int(ends[index])
            if index + 1 < len(group):
                next_start = int(starts[index + 1])
                event_observed = True
                available_gap = next_start - gap_start
            else:
                next_start = global_end
                event_observed = False
                available_gap = global_end - gap_start
            if available_gap <= 0:
                continue

            current = group.iloc[index]
            for cold_age_h in decision_ages_hours:
                decision_ms = gap_start + int(cold_age_h * HOUR_MS)
                if decision_ms > next_start:
                    continue
                if decision_ms + horizon_ms > global_end:
                    continue
                time_to_next_ms = next_start - decision_ms
                reactivates = int(event_observed and 0 <= time_to_next_ms <= horizon_ms)
                safe_hours = min(horizon_hours, max(0.0, time_to_next_ms / HOUR_MS))
                decision_time = pd.to_datetime(decision_ms, unit="ms", utc=True)
                rows.append(
                    {
                        "episode_id": f"{model}|{provider}|{index}",
                        "model": model,
                        "provider": provider,
                        "decision_ms": decision_ms,
                        "decision_time": decision_time,
                        "cold_age_h": float(cold_age_h),
                        "reactivates_within_horizon": reactivates,
                        "event_observed": int(event_observed),
                        "safe_reclaim_hours": float(safe_hours),
                        "previous_gap_h": float(previous_gap[index])
                        if np.isfinite(previous_gap[index])
                        else np.nan,
                        "requests_prior_1h": _recent_count(starts, index, HOUR_MS),
                        "requests_prior_6h": _recent_count(starts, index, 6 * HOUR_MS),
                        "requests_prior_24h": _recent_count(starts, index, 24 * HOUR_MS),
                        "prompt_tokens": current["prompt_tokens"],
                        "completion_tokens": current["completion_tokens"],
                        "cache_read_tokens": current["cache_read_tokens"],
                        "ttft_ms": current["ttft_ms"],
                        "n_messages": current["n_messages"],
                        "hour_sin": float(np.sin(2 * np.pi * decision_time.hour / 24.0)),
                        "hour_cos": float(np.cos(2 * np.pi * decision_time.hour / 24.0)),
                    }
                )
    landmarks = pd.DataFrame(rows)
    if landmarks.empty:
        raise ValueError("No mature gap landmarks were created")
    landmarks.sort_values(["decision_ms", "model", "provider", "cold_age_h"], inplace=True)
    landmarks.reset_index(drop=True, inplace=True)
    return landmarks


def assign_temporal_splits(landmarks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = landmarks.copy()
    start = int(result["decision_ms"].min())
    end = int(result["decision_ms"].max())
    span = end - start
    train_boundary = start + span // 2
    calibration_boundary = start + 3 * span // 4
    horizon_ms = HORIZON_HOURS * HOUR_MS
    result["split"] = np.select(
        [
            result["decision_ms"] <= train_boundary - horizon_ms,
            (result["decision_ms"] > train_boundary)
            & (result["decision_ms"] <= calibration_boundary - horizon_ms),
            result["decision_ms"] > calibration_boundary,
        ],
        ["train", "calibration", "test"],
        default="embargo",
    )
    design = {
        "train_boundary_utc": pd.to_datetime(train_boundary, unit="ms", utc=True).isoformat(),
        "calibration_boundary_utc": pd.to_datetime(
            calibration_boundary, unit="ms", utc=True
        ).isoformat(),
        "horizon_hours": HORIZON_HOURS,
        "decision_ages_hours": list(DECISION_AGES_HOURS),
    }
    return result, design


def _fit_model(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[HierarchicalTabularEncoder, HistGradientBoostingClassifier]:
    target = train["reactivates_within_horizon"].to_numpy(int)
    encoder = HierarchicalTabularEncoder(numeric, categorical, strength=30.0)
    if categorical:
        groups = train[["model", "provider"]].astype(str).agg("|".join, axis=1).to_numpy()
        encoded = encoder.fit_transform_group_oof(train, target, groups, n_splits=5)
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
        "cold_age": (["cold_age_h"], []),
        "hierarchical": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
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
                    "model",
                    "provider",
                    "decision_time",
                    "cold_age_h",
                    "reactivates_within_horizon",
                    "safe_reclaim_hours",
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
            "safe_reclaim_hours": 0.0,
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
    ttl_grid = list(DECISION_AGES_HOURS)
    for ttl in ttl_grid:
        cal = _first_actions(calibration, calibration["cold_age_h"] >= ttl)
        tst = _first_actions(test, test["cold_age_h"] >= ttl)
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
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_agentic_external(release_root: str | Path, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests, manifest = load_requests(release_root)
    landmarks, design = assign_temporal_splits(build_gap_landmarks(requests))
    predictions, metrics = fit_predictions(landmarks)
    policies = replay_policies(landmarks, predictions)

    landmarks.to_parquet(output_dir / "gap_landmarks.parquet", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    policies.to_csv(output_dir / "policy_replay.csv", index=False)
    split_audit = (
        landmarks.groupby("split", as_index=False)
        .agg(
            landmarks=("episode_id", "size"),
            episodes=("episode_id", "nunique"),
            model_provider_groups=("model", lambda values: 0),
            revival_rate_6h=("reactivates_within_horizon", "mean"),
        )
    )
    for idx, row in split_audit.iterrows():
        frame = landmarks.loc[landmarks["split"] == row["split"]]
        split_audit.loc[idx, "model_provider_groups"] = int(
            frame[["model", "provider"]].drop_duplicates().shape[0]
        )
    split_audit.to_csv(output_dir / "split_audit.csv", index=False)

    manifest.update(design)
    manifest["qualified_policy_count"] = int(policies["qualified"].sum())
    manifest["landmarks"] = int(len(landmarks))
    manifest["episodes"] = int(landmarks["episode_id"].nunique())
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    test = metrics.loc[metrics["split"] == "test"].set_index("method")
    split_index = split_audit.set_index("split")
    hierarchy = test.loc["hierarchical"]
    age = test.loc["cold_age"]
    qualified = policies.loc[policies["qualified"]]
    qualification_text = (
        f"{len(qualified)} 个候选在校准段获得资格"
        if not qualified.empty
        else "没有候选达到5%风险资格，策略保持ABSTAIN"
    )
    report = f"""# FreeInference Agentic 16周外部复活验证

## 结论

16周请求轨迹包含 {manifest['request_rows']:,} 次请求、{manifest['models']} 个模型、{manifest['providers']} 个Provider和 {manifest['model_provider_groups']} 个模型—Provider组合。基于模型—Provider级空闲间隔构造了 {manifest['landmarks']:,} 个成熟决策landmark，覆盖 {manifest['episodes']:,} 个空闲episode。

严格后移测试段上，ColdGuard层级风险模型取得 ROC-AUC {hierarchy['roc_auc']:.3f}、PR-AUC {hierarchy['average_precision']:.3f}；冷龄基线为 {age['roc_auc']:.3f}/{age['average_precision']:.3f}。{qualification_text}。

## 固定协议

- 单位：`model + provider`，不将其写成物理GPU部署实例。
- 冷决策年龄：1、6、12、24小时。
- 主要事件：决策后6小时内出现下一次模型请求。
- 时间切分：前50%训练、中间25%校准、最后25%测试；边界保留6小时成熟隔离带。
- 模型：冷龄风险基线与层级风险模型；层级模型加入模型、Provider、近期请求计数、Token、TTFT和时钟特征。
- 风险资格：校准段95%多重比较校正Wilson上界不超过5%，且至少30个独立episode动作。

## 数据审计

| 切分 | Landmarks | Episodes | 模型—Provider组 | 6h复活率 |
|---|---:|---:|---:|---:|
| 训练 | {int(split_index.loc['train','landmarks'])} | {int(split_index.loc['train','episodes'])} | {int(split_index.loc['train','model_provider_groups'])} | {split_index.loc['train','revival_rate_6h']:.1%} |
| 校准 | {int(split_index.loc['calibration','landmarks'])} | {int(split_index.loc['calibration','episodes'])} | {int(split_index.loc['calibration','model_provider_groups'])} | {split_index.loc['calibration','revival_rate_6h']:.1%} |
| 测试 | {int(split_index.loc['test','landmarks'])} | {int(split_index.loc['test','episodes'])} | {int(split_index.loc['test','model_provider_groups'])} | {split_index.loc['test','revival_rate_6h']:.1%} |

## 测试段预测效果

| 方法 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| 常数训练率 | {test.loc['constant_train_rate','roc_auc']:.3f} | {test.loc['constant_train_rate','average_precision']:.3f} | {test.loc['constant_train_rate','brier']:.3f} |
| 冷龄 | {age['roc_auc']:.3f} | {age['average_precision']:.3f} | {age['brier']:.3f} |
| ColdGuard层级风险 | {hierarchy['roc_auc']:.3f} | {hierarchy['average_precision']:.3f} | {hierarchy['brier']:.3f} |

## 证据定位

该数据集为ColdGuard提供独立于阿里MaaS的长期请求复活证据，但不包含GPU显存和物理部署映射。因此它只支撑复活排序、时间漂移和风险资格，不用于宣称显存回收量或GPU级安全性。
"""
    (output_dir / "report_CN.md").write_text(report, encoding="utf-8")
    return output_dir / "report_CN.md"
