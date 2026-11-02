from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from .config import load_config, require_keys, resolve_project_path
from .policy import action_metrics, first_actions_by_score


@dataclass(frozen=True)
class LTTChoice:
    threshold: float
    threshold_index: int
    actions: int
    action_p_value: float
    resource_p_value: float
    combined_p_value: float
    blocks: int
    mode: str


def bernoulli_kl(observed: float, null_mean: float) -> float:
    x = float(np.clip(observed, 0.0, 1.0))
    mu = float(np.clip(null_mean, 1e-12, 1.0 - 1e-12))
    if x == 0.0:
        return -math.log1p(-mu)
    if x == 1.0:
        return -math.log(mu)
    return x * math.log(x / mu) + (1.0 - x) * math.log((1.0 - x) / (1.0 - mu))


def hoeffding_bentkus_p_value(empirical_risk: float, n: int, target: float) -> float:
    """One-sided LTT p-value for H0: population risk > target.

    Mirrors the Hoeffding--Bentkus construction used by the official LTT code
    for losses bounded in [0, 1].
    """
    if n < 1:
        return 1.0
    risk = float(np.clip(empirical_risk, 0.0, 1.0))
    if risk >= target:
        return 1.0
    bentkus = math.e * float(binom.cdf(math.ceil(n * risk), n, target))
    hoeffding = math.exp(-n * bernoulli_kl(risk, target))
    return float(min(1.0, bentkus, hoeffding))


def daily_loss_tables(
    frame: pd.DataFrame,
    score_column: str,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Daily conditional action/resource risk for each fixed threshold."""
    thresholds = np.asarray(thresholds, dtype=float)
    days = np.arange(int(frame["day"].min()), int(frame["day"].max()) + 1)
    day_index = {day: index for index, day in enumerate(days)}
    action_losses = np.zeros((len(days), len(thresholds)), dtype=float)
    resource_losses = np.zeros_like(action_losses)
    action_counts = np.zeros(len(thresholds), dtype=int)
    for threshold_index, threshold in enumerate(thresholds):
        actions = first_actions_by_score(frame, score_column, float(threshold))
        action_counts[threshold_index] = len(actions)
        if actions.empty:
            continue
        for day, group in actions.groupby("day"):
            index = day_index[int(day)]
            labels = group["reactivates_within_horizon"].to_numpy(float)
            weights = group["remaining_mem_gib_hours"].fillna(0.0).clip(lower=0.0).to_numpy(float)
            action_losses[index, threshold_index] = float(labels.mean())
            resource_losses[index, threshold_index] = (
                float(np.dot(labels, weights) / weights.sum()) if weights.sum() > 0 else float(labels.mean())
            )
    return days, action_losses, resource_losses, action_counts


def aggregate_time_blocks(daily_losses: np.ndarray, block_days: int) -> np.ndarray:
    values = np.asarray(daily_losses, dtype=float)
    if values.ndim != 2:
        raise ValueError("daily_losses must have shape [days, thresholds]")
    if block_days < 1:
        raise ValueError("block_days must be positive")
    blocks = []
    for start in range(0, len(values), block_days):
        blocks.append(values[start : start + block_days].mean(axis=0))
    return np.vstack(blocks)


def select_ltt_threshold(
    action_block_losses: np.ndarray,
    resource_block_losses: np.ndarray,
    thresholds: np.ndarray,
    action_counts: np.ndarray,
    target_action_risk: float,
    target_resource_risk: float,
    delta: float,
    minimum_actions: int,
    mode: str,
) -> LTTChoice | None:
    thresholds = np.asarray(thresholds, dtype=float)
    n = int(action_block_losses.shape[0])
    action_p = np.array(
        [
            hoeffding_bentkus_p_value(action_block_losses[:, i].mean(), n, target_action_risk)
            for i in range(len(thresholds))
        ]
    )
    resource_p = np.array(
        [
            hoeffding_bentkus_p_value(resource_block_losses[:, i].mean(), n, target_resource_risk)
            for i in range(len(thresholds))
        ]
    )
    combined = np.maximum(action_p, resource_p)
    if mode == "bonferroni":
        passed = combined * len(thresholds) <= delta
    elif mode == "fixed_sequence":
        passed = np.zeros(len(thresholds), dtype=bool)
        for index, p_value in enumerate(combined):
            if p_value > delta:
                break
            passed[index] = True
    else:
        raise ValueError("mode must be 'bonferroni' or 'fixed_sequence'")
    eligible = np.flatnonzero(passed & (action_counts >= minimum_actions))
    if eligible.size == 0:
        return None
    index = int(eligible[-1])
    return LTTChoice(
        threshold=float(thresholds[index]),
        threshold_index=index,
        actions=int(action_counts[index]),
        action_p_value=float(action_p[index]),
        resource_p_value=float(resource_p[index]),
        combined_p_value=float(combined[index]),
        blocks=n,
        mode=mode,
    )


def run_block_ltt(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(
        config,
        "paths",
        ["multihorizon_artifacts_dir", "block_ltt_artifacts_dir"],
    )
    require_keys(config, "multihorizon", ["horizons", "target_resource_risk", "target_action_risk"])
    require_keys(config, "block_ltt", ["block_days", "confidence", "threshold_grid_size"])
    require_keys(config, "risk", ["minimum_selected"])

    source = resolve_project_path(project_root, config["paths"]["multihorizon_artifacts_dir"])
    artifacts = resolve_project_path(project_root, config["paths"]["block_ltt_artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    calibration_path = source / "calibration_predictions.parquet"
    test_path = source / "test_predictions.parquet"
    if not calibration_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run coldguard multihorizon before block-ltt")
    calibration_base = pd.read_parquet(calibration_path)
    test_base = pd.read_parquet(test_path)

    horizons = sorted(int(value) for value in config["multihorizon"]["horizons"])
    methods = sorted(
        {
            column.removeprefix("score_").rsplit("_", 1)[0]
            for column in calibration_base.columns
            if column.startswith("score_")
        }
    )
    cfg = config["block_ltt"]
    thresholds = np.linspace(0.0, 1.0, int(cfg["threshold_grid_size"]))
    delta = 1.0 - float(cfg["confidence"])
    minimum_actions = int(config["risk"]["minimum_selected"])
    action_target = float(config["multihorizon"]["target_action_risk"])
    resource_target = float(config["multihorizon"]["target_resource_risk"])
    rows: list[dict[str, Any]] = []

    for horizon in horizons:
        label_column = f"label_{horizon}d"
        for method in methods:
            score_column = f"score_{method}_{horizon}d"
            calibration = calibration_base.copy()
            test = test_base.copy()
            calibration["reactivates_within_horizon"] = calibration[label_column]
            test["reactivates_within_horizon"] = test[label_column]
            _, daily_action, daily_resource, counts = daily_loss_tables(
                calibration, score_column, thresholds
            )
            for block_days in [int(value) for value in cfg["block_days"]]:
                action_blocks = aggregate_time_blocks(daily_action, block_days)
                resource_blocks = aggregate_time_blocks(daily_resource, block_days)
                for mode in ["fixed_sequence", "bonferroni"]:
                    choice = select_ltt_threshold(
                        action_blocks,
                        resource_blocks,
                        thresholds,
                        counts,
                        action_target,
                        resource_target,
                        delta,
                        minimum_actions,
                        mode,
                    )
                    if choice is None:
                        rows.append(
                            {
                                "horizon": horizon,
                                "method": method,
                                "block_days": block_days,
                                "mode": mode,
                                "eligible": False,
                                "blocks": len(action_blocks),
                            }
                        )
                        continue
                    actions = first_actions_by_score(test, score_column, choice.threshold)
                    metrics = action_metrics(actions, confidence=1.0 - delta, comparisons=1)
                    _, test_daily_action, test_daily_resource, _ = daily_loss_tables(
                        test, score_column, np.array([choice.threshold])
                    )
                    test_action_blocks = aggregate_time_blocks(test_daily_action, block_days)
                    test_resource_blocks = aggregate_time_blocks(test_daily_resource, block_days)
                    rows.append(
                        {
                            "horizon": horizon,
                            "method": method,
                            "block_days": block_days,
                            "mode": mode,
                            "eligible": True,
                            **choice.__dict__,
                            "calibration_action_block_risk": float(
                                action_blocks[:, choice.threshold_index].mean()
                            ),
                            "calibration_resource_block_risk": float(
                                resource_blocks[:, choice.threshold_index].mean()
                            ),
                            "test_action_block_risk": float(test_action_blocks.mean()),
                            "test_resource_block_risk": float(test_resource_blocks.mean()),
                            **{f"test_{key}": value for key, value in metrics.items()},
                        }
                    )

    results = pd.DataFrame(rows)
    results.to_csv(artifacts / "block_ltt_results.csv", index=False)
    report = _render_report(results, action_target, resource_target, delta)
    report_path = artifacts / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _render_report(
    results: pd.DataFrame,
    action_target: float,
    resource_target: float,
    delta: float,
) -> str:
    lines = [
        "# ColdGuard Time-Block Learn-then-Test Report",
        "",
        f"Null risks: daily/block action risk > {action_target:.1%} or resource risk > "
        f"{resource_target:.1%}; family-wise error target: {delta:.1%}.",
        "",
        "| Horizon | Method | Block days | Mode | Eligible | Blocks | Test block action risk | Test block resource risk | Test conditional action risk |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {int(row['horizon'])} | {row['method']} | {int(row['block_days'])} | "
                f"{row['mode']} | no | {int(row['blocks'])} | — | — | — |"
            )
        else:
            lines.append(
                f"| {int(row['horizon'])} | {row['method']} | {int(row['block_days'])} | "
                f"{row['mode']} | yes | {int(row['blocks'])} | "
                f"{row['test_action_block_risk']:.2%} | {row['test_resource_block_risk']:.2%} | "
                f"{row['test_unweighted_risk']:.2%} |"
            )
    lines.extend(
        [
            "",
            "Daily losses are bounded conditional risks with zero loss on abstention days. "
            "Fixed-sequence testing is valid only for the predeclared conservative-to-permissive "
            "threshold order and stops at the first non-rejection. Bonferroni tests all fixed "
            "thresholds. Consecutive calendar blocks are not assumed independent by the data; "
            "results are a dependence-sensitivity diagnostic, not an unconditional theorem.",
        ]
    )
    return "\n".join(lines) + "\n"

