from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class PolicyChoice:
    policy: str
    parameter: float
    calibration_actions: int
    calibration_weighted_risk: float
    calibration_risk_ucb: float
    calibration_unweighted_risk: float
    calibration_unweighted_risk_ucb: float
    calibration_reclaimed_gib_hours: float


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    squared = float(np.square(weights).sum())
    if total <= 0 or squared <= 0:
        return 0.0
    return total * total / squared


def wilson_upper_bound(
    successes: float,
    trials: float,
    confidence: float = 0.95,
    comparisons: int = 1,
) -> float:
    if trials <= 0:
        return 1.0
    alpha = max(1e-12, (1.0 - confidence) / max(1, comparisons))
    z = float(norm.ppf(1.0 - alpha))
    p = min(1.0, max(0.0, successes / trials))
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return min(1.0, (center + radius) / denominator)


def first_actions_by_score(frame: pd.DataFrame, score_column: str, threshold: float) -> pd.DataFrame:
    selected = frame.loc[frame[score_column] <= threshold].copy()
    if selected.empty:
        return selected
    selected.sort_values(["episode_id", "day"], inplace=True)
    return selected.drop_duplicates("episode_id", keep="first")


def first_actions_by_ttl(frame: pd.DataFrame, ttl_days: int) -> pd.DataFrame:
    selected = frame.loc[frame["cold_age"] >= ttl_days].copy()
    if selected.empty:
        return selected
    selected.sort_values(["episode_id", "day"], inplace=True)
    return selected.drop_duplicates("episode_id", keep="first")


def action_metrics(
    actions: pd.DataFrame,
    confidence: float,
    comparisons: int,
) -> dict[str, float | int]:
    if actions.empty:
        return {
            "actions": 0,
            "false_reclaims": 0,
            "unweighted_risk": 0.0,
            "unweighted_risk_ucb": 1.0,
            "weighted_risk": 0.0,
            "risk_ucb": 1.0,
            "reclaimed_gib_hours": 0.0,
            "effective_actions": 0.0,
        }
    labels = actions["reactivates_within_horizon"].to_numpy(dtype=float)
    weights = actions["remaining_mem_gib_hours"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    total_weight = float(weights.sum())
    weighted_risk = float(np.dot(labels, weights) / total_weight) if total_weight > 0 else float(labels.mean())
    n_eff = effective_sample_size(weights) if total_weight > 0 else float(len(labels))
    weighted_ucb = wilson_upper_bound(weighted_risk * n_eff, n_eff, confidence, comparisons)
    unweighted_ucb = wilson_upper_bound(float(labels.sum()), float(len(labels)), confidence, comparisons)
    return {
        "actions": int(len(actions)),
        "false_reclaims": int(labels.sum()),
        "unweighted_risk": float(labels.mean()),
        "unweighted_risk_ucb": unweighted_ucb,
        "weighted_risk": weighted_risk,
        "risk_ucb": weighted_ucb,
        "reclaimed_gib_hours": total_weight,
        "effective_actions": n_eff,
    }


def score_frontier(
    frame: pd.DataFrame,
    score_column: str,
    grid_size: int,
    confidence: float,
) -> pd.DataFrame:
    values = frame[score_column].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return pd.DataFrame()
    quantiles = np.linspace(0.01, 0.99, max(3, grid_size))
    thresholds = np.unique(np.quantile(values, quantiles))
    rows = []
    for threshold in thresholds:
        metrics = action_metrics(
            first_actions_by_score(frame, score_column, float(threshold)),
            confidence=confidence,
            comparisons=len(thresholds),
        )
        rows.append({"parameter": float(threshold), **metrics})
    return pd.DataFrame(rows)


def ttl_frontier(frame: pd.DataFrame, confidence: float) -> pd.DataFrame:
    maximum = int(max(1, frame["cold_age"].max()))
    ttls = list(range(1, maximum + 1))
    rows = []
    for ttl in ttls:
        metrics = action_metrics(
            first_actions_by_ttl(frame, ttl),
            confidence=confidence,
            comparisons=len(ttls),
        )
        rows.append({"parameter": float(ttl), **metrics})
    return pd.DataFrame(rows)


def choose_policy(
    frontier: pd.DataFrame,
    policy: str,
    target_risk: float,
    minimum_selected: int,
    target_action_risk: float | None = None,
) -> PolicyChoice | None:
    if frontier.empty:
        return None
    mask = (frontier["risk_ucb"] <= target_risk) & (frontier["actions"] >= minimum_selected)
    if target_action_risk is not None:
        mask &= frontier["unweighted_risk_ucb"] <= target_action_risk
    eligible = frontier.loc[mask]
    if eligible.empty:
        return None
    best = eligible.sort_values(
        ["reclaimed_gib_hours", "risk_ucb"], ascending=[False, True]
    ).iloc[0]
    return PolicyChoice(
        policy=policy,
        parameter=float(best["parameter"]),
        calibration_actions=int(best["actions"]),
        calibration_weighted_risk=float(best["weighted_risk"]),
        calibration_risk_ucb=float(best["risk_ucb"]),
        calibration_unweighted_risk=float(best["unweighted_risk"]),
        calibration_unweighted_risk_ucb=float(best["unweighted_risk_ucb"]),
        calibration_reclaimed_gib_hours=float(best["reclaimed_gib_hours"]),
    )

