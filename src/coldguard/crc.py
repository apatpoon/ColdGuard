from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .policy import first_actions_by_score


DEPLOYMENT_KEYS = ["app_id", "model_id", "site_id", "pd_role"]


@dataclass(frozen=True)
class GroupedCRCChoice:
    threshold: float
    deployments: int
    actions: int
    corrected_action_harm: float
    corrected_resource_harm: float


def deployment_ids(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[DEPLOYMENT_KEYS]
        .fillna("__NULL__")
        .astype(str)
        .agg("|".join, axis=1)
    )


def monotonize_losses(losses: np.ndarray) -> np.ndarray:
    """Return the least cumulative-maximum envelope along permissiveness.

    Columns must be ordered from the most conservative score threshold to the
    most permissive. The output is pointwise no smaller than the raw loss and
    non-decreasing across columns, as required by the CRC construction after
    reversing the paper's conservativeness convention.
    """
    values = np.asarray(losses, dtype=float)
    if values.ndim != 2:
        raise ValueError("losses must have shape [groups, thresholds]")
    if not np.isfinite(values).all():
        raise ValueError("losses contain non-finite values")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("losses must be bounded in [0, 1]")
    return np.maximum.accumulate(values, axis=1)


def crc_corrected_risk(losses: np.ndarray, bound: float = 1.0) -> np.ndarray:
    """Compute ``n/(n+1) * empirical_risk + B/(n+1)`` per threshold."""
    values = np.asarray(losses, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("losses must contain at least one calibration group")
    if bound <= 0:
        raise ValueError("bound must be positive")
    n = values.shape[0]
    return (n / (n + 1.0)) * values.mean(axis=0) + bound / (n + 1.0)


def grouped_loss_tables(
    frame: pd.DataFrame,
    score_column: str,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deployment-level action and resource harm tables.

    Action harm is the indicator that a deployment incurs at least one false
    reclaim in the observed window. Resource harm is falsely reclaimed
    GiB-hours divided by the deployment's potential GiB-hours across observed
    episodes. Both are in [0, 1]. Raw losses are monotonized because changing
    the threshold can move an episode's first action to an earlier day.
    """
    if frame.empty:
        raise ValueError("frame is empty")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size < 1:
        raise ValueError("thresholds must be a non-empty vector")
    if np.any(np.diff(thresholds) < 0):
        raise ValueError("thresholds must be sorted from conservative to permissive")

    work = frame.copy()
    work["_deployment_id"] = deployment_ids(work)
    groups = np.sort(work["_deployment_id"].unique())
    group_index = {value: index for index, value in enumerate(groups)}
    earliest = (
        work.sort_values(["episode_id", "day"])
        .drop_duplicates("episode_id", keep="first")
    )
    potential = (
        earliest.groupby("_deployment_id")["remaining_mem_gib_hours"]
        .sum()
        .reindex(groups, fill_value=0.0)
        .to_numpy(float)
    )
    potential = np.maximum(potential, 1e-12)

    action_losses = np.zeros((len(groups), len(thresholds)), dtype=float)
    resource_losses = np.zeros_like(action_losses)
    action_counts = np.zeros(len(thresholds), dtype=int)
    for threshold_index, threshold in enumerate(thresholds):
        actions = first_actions_by_score(work, score_column, float(threshold))
        action_counts[threshold_index] = len(actions)
        if actions.empty:
            continue
        false_actions = actions.loc[actions["reactivates_within_horizon"].astype(bool)].copy()
        if false_actions.empty:
            continue
        false_actions["_group_index"] = false_actions["_deployment_id"].map(group_index)
        indices = false_actions["_group_index"].to_numpy(int)
        action_losses[np.unique(indices), threshold_index] = 1.0
        false_resource = (
            false_actions.groupby("_deployment_id")["remaining_mem_gib_hours"]
            .sum()
            .reindex(groups, fill_value=0.0)
            .to_numpy(float)
        )
        resource_losses[:, threshold_index] = np.clip(
            false_resource / potential, 0.0, 1.0
        )

    return (
        monotonize_losses(action_losses),
        monotonize_losses(resource_losses),
        action_counts,
    )


def choose_grouped_crc_threshold(
    frame: pd.DataFrame,
    score_column: str,
    thresholds: np.ndarray,
    target_action_harm: float,
    target_resource_harm: float,
    minimum_actions: int = 1,
) -> GroupedCRCChoice | None:
    action_losses, resource_losses, action_counts = grouped_loss_tables(
        frame, score_column, thresholds
    )
    action_risk = crc_corrected_risk(action_losses)
    resource_risk = crc_corrected_risk(resource_losses)
    eligible = np.flatnonzero(
        (action_risk <= target_action_harm)
        & (resource_risk <= target_resource_harm)
        & (action_counts >= minimum_actions)
    )
    if eligible.size == 0:
        return None
    index = int(eligible[-1])
    return GroupedCRCChoice(
        threshold=float(np.asarray(thresholds)[index]),
        deployments=int(action_losses.shape[0]),
        actions=int(action_counts[index]),
        corrected_action_harm=float(action_risk[index]),
        corrected_resource_harm=float(resource_risk[index]),
    )


def grouped_harm_metrics(frame: pd.DataFrame, actions: pd.DataFrame) -> dict[str, float | int]:
    work = frame.copy()
    work["_deployment_id"] = deployment_ids(work)
    groups = np.sort(work["_deployment_id"].unique())
    earliest = (
        work.sort_values(["episode_id", "day"])
        .drop_duplicates("episode_id", keep="first")
    )
    potential = (
        earliest.groupby("_deployment_id")["remaining_mem_gib_hours"]
        .sum()
        .reindex(groups, fill_value=0.0)
        .to_numpy(float)
    )
    if actions.empty:
        return {
            "deployments": int(len(groups)),
            "deployment_action_harm": 0.0,
            "deployment_resource_harm": 0.0,
        }
    selected = actions.copy()
    selected["_deployment_id"] = deployment_ids(selected)
    false_actions = selected.loc[selected["reactivates_within_horizon"].astype(bool)]
    harmed = false_actions["_deployment_id"].unique()
    false_resource = (
        false_actions.groupby("_deployment_id")["remaining_mem_gib_hours"]
        .sum()
        .reindex(groups, fill_value=0.0)
        .to_numpy(float)
    )
    resource_harm = np.clip(false_resource / np.maximum(potential, 1e-12), 0.0, 1.0)
    return {
        "deployments": int(len(groups)),
        "deployment_action_harm": float(len(harmed) / max(1, len(groups))),
        "deployment_resource_harm": float(resource_harm.mean()),
    }

