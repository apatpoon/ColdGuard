from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

from .policy import effective_sample_size, wilson_upper_bound


@dataclass(frozen=True)
class RatioBound:
    estimate: float
    upper: float
    cluster_upper: float
    finite_sample_floor: float
    standard_error: float
    numerator: float
    denominator: float
    observations: int
    day_clusters: int
    deployment_clusters: int
    effective_observations: float


@dataclass(frozen=True)
class DualRiskBound:
    action: RatioBound
    resource: RatioBound
    family_confidence: float
    per_risk_confidence: float


def deployment_ids(
    frame: pd.DataFrame,
    columns: Sequence[str] = ("app_id", "model_id", "site_id", "pd_role"),
) -> pd.Series:
    """Return a stable logical-deployment identifier without exposing raw keys."""
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"Missing deployment columns: {missing}")
    return frame.loc[:, list(columns)].astype("string").fillna("<NA>").agg("|".join, axis=1)


def _cluster_meat(values: np.ndarray, groups: Iterable[object]) -> tuple[float, int]:
    table = pd.DataFrame({"value": values, "group": pd.Series(groups, dtype="string")})
    totals = table.groupby("group", observed=True, sort=False)["value"].sum().to_numpy(float)
    count = int(len(totals))
    if count < 2:
        return 0.0, count
    correction = count / (count - 1.0)
    return float(correction * np.square(totals).sum()), count


def clustered_ratio_upper_bound(
    labels: np.ndarray,
    denominator_weights: np.ndarray,
    days: Iterable[object],
    deployments: Iterable[object],
    confidence: float,
) -> RatioBound:
    """One-sided empirical upper bound for a pooled weighted event ratio.

    The point estimate is ``sum(weight * label) / sum(weight)``.  Its primary
    uncertainty term is an additive two-axis cluster sandwich: calendar-day
    and logical-deployment score contributions are both retained.  Additivity
    deliberately double-counts their intersection instead of subtracting a
    potentially unstable term.  A Wilson/Kish bound is retained only as a
    finite-event and weight-concentration floor.

    This is an empirical dependence-stress bound, not a finite-sample theorem
    for arbitrary serial processes.  The manuscript therefore couples it to
    chronological forward audit and simulation coverage checks.
    """
    y = np.asarray(labels, dtype=float)
    weight = np.asarray(denominator_weights, dtype=float)
    if y.ndim != 1 or weight.ndim != 1 or len(y) != len(weight):
        raise ValueError("labels and denominator_weights must be aligned 1D arrays")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    if len(y) == 0:
        return RatioBound(
            estimate=0.0,
            upper=1.0,
            cluster_upper=1.0,
            finite_sample_floor=1.0,
            standard_error=math.inf,
            numerator=0.0,
            denominator=0.0,
            observations=0,
            day_clusters=0,
            deployment_clusters=0,
            effective_observations=0.0,
        )
    if np.any(~np.isfinite(y)) or np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("labels must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("denominator weights must be finite and non-negative")

    denominator = float(weight.sum())
    if denominator <= 0.0:
        return RatioBound(
            estimate=float(y.mean()),
            upper=1.0,
            cluster_upper=1.0,
            finite_sample_floor=1.0,
            standard_error=math.inf,
            numerator=0.0,
            denominator=0.0,
            observations=int(len(y)),
            day_clusters=int(pd.Series(days).nunique()),
            deployment_clusters=int(pd.Series(deployments).nunique()),
            effective_observations=0.0,
        )

    numerator = float(np.dot(weight, y))
    estimate = float(numerator / denominator)
    score = weight * (y - estimate)
    day_meat, day_count = _cluster_meat(score, days)
    deployment_meat, deployment_count = _cluster_meat(score, deployments)
    variance = max(0.0, day_meat + deployment_meat) / (denominator * denominator)
    standard_error = float(math.sqrt(variance))
    degrees_freedom = max(1, min(day_count, deployment_count) - 1)
    critical = float(t.ppf(confidence, degrees_freedom))
    cluster_upper = float(min(1.0, estimate + critical * standard_error))

    effective = effective_sample_size(weight)
    floor = wilson_upper_bound(
        estimate * effective,
        effective,
        confidence=confidence,
        comparisons=1,
    )
    upper = float(max(cluster_upper, floor))
    return RatioBound(
        estimate=estimate,
        upper=upper,
        cluster_upper=cluster_upper,
        finite_sample_floor=float(floor),
        standard_error=standard_error,
        numerator=numerator,
        denominator=denominator,
        observations=int(len(y)),
        day_clusters=day_count,
        deployment_clusters=deployment_count,
        effective_observations=float(effective),
    )


def dual_clustered_risk_bound(
    actions: pd.DataFrame,
    family_confidence: float = 0.95,
    label_column: str = "reactivates_within_horizon",
    exposure_column: str = "remaining_mem_gib_hours",
) -> DualRiskBound:
    """Joint action/resource audit with Bonferroni control over two risks."""
    if not 0.5 < family_confidence < 1.0:
        raise ValueError("family_confidence must lie in (0.5, 1)")
    per_risk_confidence = 1.0 - (1.0 - family_confidence) / 2.0
    if actions.empty:
        empty = clustered_ratio_upper_bound(
            np.array([], dtype=float),
            np.array([], dtype=float),
            [],
            [],
            per_risk_confidence,
        )
        return DualRiskBound(empty, empty, family_confidence, per_risk_confidence)

    required = {label_column, exposure_column, "day"}
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise KeyError(f"Missing action columns: {missing}")
    labels = actions[label_column].to_numpy(float)
    exposure = (
        actions[exposure_column].fillna(0.0).clip(lower=0.0).to_numpy(float)
    )
    days = actions["day"].to_numpy()
    deployments = deployment_ids(actions).to_numpy()
    action = clustered_ratio_upper_bound(
        labels,
        np.ones(len(actions), dtype=float),
        days,
        deployments,
        per_risk_confidence,
    )
    resource = clustered_ratio_upper_bound(
        labels,
        exposure,
        days,
        deployments,
        per_risk_confidence,
    )
    return DualRiskBound(action, resource, family_confidence, per_risk_confidence)
