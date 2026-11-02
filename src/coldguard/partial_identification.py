from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clustered_risk import DualRiskBound, dual_clustered_risk_bound


@dataclass(frozen=True)
class ProposalStreamAudit:
    lower: DualRiskBound
    upper: DualRiskBound
    proposed_actions: int
    observed_actions: int
    unresolved_actions: int
    route: str


def coupled_fractional_imputation(
    actions: pd.DataFrame,
    unresolved_revival_fraction: float,
    *,
    observed_column: str = "outcome_observed",
    observed_label_column: str = "observed_label",
    partial_exposure_column: str = "horizon_mem_gib_hours",
    persistent_exposure_column: str = "persistent_horizon_mem_gib_hours",
) -> pd.DataFrame:
    """Impute an unresolved-outcome sensitivity point on a fixed action stream.

    The endpoints reproduce the two extreme scenarios used by the manuscript:
    ``q=0`` treats all unresolved outcomes as non-revivals and retains only
    trace-observed exposure; ``q=1`` treats them as revivals and extends their
    exposure to the declared persistence envelope. Intermediate values are a
    coupled fractional sensitivity path, not a missing-data probability model.
    """
    q = float(unresolved_revival_fraction)
    if not 0.0 <= q <= 1.0:
        raise ValueError("unresolved_revival_fraction must lie in [0, 1]")
    required = {
        observed_column,
        observed_label_column,
        partial_exposure_column,
        persistent_exposure_column,
    }
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise KeyError(f"Missing proposal-stream columns: {missing}")

    result = actions.copy()
    observed = result[observed_column].astype(bool)
    partial = (
        pd.to_numeric(result[partial_exposure_column], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    persistent = (
        pd.to_numeric(result[persistent_exposure_column], errors="coerce")
        .fillna(partial)
        .clip(lower=0.0)
    )
    persistent = np.maximum(partial.to_numpy(float), persistent.to_numpy(float))

    result["reactivates_within_horizon"] = pd.to_numeric(
        result[observed_label_column], errors="raise"
    ).astype(float)
    result.loc[~observed, "reactivates_within_horizon"] = q
    exposure = partial.to_numpy(float)
    exposure[~observed.to_numpy()] = (
        exposure[~observed.to_numpy()]
        + q
        * (
            persistent[~observed.to_numpy()]
            - exposure[~observed.to_numpy()]
        )
    )
    result["sensitivity_exposure_gib_hours"] = exposure
    return result


def route_from_extreme_bounds(
    lower: DualRiskBound,
    upper: DualRiskBound,
    *,
    action_target: float,
    resource_target: float,
) -> str:
    """Map extreme-scenario bounds to a fail-closed operational route."""
    lower_pass = (
        lower.action.upper <= action_target
        and lower.resource.upper <= resource_target
    )
    upper_pass = (
        upper.action.upper <= action_target
        and upper.resource.upper <= resource_target
    )
    if upper_pass:
        return "FORWARD_SUPPORTED_FULL_BOUND"
    if lower_pass:
        return "SHADOW_ONLY_UNRESOLVED_BOUND"
    return "EXPIRED_RECALIBRATE"


def audit_proposal_stream(
    actions: pd.DataFrame,
    *,
    family_confidence: float,
    action_target: float,
    resource_target: float,
) -> ProposalStreamAudit:
    """Audit both unresolved-outcome extremes on one frozen proposal stream."""
    if actions.empty:
        empty = dual_clustered_risk_bound(
            actions.assign(
                reactivates_within_horizon=pd.Series(dtype=float),
                sensitivity_exposure_gib_hours=pd.Series(dtype=float),
            ),
            family_confidence=family_confidence,
            exposure_column="sensitivity_exposure_gib_hours",
        )
        return ProposalStreamAudit(empty, empty, 0, 0, 0, "ABSTAIN_NO_ACTIONS")

    observed = actions["outcome_observed"].astype(bool)
    lower_frame = coupled_fractional_imputation(actions, 0.0)
    upper_frame = coupled_fractional_imputation(actions, 1.0)
    lower = dual_clustered_risk_bound(
        lower_frame,
        family_confidence=family_confidence,
        exposure_column="sensitivity_exposure_gib_hours",
    )
    upper = dual_clustered_risk_bound(
        upper_frame,
        family_confidence=family_confidence,
        exposure_column="sensitivity_exposure_gib_hours",
    )
    route = route_from_extreme_bounds(
        lower,
        upper,
        action_target=action_target,
        resource_target=resource_target,
    )
    return ProposalStreamAudit(
        lower=lower,
        upper=upper,
        proposed_actions=int(len(actions)),
        observed_actions=int(observed.sum()),
        unresolved_actions=int((~observed).sum()),
        route=route,
    )


def tipping_curve(
    actions: pd.DataFrame,
    fractions: np.ndarray,
    *,
    family_confidence: float,
    action_target: float,
    resource_target: float,
) -> pd.DataFrame:
    """Evaluate the declared coupled sensitivity path on a fixed grid."""
    rows: list[dict[str, float | bool]] = []
    for fraction in np.asarray(fractions, dtype=float):
        imputed = coupled_fractional_imputation(actions, float(fraction))
        bound = dual_clustered_risk_bound(
            imputed,
            family_confidence=family_confidence,
            exposure_column="sensitivity_exposure_gib_hours",
        )
        rows.append(
            {
                "unresolved_revival_fraction": float(fraction),
                "action_estimate": bound.action.estimate,
                "action_upper": bound.action.upper,
                "resource_estimate": bound.resource.estimate,
                "resource_upper": bound.resource.upper,
                "dual_pass": bool(
                    bound.action.upper <= action_target
                    and bound.resource.upper <= resource_target
                ),
            }
        )
    return pd.DataFrame(rows)

