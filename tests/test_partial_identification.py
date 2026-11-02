import numpy as np
import pandas as pd

from coldguard.partial_identification import (
    audit_proposal_stream,
    coupled_fractional_imputation,
    route_from_extreme_bounds,
    tipping_curve,
)


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": [1, 2, 3, 4],
            "app_id": ["a", "b", "c", "d"],
            "model_id": ["m1", "m2", "m3", "m4"],
            "site_id": ["s"] * 4,
            "pd_role": ["r"] * 4,
            "outcome_observed": [True, True, False, False],
            "observed_label": [0, 1, 0, 0],
            "horizon_mem_gib_hours": [10.0, 20.0, 5.0, 5.0],
            "persistent_horizon_mem_gib_hours": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_fractional_imputation_reproduces_declared_endpoints():
    lower = coupled_fractional_imputation(_actions(), 0.0)
    upper = coupled_fractional_imputation(_actions(), 1.0)
    assert lower["reactivates_within_horizon"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert upper["reactivates_within_horizon"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert lower["sensitivity_exposure_gib_hours"].tolist() == [10.0, 20.0, 5.0, 5.0]
    assert upper["sensitivity_exposure_gib_hours"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_fractional_path_is_between_endpoints():
    middle = coupled_fractional_imputation(_actions(), 0.5)
    assert middle["reactivates_within_horizon"].tolist() == [0.0, 1.0, 0.5, 0.5]
    assert middle["sensitivity_exposure_gib_hours"].tolist() == [10.0, 20.0, 17.5, 22.5]


def test_audit_is_fail_closed_when_only_optimistic_endpoint_passes():
    audit = audit_proposal_stream(
        _actions(), family_confidence=0.95, action_target=0.95, resource_target=0.95
    )
    assert audit.proposed_actions == 4
    assert audit.observed_actions == 2
    assert audit.unresolved_actions == 2
    assert audit.route in {
        "FORWARD_SUPPORTED_FULL_BOUND",
        "SHADOW_ONLY_UNRESOLVED_BOUND",
        "EXPIRED_RECALIBRATE",
    }
    assert route_from_extreme_bounds(
        audit.lower,
        audit.upper,
        action_target=0.95,
        resource_target=0.95,
    ) == audit.route


def test_tipping_curve_contains_endpoints_and_declared_grid():
    curve = tipping_curve(
        _actions(),
        np.array([0.0, 0.5, 1.0]),
        family_confidence=0.95,
        action_target=0.9,
        resource_target=0.9,
    )
    assert curve["unresolved_revival_fraction"].tolist() == [0.0, 0.5, 1.0]
    assert curve.loc[0, "action_estimate"] < curve.loc[2, "action_estimate"]
    assert curve.loc[0, "resource_estimate"] < curve.loc[2, "resource_estimate"]

