import numpy as np
import pandas as pd

from coldguard.multihorizon import (
    attach_horizon_label,
    horizon_is_observed,
    monotonicity_audit,
)


def _frame():
    return pd.DataFrame(
        {
            "day": [10, 10, 10],
            "event_day": [13.0, np.nan, np.nan],
            "end_cold_day": [12, 15, 11],
        }
    )


def test_horizon_maturity_distinguishes_event_observation_and_censoring():
    observed = horizon_is_observed(_frame(), 3)
    assert observed.tolist() == [True, True, False]
    labeled = attach_horizon_label(_frame(), 3)
    assert labeled["reactivates_within_horizon"].tolist() == [1, 0]


def test_monotonicity_audit_detects_direct_model_crossing():
    result = monotonicity_audit(
        {1: np.array([0.1, 0.4]), 3: np.array([0.2, 0.3]), 7: np.array([0.5, 0.6])}
    )
    assert result["violation_rows"] == 1
    assert result["violation_rate"] == 0.5


def test_monotonicity_audit_accepts_survival_composition():
    result = monotonicity_audit(
        {1: np.array([0.1]), 3: np.array([0.2]), 7: np.array([0.5])}
    )
    assert result["violation_rate"] == 0.0

