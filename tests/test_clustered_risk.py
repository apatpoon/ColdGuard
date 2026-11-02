import numpy as np
import pandas as pd

from coldguard.clustered_risk import (
    clustered_ratio_upper_bound,
    dual_clustered_risk_bound,
)


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": [1, 1, 2, 2, 3, 3],
            "app_id": ["a", "b", "a", "b", "a", "b"],
            "model_id": ["m1", "m2", "m1", "m2", "m1", "m2"],
            "site_id": ["s"] * 6,
            "pd_role": ["r"] * 6,
            "reactivates_within_horizon": [0, 1, 0, 0, 1, 0],
            "remaining_mem_gib_hours": [1, 2, 1, 2, 1, 2],
        }
    )


def test_ratio_bound_targets_pooled_ratio_not_mean_of_daily_ratios():
    labels = np.array([1.0, 0.0, 0.0])
    weights = np.ones(3)
    bound = clustered_ratio_upper_bound(
        labels,
        weights,
        days=[1, 2, 2],
        deployments=["a", "b", "c"],
        confidence=0.95,
    )
    assert bound.estimate == 1 / 3
    assert bound.numerator == 1
    assert bound.denominator == 3


def test_dual_bound_uses_distinct_action_and_resource_estimands():
    bound = dual_clustered_risk_bound(_actions())
    assert bound.action.estimate == 2 / 6
    assert bound.resource.estimate == 3 / 9
    assert bound.action.upper >= bound.action.estimate
    assert bound.resource.upper >= bound.resource.estimate
    assert bound.per_risk_confidence == 0.975


def test_empty_action_set_abstains():
    empty = _actions().iloc[:0]
    bound = dual_clustered_risk_bound(empty)
    assert bound.action.upper == 1.0
    assert bound.resource.upper == 1.0
