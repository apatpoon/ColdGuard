from coldguard.robustness import OperationalFold, fallback_route


def test_rolling_fold_respects_horizon_maturity():
    fold = OperationalFold("early", 61, 90, 91, 120)
    assert fold.conditions(7) == {
        "calibration": "day >= 61 AND day <= 83",
        "test": "day >= 91 AND day <= 113",
    }


def test_fallback_route_prefers_validated_global_then_group_then_abstain():
    safe = {"actions": 60, "unweighted_risk_ucb": 0.08, "risk_ucb": 0.09}
    unsafe = {"actions": 60, "unweighted_risk_ucb": 0.12, "risk_ucb": 0.09}
    sparse = {"actions": 20, "unweighted_risk_ucb": 0.01, "risk_ucb": 0.01}
    assert fallback_route(safe, True, 50, 0.10, 0.10) == "global_validated"
    assert fallback_route(unsafe, True, 50, 0.10, 0.10) == "group_specific"
    assert fallback_route(sparse, False, 50, 0.10, 0.10) == "abstain"
