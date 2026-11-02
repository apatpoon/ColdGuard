import math

from coldguard.policy import PolicyChoice
from coldguard.window_selector import (
    WindowCandidate,
    familywise_confidence,
    select_window_candidate,
)


def _candidate(window_days: int, reclaimed: float) -> WindowCandidate:
    return WindowCandidate(
        window_days=window_days,
        calibration_start_day=1,
        calibration_end_day=90,
        calibration_mature_end_day=89,
        calibration_rows=1000,
        choice=PolicyChoice(
            policy="hazard_hierarchical",
            parameter=0.1,
            calibration_actions=200,
            calibration_weighted_risk=0.01,
            calibration_risk_ucb=0.05,
            calibration_unweighted_risk=0.01,
            calibration_unweighted_risk_ucb=0.05,
            calibration_reclaimed_gib_hours=reclaimed,
        ),
    )


def test_familywise_confidence_spends_alpha_across_windows():
    assert math.isclose(familywise_confidence(0.95, 1), 0.95)
    assert math.isclose(familywise_confidence(0.95, 2), 0.975)
    assert math.isclose(familywise_confidence(0.95, 3), 1.0 - 0.05 / 3.0)


def test_selector_uses_calibration_yield_and_breaks_ties_by_recency():
    assert select_window_candidate([]) is None
    chosen = select_window_candidate(
        [_candidate(30, 10.0), _candidate(60, 20.0), _candidate(90, 20.0)]
    )
    assert chosen is not None
    assert chosen.window_days == 60
