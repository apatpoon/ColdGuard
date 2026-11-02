from coldguard.calibration_window import EvaluationFold, calibration_window


def test_calibration_window_applies_horizon_maturity():
    fold = EvaluationFold("middle", 121, 150)
    window = calibration_window(fold, 60, horizon=7, earliest_day=61)
    assert window is not None
    assert window.start_day == 61
    assert window.end_day == 120
    assert window.mature_end_day == 113
    assert window.mature_calendar_days == 53
    assert window.condition() == "day >= 61 AND day <= 113"


def test_calibration_window_rejects_unavailable_history():
    fold = EvaluationFold("early", 91, 120)
    assert calibration_window(fold, 60, horizon=1, earliest_day=61) is None
