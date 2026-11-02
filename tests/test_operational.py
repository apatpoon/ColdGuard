from coldguard.operational import operational_conditions


def test_operational_windows_use_all_horizon_matured_days():
    assert operational_conditions(60, 120, 180, 1) == {
        "calibration": "day > 60 AND day <= 119",
        "test": "day > 120 AND day <= 179",
    }
    assert operational_conditions(60, 120, 180, 30) == {
        "calibration": "day > 60 AND day <= 90",
        "test": "day > 120 AND day <= 150",
    }

