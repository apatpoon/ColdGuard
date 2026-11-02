import numpy as np
import pandas as pd
import pytest

from coldguard.survival import (
    RollingFold,
    advance_covariates,
    cumulative_incidence_from_hazards,
    interval_to_horizon_score,
    predict_cumulative_incidence_path,
)


def test_cumulative_incidence_matches_constant_hazard_solution():
    hazards = np.full((2, 3), 0.1)
    result = cumulative_incidence_from_hazards(hazards)
    np.testing.assert_allclose(result, [1 - 0.9**3, 1 - 0.9**3])


def test_cumulative_incidence_rejects_invalid_shape_and_probabilities():
    with pytest.raises(ValueError):
        cumulative_incidence_from_hazards(np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        cumulative_incidence_from_hazards(np.array([[1.1]]))


def test_advance_covariates_advances_clocks_and_preserves_missingness():
    frame = pd.DataFrame(
        {"cold_age": [2, 4], "days_since_last_active": [3.0, np.nan], "x": [9, 8]}
    )
    advanced = advance_covariates(frame, 5)
    assert advanced["cold_age"].tolist() == [7, 9]
    assert advanced.loc[0, "days_since_last_active"] == 8
    assert np.isnan(advanced.loc[1, "days_since_last_active"])
    assert advanced["x"].tolist() == [9, 8]


def test_rolling_fold_has_maturation_embargoes():
    conditions = RollingFold("x", 36, 72, 108).conditions(7)
    assert conditions == {
        "train": "day <= 29",
        "calibration": "day > 36 AND day <= 65",
        "test": "day > 72 AND day <= 101",
    }


def test_zero_offset_does_not_modify_covariates():
    frame = pd.DataFrame(
        {"cold_age": [2], "days_since_last_active": [3.0], "x": [9]}
    )
    pd.testing.assert_frame_equal(advance_covariates(frame, 0), frame)


def test_interval_score_is_coherent_across_horizons_and_intervals():
    predicted = np.array([1.0, 7.0, 30.0])
    score_1d = interval_to_horizon_score(predicted, 1)
    score_7d = interval_to_horizon_score(predicted, 7)
    assert np.all(np.diff(score_1d) < 0)
    assert np.all(score_7d >= score_1d)


def test_interval_score_rejects_invalid_values():
    with pytest.raises(ValueError):
        interval_to_horizon_score(np.array([1.0]), 0)
    with pytest.raises(ValueError):
        interval_to_horizon_score(np.array([np.nan]), 1)


class _IdentityEncoder:
    def transform(self, frame):
        return frame[["prior_active_days_30"]].to_numpy(float)


class _PathModel:
    def predict_proba(self, values):
        probability = np.clip(values[:, 0] / 10.0, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


def test_path_prediction_uses_declared_active_day_trajectory():
    frame = pd.DataFrame(
        {
            "cold_age": [2],
            "days_since_last_active": [3.0],
            "prior_active_days_30": [3.0],
        }
    )
    result = predict_cumulative_incidence_path(
        frame,
        _IdentityEncoder(),
        _PathModel(),
        3,
        np.array([[3.0, 2.0, 1.0]]),
    )
    np.testing.assert_allclose(result, [1.0 - 0.7 * 0.8 * 0.9])


def test_path_prediction_validates_shape_and_values():
    frame = pd.DataFrame(
        {
            "cold_age": [2],
            "days_since_last_active": [3.0],
            "prior_active_days_30": [3.0],
        }
    )
    with pytest.raises(ValueError):
        predict_cumulative_incidence_path(
            frame, _IdentityEncoder(), _PathModel(), 2, np.array([[3.0]])
        )
    with pytest.raises(ValueError):
        predict_cumulative_incidence_path(
            frame, _IdentityEncoder(), _PathModel(), 1, np.array([[-1.0]])
        )

