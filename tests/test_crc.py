import numpy as np
import pandas as pd

from coldguard.crc import (
    choose_grouped_crc_threshold,
    crc_corrected_risk,
    grouped_loss_tables,
    monotonize_losses,
)


def _frame():
    return pd.DataFrame(
        {
            "app_id": ["a", "a", "b"],
            "model_id": ["m", "m", "n"],
            "site_id": ["s", "s", "s"],
            "pd_role": ["f", "f", "f"],
            "episode_id": [1, 1, 2],
            "day": [1, 2, 1],
            "score": [0.8, 0.2, 0.3],
            "reactivates_within_horizon": [0, 1, 0],
            "remaining_mem_gib_hours": [100.0, 50.0, 80.0],
        }
    )


def test_monotonize_losses_is_conservative_and_ordered():
    raw = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = monotonize_losses(raw)
    np.testing.assert_array_equal(result, [[0, 1, 1], [0, 0, 1]])
    assert np.all(result >= raw)


def test_crc_correction_matches_paper_formula():
    losses = np.array([[0.0], [1.0], [0.0]])
    np.testing.assert_allclose(crc_corrected_risk(losses), [(3 / 4) * (1 / 3) + 1 / 4])


def test_grouped_tables_monotonize_first_action_timing():
    action, resource, counts = grouped_loss_tables(_frame(), "score", np.array([0.2, 0.8]))
    assert action.shape == (2, 2)
    assert np.all(np.diff(action, axis=1) >= 0)
    assert np.all(np.diff(resource, axis=1) >= 0)
    assert counts.tolist() == [1, 2]


def test_grouped_crc_can_abstain_when_correction_exceeds_target():
    choice = choose_grouped_crc_threshold(
        _frame(), "score", np.array([0.2, 0.8]), 0.1, 0.1, minimum_actions=1
    )
    assert choice is None

