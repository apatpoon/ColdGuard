import numpy as np

from coldguard.block_ltt import (
    aggregate_time_blocks,
    hoeffding_bentkus_p_value,
    select_ltt_threshold,
)


def test_zero_loss_can_reject_with_enough_daily_blocks():
    assert hoeffding_bentkus_p_value(0.0, 30, 0.1) < 0.05
    assert hoeffding_bentkus_p_value(0.0, 5, 0.1) > 0.05


def test_time_block_aggregation_preserves_bounds():
    daily = np.array([[0.0, 0.2], [0.2, 0.4], [1.0, 0.0]])
    blocks = aggregate_time_blocks(daily, 2)
    np.testing.assert_allclose(blocks, [[0.1, 0.3], [1.0, 0.0]])


def test_fixed_sequence_stops_at_first_non_rejection():
    action = np.column_stack([np.zeros(30), np.full(30, 0.5), np.zeros(30)])
    resource = action.copy()
    choice = select_ltt_threshold(
        action,
        resource,
        np.array([0.0, 0.5, 1.0]),
        np.array([250, 300, 350]),
        0.1,
        0.1,
        0.05,
        200,
        "fixed_sequence",
    )
    assert choice is not None
    assert choice.threshold == 0.0

