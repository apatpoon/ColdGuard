import numpy as np
import pandas as pd

from coldguard.pilot import HierarchicalTabularEncoder


def test_group_oof_target_encoding_does_not_reuse_heldout_group_labels():
    frame = pd.DataFrame(
        {
            "cold_age": [1, 2, 1, 2, 1, 2, 1, 2],
            "app_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )
    target = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = frame["app_id"].to_numpy()
    encoder = HierarchicalTabularEncoder(["cold_age"], ["app_id"], strength=1.0)
    encoded = encoder.fit_transform_group_oof(frame, target, groups, n_splits=4)

    assert encoded.shape == (8, 3)
    assert np.isfinite(encoded).all()
    # The target-encoding column cannot simply reproduce the held-out label.
    assert not np.array_equal(encoded[:, 2], target.astype(np.float32))

