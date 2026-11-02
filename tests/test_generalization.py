import pandas as pd

from coldguard.generalization import assign_novelty_bucket


def test_novelty_buckets_are_hierarchical_and_mutually_exclusive():
    frame = pd.DataFrame(
        {
            "app_id": ["a", "x", "x", "x"],
            "model_id": ["m", "m", "new-m", "newer-m"],
            "base_id": ["b", "b", "b", "new-b"],
            "site_id": ["s", "new-s", "s", "s"],
            "pd_role": ["f", "f", "f", "f"],
        }
    )
    result = assign_novelty_bucket(
        frame,
        {"a|m|s|f"},
        {"m"},
        {"b"},
    )
    assert result.tolist() == [
        "seen_deployment",
        "unseen_deployment_seen_model",
        "unseen_model_seen_base",
        "unseen_base",
    ]

