from __future__ import annotations

import numpy as np
import pandas as pd

from coldguard.gentd_external import assign_splits, build_gentd_landmarks


def test_gentd_landmark_detects_one_hour_revival() -> None:
    timestamps = np.arange(0, 8_000, 60, dtype=float)
    duty = np.zeros(len(timestamps), dtype=float)
    duty[timestamps >= 3_000] = 20.0
    aligned = pd.DataFrame(
        {
            "container_ip": "c1",
            "timestamp_anon": timestamps,
            "gpu_duty": duty,
            "gpu_memory_bytes": 20 * 1024**3,
            "memory_gib": 20.0,
        }
    )
    landmarks, _ = build_gentd_landmarks(aligned)
    assert not landmarks.empty
    assert landmarks["reactivates_within_horizon"].eq(1).all()


def test_episode_revival_is_aggregated_across_landmarks() -> None:
    timestamps = np.arange(0, 11_000, 60, dtype=float)
    duty = np.zeros(len(timestamps), dtype=float)
    duty[timestamps >= 6_000] = 20.0
    aligned = pd.DataFrame(
        {
            "container_ip": "c1",
            "timestamp_anon": timestamps,
            "gpu_duty": duty,
            "gpu_memory_bytes": 20 * 1024**3,
            "memory_gib": 20.0,
        }
    )
    landmarks, episodes = build_gentd_landmarks(aligned)
    assert landmarks["reactivates_within_horizon"].eq(0).any()
    assert landmarks["reactivates_within_horizon"].eq(1).any()
    assert episodes.loc[0, "revival_timestamp"] == 6_000


def test_cross_split_episode_is_audited_without_label_maturity_leakage() -> None:
    frame = pd.DataFrame(
        {
            "episode_id": ["anchor-left", "cross", "cross", "anchor-right"],
            "decision_timestamp": [0.0, 5_000.0, 10_000.0, 19_000.0],
        }
    )
    assigned, audit = assign_splits(frame)
    cross = assigned.loc[assigned["episode_id"] == "cross"].sort_values(
        "decision_timestamp"
    )
    assert cross["split"].tolist() == ["train", "calibration"]
    assert audit["cross_split_episode_count"] == 1
