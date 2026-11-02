from __future__ import annotations

import pandas as pd

from coldguard.agentic_external import build_gap_landmarks


def test_gap_landmarks_respect_horizon_and_censoring() -> None:
    hour = 3_600_000
    requests = pd.DataFrame(
        {
            "session_id": ["a", "b", "c"],
            "step": [0, 0, 0],
            "start_ms": [0, 8 * hour, 40 * hour],
            "end_ms": [0, 8 * hour, 40 * hour],
            "ttft_ms": [1, 1, 1],
            "model": ["m", "m", "m"],
            "provider": ["p", "p", "p"],
            "prompt_tokens": [1, 1, 1],
            "completion_tokens": [1, 1, 1],
            "cache_read_tokens": [0, 0, 0],
            "n_messages": [1, 1, 1],
        }
    )
    landmarks = build_gap_landmarks(
        requests, horizon_hours=6, decision_ages_hours=(1, 6, 12, 24)
    )
    first = landmarks.loc[landmarks["episode_id"] == "m|p|0"].set_index("cold_age_h")
    assert first.loc[1.0, "reactivates_within_horizon"] == 0
    assert first.loc[6.0, "reactivates_within_horizon"] == 1
    second = landmarks.loc[landmarks["episode_id"] == "m|p|1"].set_index("cold_age_h")
    assert second.loc[24.0, "reactivates_within_horizon"] == 0
