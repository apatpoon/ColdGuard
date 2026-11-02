from __future__ import annotations

import pandas as pd

from coldguard.chutes_external import (
    build_daily_landmarks,
    calibrate_chutes_policies,
    load_model_metadata,
    rolling_recalibration_audit,
)


def test_chutes_daily_landmarks_respect_future_horizons(tmp_path) -> None:
    daily = pd.DataFrame(
        {
            "chute_id": ["a", "a", "a", "b", "b"],
            "activity_date": pd.to_datetime(
                ["1970-01-01", "1970-01-04", "1970-01-10", "1970-01-02", "1970-01-10"]
            ),
            "requests": [10, 20, 5, 3, 2],
        }
    )
    mapping_path = tmp_path / "models.csv"
    mapping_path.write_text(
        "chute_id,name,invocation_count,current_estimated_price,supported_gpus\n"
        'a,owner/model-a,35,2.5,"[\'a100\', \'h100\']"\n',
        encoding="utf-8",
    )
    mapping = load_model_metadata(mapping_path)
    landmarks, audit = build_daily_landmarks(daily, mapping)
    first = landmarks.loc[
        landmarks["chute_id"].eq("a") & landmarks["decision_date"].eq("1970-01-02")
    ].iloc[0]
    assert first["cold_age"] == 1
    assert first["revives_1d"] == 0
    assert first["revives_3d"] == 1
    assert first["requests_prior_1d"] == 10
    assert first["supported_gpu_count"] == 2
    assert audit["observed_chutes"] == 2
    assert audit["mapped_chutes"] == 1


def test_chutes_tail_is_right_censored(tmp_path) -> None:
    daily = pd.DataFrame(
        {
            "chute_id": ["a", "a", "b"],
            "activity_date": pd.to_datetime(["1970-01-01", "1970-01-05", "1970-01-06"]),
            "requests": [1, 1, 1],
        }
    )
    mapping_path = tmp_path / "models.csv"
    mapping_path.write_text(
        "chute_id,name,invocation_count,current_estimated_price,supported_gpus\n",
        encoding="utf-8",
    )
    landmarks, _ = build_daily_landmarks(daily, load_model_metadata(mapping_path))
    tail = landmarks.loc[
        landmarks["chute_id"].eq("a") & landmarks["decision_date"].eq("1970-01-06")
    ].iloc[0]
    assert pd.isna(tail["revives_1d"])
    assert pd.isna(tail["revives_30d"])


def test_chutes_policy_counts_one_action_per_episode() -> None:
    rows = []
    for split, day_start in [("calibration", 60), ("test", 120)]:
        for episode in range(240):
            for offset in [0, 1]:
                label = int(episode < 3)
                rows.append(
                    {
                        "episode_id": episode + (0 if split == "calibration" else 1_000),
                        "chute_id": f"c{episode}",
                        "day": day_start + offset,
                        "decision_date": pd.Timestamp("1970-01-01")
                        + pd.to_timedelta(day_start + offset, unit="D"),
                        "split": split,
                        "cold_age": 30 + offset,
                        "time_to_next_days": 1.0 if label else float("nan"),
                        "seen_in_train": 1,
                        **{f"revives_{h}d": label for h in [1, 3, 7, 14, 30]},
                        **{
                            f"score_{method}_{h}d": 0.01 if not label else 0.9
                            for method in ["age_hazard", "hierarchical_hazard"]
                            for h in [1, 3, 7, 14, 30]
                        },
                    }
                )
    scores = pd.DataFrame(rows)
    _, selected = calibrate_chutes_policies(
        scores, target_risk=0.05, minimum_actions=200, threshold_grid_size=10
    )
    age_one_day = selected.loc[
        selected["policy"].eq("age_hazard") & selected["horizon_days"].eq(1)
    ].iloc[0]
    assert bool(age_one_day["selected"])
    assert age_one_day["cal_actions"] <= 240


def test_rolling_calibration_ends_before_test_label_window() -> None:
    rows = []
    for day in range(120, 210):
        rows.append(
            {
                "episode_id": day,
                "chute_id": f"c{day}",
                "day": day,
                "decision_date": pd.Timestamp("1970-01-01")
                + pd.to_timedelta(day, unit="D"),
                "split": "forward",
                "cold_age": 10,
                "time_to_next_days": float("nan"),
                "seen_in_train": 0,
                **{f"revives_{h}d": 0 for h in [1, 3, 7, 14, 30]},
                **{
                    f"score_{method}_{h}d": 0.01
                    for method in ["age_hazard", "hierarchical_hazard"]
                    for h in [1, 3, 7, 14, 30]
                },
            }
        )
    audit = rolling_recalibration_audit(
        pd.DataFrame(rows), minimum_actions=1, threshold_grid_size=3
    )
    one_day = audit.loc[
        audit["test_start_day"].eq(180) & audit["horizon_days"].eq(1)
    ]
    assert one_day["calibration_end_day"].eq(178).all()
    thirty_day = audit.loc[
        audit["test_start_day"].eq(180) & audit["horizon_days"].eq(30)
    ]
    assert thirty_day["calibration_end_day"].eq(149).all()
