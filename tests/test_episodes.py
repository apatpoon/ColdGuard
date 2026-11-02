from pathlib import Path

import duckdb
import pandas as pd
import yaml

from coldguard.episodes import build_episodes


def test_episode_builder_labels_reactivation_and_censoring(tmp_path: Path):
    daily = pd.DataFrame(
        {
            "day": [0, 1, 2, 3, 4, 5, 6, 7],
            "app_id": ["a"] * 8,
            "model_id": ["m"] * 8,
            "site_id": ["s"] * 8,
            "pd_role": ["fused"] * 8,
            "base_id": ["b"] * 8,
            "cluster_id": ["c"] * 8,
            "gpu_type": ["g"] * 8,
            "model_arch": ["dense"] * 8,
            "base_arch": ["dense"] * 8,
            "variant_arch": ["dense"] * 8,
            "params_b": [7.0] * 8,
            "instance_count": [1] * 8,
            "cluster_count": [1] * 8,
            "gpu_type_count": [1] * 8,
            "gpu_hours": [24.0] * 8,
            "observed_hours": [24.0] * 8,
            "sm_util": [10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0],
            "zero_fraction": [0.1, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0],
            "mem_gib_hours": [240.0] * 8,
            "avg_mem_gib_calendar": [10.0] * 8,
            "avg_gpu_count": [1.0] * 8,
            "is_cold": [False, True, True, False, False, True, True, True],
        }
    )
    daily_path = tmp_path / "daily.parquet"
    daily.to_parquet(daily_path, index=False)
    config = {
        "paths": {
            "deployment_daily": str(daily_path),
            "episodes": str(tmp_path / "episodes.parquet"),
            "landmarks": str(tmp_path / "landmarks.parquet"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "data": {"logical_keys": ["app_id", "model_id", "site_id", "pd_role"]},
        "label": {"horizon_days": 1},
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    episodes_path, landmarks_path = build_episodes(config_path)
    con = duckdb.connect()
    episodes = con.execute(f"SELECT * FROM read_parquet('{episodes_path}') ORDER BY start_day").fetchdf()
    landmarks = con.execute(f"SELECT * FROM read_parquet('{landmarks_path}') ORDER BY day").fetchdf()
    con.close()

    assert episodes["reactivated"].tolist() == [True, False]
    assert landmarks.loc[landmarks["day"] == 2, "reactivates_within_horizon"].item() == 1
    assert landmarks.loc[landmarks["day"] == 1, "remaining_mem_gib_hours"].item() == 480.0

