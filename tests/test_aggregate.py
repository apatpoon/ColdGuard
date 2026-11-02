from pathlib import Path

import duckdb
import pandas as pd
import yaml

from coldguard.aggregate import aggregate_daily


def test_raw_segments_are_aggregated_into_deployment_days(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "instance_daily"
    raw_dir.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "day": [0, 0, 1],
            "instance_id": ["i1", "i2", "i1"],
            "app_id": ["a", "a", "a"],
            "model_id": ["m", "m", "m"],
            "base_id": ["b", "b", "b"],
            "cluster_id": ["c", "c", "c"],
            "site_id": ["s", "s", "s"],
            "gpu_type": ["g", "g", "g"],
            "workload": ["online", "online", "online"],
            "model_arch": ["dense", "dense", "dense"],
            "base_arch": ["dense", "dense", "dense"],
            "variant_arch": ["dense", "dense", "dense"],
            "params_b": [7.0, 7.0, 7.0],
            "pd_role": ["fused", "fused", "fused"],
            "gpu_count": [1.0, 1.0, 1.0],
            "gpu_hours": [12.0, 12.0, 24.0],
            "samples": [100, 100, 100],
            "zero_samples": [100, 100, 10],
            "sm_util": [0.0, 0.5, 20.0],
            "gpu_mem_alloc": [10.0, 20.0, 10.0],
        }
    )
    raw.to_parquet(raw_dir / "part_0.parquet", index=False)
    config = {
        "paths": {
            "raw_daily_glob": str(raw_dir / "part_*.parquet"),
            "deployment_daily": str(tmp_path / "deployment.parquet"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "data": {
            "workload": "online",
            "day_min": 0,
            "day_max": 1,
            "logical_keys": ["app_id", "model_id", "site_id", "pd_role"],
        },
        "cold_definition": {
            "sm_util_max": 1.0,
            "zero_fraction_min": 0.9,
            "min_observed_hours": 1.0,
            "min_mem_gib_hours": 1.0,
        },
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output = aggregate_daily(config_path)
    con = duckdb.connect()
    result = con.execute(f"SELECT * FROM read_parquet('{output}') ORDER BY day").fetchdf()
    con.close()

    assert result["is_cold"].tolist() == [True, False]
    assert result.loc[0, "mem_gib_hours"] == 360.0
    assert result.loc[0, "instance_count"] == 2

