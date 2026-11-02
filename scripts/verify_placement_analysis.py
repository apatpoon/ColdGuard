from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/placement_analysis"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    summary = pd.read_csv(ARTIFACT / "scenario_summary.csv")
    daily = pd.read_csv(ARTIFACT / "daily_results.csv")
    blocks = pd.read_parquet(ARTIFACT / "action_blocks.parquet")
    audit = json.loads((ARTIFACT / "reconstruction_audit.json").read_text())
    checks: dict[str, object] = {}

    require(len(summary) == 252, "unexpected placement scenario count")
    checks["scenario_rows"] = int(len(summary))
    require(len(daily) == 8064, "unexpected daily placement row count")
    checks["daily_rows"] = int(len(daily))
    require(int(audit["unique_actions"]) == 2357, "unexpected reconstructed action count")
    require(int(audit["missing_actions"]) == 0, "placement actions are missing blocks")
    require(
        float(audit["maximum_capacity_conservation_relative_error"]) < 1e-12,
        "placement block reconstruction violates memory conservation",
    )
    checks["reconstruction_audit"] = audit

    require(
        bool(
            (
                daily["global_pool_replicas"]
                >= daily["site_gpu_pool_replicas"]
            ).all()
            and (
                daily["site_gpu_pool_replicas"]
                >= daily["cluster_gpu_pool_replicas"]
            ).all()
            and (
                daily["cluster_gpu_pool_replicas"]
                >= daily["block_packed_replicas"]
            ).all()
        ),
        "daily placement bounds are not nested",
    )
    checks["nested_daily_bounds"] = True
    require((blocks["block_gib"] >= 0).all(), "negative virtual block capacity")
    checks["nonnegative_virtual_blocks"] = True

    canonical = summary.loc[
        summary["cold_definition"].eq("sm_le_0.1_pct")
        & summary["horizon"].eq(3)
        & summary["method"].eq("hazard_hierarchical")
        & summary["precision"].eq("FP16/BF16")
        & np.isclose(summary["runtime_headroom"], 0.20)
    ].set_index("template")
    expected = {
        "7B": (4606, 4477, 4463, 3919, 0.850847),
        "13B": (2472, 2364, 2342, 1682, 0.680421),
        "32B": (997, 855, 835, 546, 0.547643),
        "70B": (447, 336, 324, 186, 0.416107),
    }
    canonical_record: dict[str, object] = {}
    for template, values in expected.items():
        row = canonical.loc[template]
        actual = (
            int(row["global_pool_replica_days"]),
            int(row["site_gpu_pool_replica_days"]),
            int(row["cluster_gpu_pool_replica_days"]),
            int(row["block_packed_replica_days"]),
        )
        require(actual == values[:4], f"canonical placement mismatch for {template}")
        require(
            np.isclose(float(row["placement_retention"]), values[4], atol=5e-7),
            f"canonical retention mismatch for {template}",
        )
        canonical_record[template] = {
            "global": actual[0],
            "site_gpu": actual[1],
            "cluster_gpu": actual[2],
            "block_packed": actual[3],
            "retention": float(row["placement_retention"]),
        }
    checks["strict_h3_fp16_headroom20"] = canonical_record

    precision = summary.loc[
        summary["cold_definition"].eq("sm_le_0.1_pct")
        & summary["horizon"].eq(3)
        & summary["method"].eq("hazard_hierarchical")
        & np.isclose(summary["runtime_headroom"], 0.20)
    ].pivot(index="template", columns="precision", values="block_packed_replica_days")
    expected_precision = {
        "7B": {"INT8": 8207, "INT4": 17054},
        "13B": {"INT8": 4190, "INT4": 9203},
        "32B": {"INT8": 1489, "INT4": 3219},
        "70B": {"INT8": 493, "INT4": 1080},
    }
    for template, values in expected_precision.items():
        for precision_name, expected_value in values.items():
            require(
                int(precision.loc[template, precision_name]) == expected_value,
                f"precision sensitivity mismatch for {template}/{precision_name}",
            )
    checks["precision_sensitivity"] = expected_precision

    figure_source = ROOT / "figures/figure_placement_analysis_source.csv"
    require(figure_source.exists(), "placement figure source table is missing")
    checks["figure_source_rows"] = int(len(pd.read_csv(figure_source)))

    destination = ARTIFACT / "claim_verification_placement.json"
    destination.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
