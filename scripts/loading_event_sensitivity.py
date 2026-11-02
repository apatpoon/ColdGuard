from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GIB = float(2**30)


def _p95(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, 0.95)) if len(array) else float("nan")


def _join_trace_fields(actions: pd.DataFrame, definition: str) -> pd.DataFrame:
    path = (
        ROOT / "data/interim/cold_landmarks_all_strict.parquet"
        if definition == "sm_le_0.1_pct"
        else ROOT / "data/interim/cold_landmarks_all.parquet"
    )
    fields = pd.read_parquet(
        path,
        columns=[
            "episode_id",
            "day",
            "params_b",
            "avg_gpu_count",
            "instance_count",
            "base_id",
            "model_arch",
        ],
    )
    actions = actions.copy()
    actions["episode_id"] = actions["episode_id"].astype(str)
    fields["episode_id"] = fields["episode_id"].astype(str)
    return actions.merge(
        fields,
        left_on=["episode_id", "action_day"],
        right_on=["episode_id", "day"],
        how="left",
        validate="many_to_one",
    )


def _add_transfer_scenarios(
    frame: pd.DataFrame,
    bandwidth: float,
    overhead: float,
) -> pd.DataFrame:
    result = frame.copy()
    ranks = (
        pd.to_numeric(result["avg_gpu_count"], errors="coerce")
        .fillna(1.0)
        .round()
        .clip(lower=1.0)
    )
    result["parallel_ranks"] = ranks.astype(int)
    result["allocation_proxy_gib"] = result["model_gib"].astype(float)
    result["fp16_weight_gib"] = result["params_b"] * 1e9 * 2.0 / GIB
    result["int8_weight_gib"] = result["params_b"] * 1e9 / GIB
    result["fp16_per_rank_gib"] = result["fp16_weight_gib"] / result["parallel_ranks"]
    for volume in [
        "allocation_proxy_gib",
        "fp16_weight_gib",
        "int8_weight_gib",
        "fp16_per_rank_gib",
    ]:
        result[f"{volume.removesuffix('_gib')}_added_s"] = (
            overhead + result[volume] / bandwidth
        )
    return result


def _loo_rows(
    events: pd.DataFrame,
    definition: str,
    method: str,
    scenario: str,
    latency_column: str,
) -> list[dict[str, object]]:
    eligible = events.loc[events[latency_column].notna()].copy()
    rows: list[dict[str, object]] = []
    for omitted in eligible.itertuples(index=False):
        retained = eligible.loc[eligible["action_id"] != omitted.action_id]
        rows.append(
            {
                "cold_definition": definition,
                "horizon": 3,
                "method": method,
                "scenario": scenario,
                "omitted_action_id": omitted.action_id,
                "omitted_model_gib": float(omitted.model_gib),
                "events_retained": int(len(retained)),
                "loo_p95_added_s": _p95(retained[latency_column]),
            }
        )
    return rows


def run(output_dir: Path, bandwidth: float, overhead: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_parquet(ROOT / "artifacts/trace_simulation/selected_actions.parquet")
    definitions = ["sm_le_1.0_pct", "sm_le_0.1_pct"]
    methods = ["hazard_hierarchical", "direct_monotone"]
    scenario_columns = {
        "allocation_proxy": "allocation_proxy_added_s",
        "fp16_checkpoint": "fp16_weight_added_s",
        "int8_checkpoint": "int8_weight_added_s",
        "fp16_per_rank_parallel": "fp16_per_rank_added_s",
    }
    event_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []

    for definition in definitions:
        subset = selected.loc[
            (selected["cold_definition"] == definition)
            & (selected["horizon"] == 3)
            & selected["method"].isin(methods)
            & selected["revives_within_horizon"].astype(bool)
        ].copy()
        joined = _join_trace_fields(subset, definition)
        joined = _add_transfer_scenarios(joined, bandwidth, overhead)
        event_frames.append(joined)
        for method in methods:
            events = joined.loc[joined["method"] == method].copy()
            for scenario, latency_column in scenario_columns.items():
                eligible = events.loc[events[latency_column].notna()].copy()
                loo = _loo_rows(
                    events,
                    definition,
                    method,
                    scenario,
                    latency_column,
                )
                loo_rows.extend(loo)
                loo_values = [float(row["loo_p95_added_s"]) for row in loo]
                summary_rows.append(
                    {
                        "cold_definition": definition,
                        "horizon": 3,
                        "method": method,
                        "scenario": scenario,
                        "events_total": int(len(events)),
                        "events_with_volume": int(len(eligible)),
                        "volume_coverage": (
                            float(len(eligible) / len(events)) if len(events) else float("nan")
                        ),
                        "p50_added_s": float(eligible[latency_column].median()),
                        "p95_added_s": _p95(eligible[latency_column]),
                        "max_added_s": float(eligible[latency_column].max()),
                        "events_over_30s": int((eligible[latency_column] > 30.0).sum()),
                        "loo_p95_min_s": min(loo_values) if loo_values else float("nan"),
                        "loo_p95_max_s": max(loo_values) if loo_values else float("nan"),
                        "loo_p95_range_s": (
                            max(loo_values) - min(loo_values)
                            if loo_values
                            else float("nan")
                        ),
                    }
                )

    events = pd.concat(event_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    loo = pd.DataFrame(loo_rows)
    events.to_csv(output_dir / "event_level_transfer_scenarios.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    loo.to_csv(output_dir / "leave_one_event_out.csv", index=False)

    strict = summary.loc[summary["cold_definition"] == "sm_le_0.1_pct"]
    verification = {
        "bandwidth_gib_per_s": bandwidth,
        "fixed_overhead_s": overhead,
        "strict_hazard_events": int(
            strict.loc[
                (strict["method"] == "hazard_hierarchical")
                & (strict["scenario"] == "allocation_proxy"),
                "events_total",
            ].iloc[0]
        ),
        "strict_direct_events": int(
            strict.loc[
                (strict["method"] == "direct_monotone")
                & (strict["scenario"] == "allocation_proxy"),
                "events_total",
            ].iloc[0]
        ),
        "scope": (
            "finite event-set sensitivity; parameter-derived scenarios use only events "
            "with non-missing params_b; no production latency or transfer measurement"
        ),
    }
    (output_dir / "design.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="artifacts/revision_round2/loading_event_sensitivity"
    )
    parser.add_argument("--bandwidth", type=float, default=16.0)
    parser.add_argument("--overhead", type=float, default=2.0)
    args = parser.parse_args()
    run(ROOT / args.output_dir, args.bandwidth, args.overhead)


if __name__ == "__main__":
    main()

