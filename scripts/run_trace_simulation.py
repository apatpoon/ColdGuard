from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from coldguard.policy import first_actions_by_score, first_actions_by_ttl
from coldguard.simulator import SimulationConfig, replay_scenarios


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/trace_simulation"
DEFINITIONS = {
    "sm_le_1.0_pct": {
        "label": "Primary (SM <= 1%)",
        "predictions": ROOT / "artifacts/multihorizon/test_predictions.parquet",
        "landmarks": ROOT / "data/interim/cold_landmarks_all.parquet",
    },
    "sm_le_0.1_pct": {
        "label": "Strict (SM <= 0.1%)",
        "predictions": ROOT / "artifacts/multihorizon_strict/test_predictions.parquet",
        "landmarks": ROOT / "data/interim/cold_landmarks_all_strict.parquet",
    },
}
POLICY_LABELS = {
    "always_retain": "Always retain",
    "ttl": "TTL",
    "hazard_hierarchical": "ColdGuard hazard",
    "direct_monotone": "Direct + projection",
    "recency_frequency": "Recency-frequency",
    "oracle": "Horizon oracle",
}


def _load_stage3_frame(definition: str, horizon: int) -> pd.DataFrame:
    info = DEFINITIONS[definition]
    predictions = pd.read_parquet(info["predictions"]).copy()
    landmarks = pd.read_parquet(info["landmarks"]).sort_values(
        ["episode_id", "day"]
    )
    exposure = np.zeros(len(landmarks), dtype=float)
    groups = landmarks.groupby("episode_id", observed=True, sort=False)["mem_gib_hours"]
    for offset in range(horizon):
        exposure += groups.shift(-offset).fillna(0.0).to_numpy(float)
    landmarks = landmarks.assign(horizon_mem_gib_hours=exposure)
    join_columns = [
        "episode_id",
        "day",
        "event_day",
        "end_cold_day",
        "avg_mem_gib_calendar",
        "prior_active_days_30",
        "horizon_mem_gib_hours",
    ]
    if landmarks[["episode_id", "day"]].duplicated().any():
        raise ValueError("landmark episode/day keys are not unique")
    frame = predictions.merge(
        landmarks[join_columns], on=["episode_id", "day"], how="left", validate="one_to_one"
    )
    if frame["avg_mem_gib_calendar"].isna().any():
        raise ValueError("stage-3 predictions failed to join landmark memory fields")
    frame["reactivates_within_horizon"] = frame[f"label_{horizon}d"].astype(int)
    frame["score_hazard_hierarchical"] = frame[
        f"score_hazard_hierarchical_{horizon}d"
    ]
    direct_columns = [
        f"score_direct_hierarchical_{value}d"
        for value in [1, 3, 7]
        if value <= horizon
    ]
    frame["score_direct_monotone"] = frame[direct_columns].max(axis=1)
    frame["score_recency_frequency"] = (
        frame["prior_active_days_30"].fillna(0.0) + 1.0
    ) / (
        frame["prior_active_days_30"].fillna(0.0)
        + frame["cold_age"].clip(lower=1.0)
        + 2.0
    )
    return frame.sort_values(["episode_id", "day"]).reset_index(drop=True)


def _action_columns(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "action_id",
                "action_day",
                "model_gib",
                "exposure_gib_hours",
                "revives_within_horizon",
                "revival_day",
                "episode_id",
                "app_id",
                "model_id",
                "site_id",
                "gpu_type",
                "method",
            ]
        )
    output = frame[
        [
            "episode_id",
            "day",
            "avg_mem_gib_calendar",
            "horizon_mem_gib_hours",
            "reactivates_within_horizon",
            "event_day",
            "app_id",
            "model_id",
            "site_id",
            "gpu_type",
        ]
    ].copy()
    output["action_id"] = (
        output["episode_id"].astype(str) + ":" + output["day"].astype(str)
    )
    output["episode_id"] = output["episode_id"].astype(str)
    output["arrival_key"] = output["episode_id"]
    output.rename(
        columns={
            "day": "action_day",
            "avg_mem_gib_calendar": "model_gib",
            "horizon_mem_gib_hours": "exposure_gib_hours",
            "reactivates_within_horizon": "revives_within_horizon",
            "event_day": "revival_day",
        },
        inplace=True,
    )
    output["model_gib"] = output["model_gib"].clip(lower=0.0)
    output["exposure_gib_hours"] = output["exposure_gib_hours"].clip(lower=0.0)
    output["method"] = method
    return output


def _select_actions(
    frame: pd.DataFrame,
    method: str,
    policy_row: pd.Series | None,
) -> pd.DataFrame:
    if method == "always_retain":
        return _action_columns(frame.iloc[0:0], method)
    if method == "oracle":
        safe = frame.loc[frame["reactivates_within_horizon"] == 0].copy()
        safe = safe.sort_values(["episode_id", "day"]).drop_duplicates(
            "episode_id", keep="first"
        )
        return _action_columns(safe, method)
    if policy_row is None or not bool(policy_row["calibration_qualified"]):
        return _action_columns(frame.iloc[0:0], method)
    parameter = float(policy_row["selected_parameter"])
    if method == "ttl":
        selected = first_actions_by_ttl(frame, int(round(parameter)))
    else:
        score_columns = {
            "hazard_hierarchical": "score_hazard_hierarchical",
            "direct_monotone": "score_direct_monotone",
            "recency_frequency": "score_recency_frequency",
        }
        selected = first_actions_by_score(frame, score_columns[method], parameter)
    return _action_columns(selected, method)


def _confidence_interval(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        mean = float(clean.mean()) if len(clean) else 0.0
        return mean, mean
    mean = float(clean.mean())
    standard_error = float(clean.std(ddof=1) / math.sqrt(len(clean)))
    radius = float(t.ppf(0.975, len(clean) - 1) * standard_error)
    return mean - radius, mean + radius


def _summarize(replications: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "cold_definition",
        "definition_label",
        "horizon",
        "method",
        "method_label",
        "qualification_route_full_forward",
        "bandwidth_gib_per_s",
        "load_slots",
        "fixed_overhead_s",
        "arrival_spread_s",
    ]
    metric_columns = [
        "actions",
        "reloads",
        "successful_actions",
        "simulated_released_gib_hours",
        "safe_released_gib_hours",
        "revival_exposure_gib_hours",
        "reload_volume_gib",
        "action_risk",
        "exposure_risk",
        "mean_queue_wait_s",
        "p95_queue_wait_s",
        "mean_added_latency_s",
        "p50_added_latency_s",
        "p95_added_latency_s",
        "p99_added_latency_s",
        "max_queue_length",
        "load_slot_utilization",
        "reload_slo_violation_rate_1s",
        "reload_slo_violation_rate_3s",
        "reload_slo_violation_rate_5s",
        "reload_slo_violation_rate_10s",
        "reload_slo_violation_rate_30s",
        "action_slo_violation_rate_1s",
        "action_slo_violation_rate_3s",
        "action_slo_violation_rate_5s",
        "action_slo_violation_rate_10s",
        "action_slo_violation_rate_30s",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in replications.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, keys))
        row["replications"] = int(group["seed"].nunique())
        for metric in metric_columns:
            row[metric] = float(group[metric].mean())
        for metric in [
            "mean_added_latency_s",
            "p95_queue_wait_s",
            "p95_added_latency_s",
            "p99_added_latency_s",
            "reload_slo_violation_rate_30s",
            "action_slo_violation_rate_30s",
        ]:
            low, high = _confidence_interval(group[metric])
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    canonical = summary[
        (summary["bandwidth_gib_per_s"] == 16.0)
        & (summary["load_slots"] == 2)
        & (summary["arrival_spread_s"] == 3_600.0)
        & (summary["horizon"] == 3)
    ].copy()
    display = canonical[
        [
            "definition_label",
            "method_label",
            "qualification_route_full_forward",
            "actions",
            "simulated_released_gib_hours",
            "reloads",
            "p95_added_latency_s",
            "reload_slo_violation_rate_30s",
        ]
    ]
    headers = [
        "Definition",
        "Method",
        "Route",
        "Actions",
        "Released GiB-h",
        "Reloads",
        "p95 latency (s)",
        "Reload SLO>30s",
    ]
    table = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in display.itertuples(index=False):
        table.append(
            "| "
            + " | ".join(
                [
                    str(row.definition_label),
                    str(row.method_label),
                    str(row.qualification_route_full_forward),
                    f"{float(row.actions):.0f}",
                    f"{float(row.simulated_released_gib_hours):.1f}",
                    f"{float(row.reloads):.0f}",
                    f"{float(row.p95_added_latency_s):.3f}",
                    f"{100.0 * float(row.reload_slo_violation_rate_30s):.2f}%",
                ]
            )
            + " |"
        )
    lines = [
        "# Trace-driven counterfactual simulation",
        "",
        "The simulator replays the stage-3 daily test window (days 121--150) in a logical GPU loading pool. It uses observed cold episodes, revival days, allocated-memory exposure, and frozen policy thresholds. Daily revival times are ordered by seed-specific common random numbers. Results are model-based scenario contrasts, not observed production interventions.",
        "",
        "## Canonical scenario",
        "",
        "Loading bandwidth: 16 GiB/s; parallel loading slots: 2; revival arrivals distributed within one hour; fixed overhead: 2 s; horizon: 3 days.",
        "",
        "\n".join(table),
        "",
        "## Interpretation boundary",
        "",
        "`simulated_released_gib_hours` is a counterfactual release of the trace's allocated-memory exposure. One representative reload event is generated per within-horizon episode revival. The daily public trace does not identify request multiplicity, physical GPU placement, model-weight bytes, or measured reload latency; bandwidth and concurrency are therefore explicit scenario parameters.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    definitions: list[str],
    horizons: list[int],
    bandwidths: list[float],
    load_slots: list[int],
    arrival_spreads_s: list[float],
    seeds: list[int],
    fixed_overhead_s: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(ROOT / "artifacts/revision_round1/operational_policy_audit.csv")
    methods = list(POLICY_LABELS)
    all_actions: list[pd.DataFrame] = []
    replication_frames: list[pd.DataFrame] = []

    for definition in definitions:
        for horizon in horizons:
            frame = _load_stage3_frame(definition, horizon)
            for method in methods:
                matching = audit[
                    (audit["cold_definition"] == definition)
                    & (audit["horizon"] == horizon)
                    & (audit["method"] == method)
                ]
                policy_row = None if matching.empty else matching.iloc[0]
                actions = _select_actions(frame, method, policy_row)
                route = (
                    "CONTROL_RETAIN"
                    if method == "always_retain"
                    else "ORACLE_BOUND"
                    if method == "oracle"
                    else "ABSTAIN_NO_CALIBRATION_SUPPORT"
                    if policy_row is None or not bool(policy_row["calibration_qualified"])
                    else str(policy_row["test_route"])
                )
                actions["cold_definition"] = definition
                actions["definition_label"] = DEFINITIONS[definition]["label"]
                actions["horizon"] = horizon
                actions["method_label"] = POLICY_LABELS[method]
                actions["qualification_route_full_forward"] = route
                actions["time_block"] = pd.cut(
                    actions["action_day"],
                    bins=[120, 130, 140, 150],
                    labels=["days_121_130", "days_131_140", "days_141_150"],
                ).astype("string")
                if not actions.empty:
                    all_actions.append(actions)

                configs = [
                    SimulationConfig(
                        bandwidth_gib_per_s=bandwidth,
                        load_slots=slots,
                        fixed_overhead_s=fixed_overhead_s,
                        arrival_spread_s=arrival_spread,
                        seed=seed,
                    )
                    for bandwidth in bandwidths
                    for slots in load_slots
                    for arrival_spread in arrival_spreads_s
                    for seed in seeds
                ]
                metadata = {
                    "cold_definition": definition,
                    "definition_label": DEFINITIONS[definition]["label"],
                    "horizon": horizon,
                    "method": method,
                    "method_label": POLICY_LABELS[method],
                    "qualification_route_full_forward": route,
                    "stage3_start_day": int(frame["day"].min()),
                    "stage3_end_day": int(frame["day"].max()),
                }
                replication_frames.append(
                    replay_scenarios(actions, configs, metadata=metadata)
                )

    actions_frame = pd.concat(all_actions, ignore_index=True)
    replications = pd.concat(replication_frames, ignore_index=True)
    summary = _summarize(replications)
    actions_frame.to_parquet(output_dir / "selected_actions.parquet", index=False)
    replications.to_csv(output_dir / "replication_results.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    canonical = summary[
        (summary["bandwidth_gib_per_s"] == 16.0)
        & (summary["load_slots"] == 2)
        & (summary["arrival_spread_s"] == 3_600.0)
    ].copy()
    canonical.to_csv(output_dir / "canonical_summary.csv", index=False)
    design = {
        "schema_version": 1,
        "purpose": "Counterfactual logical-pool loading consequences of frozen cold-model policies",
        "time_unit": "seconds",
        "trace_window_days": [121, 150],
        "definitions": definitions,
        "horizons_days": horizons,
        "bandwidth_gib_per_s": bandwidths,
        "parallel_loading_slots": load_slots,
        "arrival_spread_s": arrival_spreads_s,
        "fixed_overhead_s": fixed_overhead_s,
        "seeds": seeds,
        "common_random_numbers": "the same cold episode and seed produce the same within-day revival offset across policies",
        "replication_unit": "complete seed-specific replay; time blocks are retained as diagnostic blocks, not IID observations",
        "termination": "finite stage-3 action set with a worst-case FIFO drain bound",
        "assumptions": [
            "one representative reload event per within-horizon episode revival",
            "logical deployment memory is an aggregate pool quantity, not a physical single-GPU placement",
            "loading service time equals fixed overhead plus allocated GiB divided by scenario bandwidth",
            "the trace supplies demand timing and H-capped allocated-memory exposure",
        ],
    }
    (output_dir / "design.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8"
    )
    _write_report(summary, output_dir)
    print(output_dir / "scenario_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--definitions", nargs="+", choices=sorted(DEFINITIONS), default=sorted(DEFINITIONS)
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 7])
    parser.add_argument("--bandwidths", nargs="+", type=float, default=[8.0, 16.0, 32.0])
    parser.add_argument("--load-slots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--arrival-spreads-s", nargs="+", type=float, default=[10.0, 300.0, 3_600.0, 86_400.0]
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20261028)
    parser.add_argument("--fixed-overhead-s", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [args.seed_base + index for index in range(args.seeds)]
    run(
        definitions=args.definitions,
        horizons=args.horizons,
        bandwidths=args.bandwidths,
        load_slots=args.load_slots,
        arrival_spreads_s=args.arrival_spreads_s,
        seeds=seeds,
        fixed_overhead_s=args.fixed_overhead_s,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
