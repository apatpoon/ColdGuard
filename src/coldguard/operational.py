from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from .block_ltt import aggregate_time_blocks, daily_loss_tables, select_ltt_threshold
from .config import load_config, require_keys, resolve_project_path
from .generalization import assign_novelty_bucket, deployment_key, normalized_entity
from .multihorizon import attach_horizon_label
from .policy import (
    action_metrics,
    choose_policy,
    first_actions_by_score,
    first_actions_by_ttl,
    score_frontier,
    ttl_frontier,
)
from .survival import predict_cumulative_incidence, predict_next_interval_score


def operational_conditions(
    train_end: int, calibration_end: int, test_end: int, horizon: int
) -> dict[str, str]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return {
        "calibration": f"day > {train_end} AND day <= {calibration_end - horizon}",
        "test": f"day > {calibration_end} AND day <= {test_end - horizon}",
    }


def run_operational(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(
        config,
        "paths",
        [
            "landmarks_all",
            "multihorizon_artifacts_dir",
            "operational_artifacts_dir",
        ],
    )
    require_keys(
        config,
        "multihorizon",
        [
            "horizons",
            "train_end_day",
            "calibration_end_day",
            "test_end_day",
            "target_resource_risk",
            "target_action_risk",
        ],
    )
    require_keys(config, "risk", ["confidence", "threshold_grid_size", "minimum_selected"])
    require_keys(config, "generalization", ["minimum_group_calibration_actions"])
    require_keys(config, "block_ltt", ["block_days", "confidence", "threshold_grid_size"])

    landmarks = resolve_project_path(project_root, config["paths"]["landmarks_all"])
    trained = resolve_project_path(
        project_root, config["paths"]["multihorizon_artifacts_dir"]
    )
    artifacts = resolve_project_path(project_root, config["paths"]["operational_artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(trained / "models" / "hazard_hierarchical.joblib")
    encoder = bundle["encoder"]
    model = bundle["model"]
    interval_bundle = joblib.load(
        trained / "models" / "next_interval_hierarchical.joblib"
    )
    interval_encoder = interval_bundle["encoder"]
    interval_model = interval_bundle["model"]

    cfg = config["multihorizon"]
    horizons = sorted(int(value) for value in cfg["horizons"])
    train_end = int(cfg["train_end_day"])
    calibration_end = int(cfg["calibration_end_day"])
    test_end = int(cfg["test_end_day"])
    action_target = float(cfg["target_action_risk"])
    resource_target = float(cfg["target_resource_risk"])
    risk_cfg = config["risk"]
    confidence = float(risk_cfg["confidence"])
    grid_size = int(risk_cfg["threshold_grid_size"])
    minimum_selected = int(risk_cfg["minimum_selected"])
    group_minimum = int(config["generalization"]["minimum_group_calibration_actions"])

    entities = _load_frame(landmarks, f"day <= {train_end - 1}")
    seen_deployments = set(deployment_key(entities))
    seen_models = set(normalized_entity(entities["model_id"]))
    seen_bases = set(normalized_entity(entities["base_id"]))

    global_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    ltt_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        conditions = operational_conditions(train_end, calibration_end, test_end, horizon)
        calibration = attach_horizon_label(
            _load_frame(landmarks, conditions["calibration"]), horizon
        )
        test = attach_horizon_label(_load_frame(landmarks, conditions["test"]), horizon)
        calibration["score"] = predict_cumulative_incidence(
            calibration, encoder, model, horizon
        )
        test["score"] = predict_cumulative_incidence(test, encoder, model, horizon)
        calibration["score_next_interval"] = predict_next_interval_score(
            calibration, interval_encoder, interval_model, horizon
        )
        test["score_next_interval"] = predict_next_interval_score(
            test, interval_encoder, interval_model, horizon
        )
        calibration["novelty_bucket"] = assign_novelty_bucket(
            calibration, seen_deployments, seen_models, seen_bases
        )
        test["novelty_bucket"] = assign_novelty_bucket(
            test, seen_deployments, seen_models, seen_bases
        )

        frontiers = {
            "ttl": ttl_frontier(calibration, confidence),
            "hazard_hierarchical": score_frontier(
                calibration, "score", grid_size, confidence
            ),
            "next_interval_hierarchical": score_frontier(
                calibration, "score_next_interval", grid_size, confidence
            ),
        }
        selected_actions: pd.DataFrame | None = None
        for method, frontier in frontiers.items():
            choice = choose_policy(
                frontier,
                method,
                resource_target,
                minimum_selected,
                target_action_risk=action_target,
            )
            if choice is None:
                global_rows.append(
                    {
                        "horizon": horizon,
                        "method": method,
                        "eligible": False,
                        "calibration_rows": len(calibration),
                        "test_rows": len(test),
                    }
                )
                continue
            score_column = (
                "score_next_interval"
                if method == "next_interval_hierarchical"
                else "score"
            )
            actions = (
                first_actions_by_ttl(test, int(round(choice.parameter)))
                if method == "ttl"
                else first_actions_by_score(test, score_column, choice.parameter)
            )
            metrics = action_metrics(actions, confidence=confidence, comparisons=1)
            global_rows.append(
                {
                    "horizon": horizon,
                    "method": method,
                    "eligible": True,
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                    **asdict(choice),
                    **{f"test_{key}": value for key, value in metrics.items()},
                    "test_dual_contract_pass": (
                        float(metrics["unweighted_risk_ucb"]) <= action_target
                        and float(metrics["risk_ucb"]) <= resource_target
                    ),
                }
            )
            if method == "hazard_hierarchical":
                selected_actions = actions

        buckets = sorted(calibration["novelty_bucket"].unique())
        group_confidence = 1.0 - (1.0 - confidence) / max(1, len(buckets))
        combined = []
        for bucket in buckets:
            cal_group = calibration.loc[calibration["novelty_bucket"] == bucket]
            test_group = test.loc[test["novelty_bucket"] == bucket]
            frontier = score_frontier(cal_group, "score", grid_size, group_confidence)
            choice = choose_policy(
                frontier,
                "hazard_hierarchical",
                resource_target,
                group_minimum,
                target_action_risk=action_target,
            )
            if choice is None:
                guard_rows.append(
                    {
                        "horizon": horizon,
                        "novelty_bucket": bucket,
                        "eligible": False,
                    }
                )
                continue
            actions = first_actions_by_score(test_group, "score", choice.parameter)
            combined.append(actions)
            metrics = action_metrics(actions, confidence=group_confidence, comparisons=1)
            guard_rows.append(
                {
                    "horizon": horizon,
                    "novelty_bucket": bucket,
                    "eligible": True,
                    **asdict(choice),
                    **{f"test_{key}": value for key, value in metrics.items()},
                    "test_dual_contract_pass": (
                        float(metrics["unweighted_risk_ucb"]) <= action_target
                        and float(metrics["risk_ucb"]) <= resource_target
                    ),
                }
            )
        if combined:
            combined_actions = pd.concat(combined, ignore_index=True)
            metrics = action_metrics(combined_actions, confidence=confidence, comparisons=1)
            guard_rows.append(
                {
                    "horizon": horizon,
                    "novelty_bucket": "ALL_ELIGIBLE_BUCKETS",
                    "eligible": True,
                    **{f"test_{key}": value for key, value in metrics.items()},
                    "test_dual_contract_pass": (
                        float(metrics["unweighted_risk_ucb"]) <= action_target
                        and float(metrics["risk_ucb"]) <= resource_target
                    ),
                }
            )

        thresholds = np.linspace(
            0.0, 1.0, int(config["block_ltt"]["threshold_grid_size"])
        )
        _, daily_action, daily_resource, counts = daily_loss_tables(
            calibration, "score", thresholds
        )
        delta = 1.0 - float(config["block_ltt"]["confidence"])
        for block_days in [int(value) for value in config["block_ltt"]["block_days"]]:
            action_blocks = aggregate_time_blocks(daily_action, block_days)
            resource_blocks = aggregate_time_blocks(daily_resource, block_days)
            for mode in ["fixed_sequence", "bonferroni"]:
                choice = select_ltt_threshold(
                    action_blocks,
                    resource_blocks,
                    thresholds,
                    counts,
                    action_target,
                    resource_target,
                    delta,
                    minimum_selected,
                    mode,
                )
                row = {
                    "horizon": horizon,
                    "block_days": block_days,
                    "mode": mode,
                    "blocks": len(action_blocks),
                    "eligible": choice is not None,
                    **({} if choice is None else asdict(choice)),
                }
                if choice is not None:
                    actions = first_actions_by_score(test, "score", choice.threshold)
                    metrics = action_metrics(actions, confidence=confidence, comparisons=1)
                    _, test_daily_action, test_daily_resource, _ = daily_loss_tables(
                        test, "score", np.array([choice.threshold])
                    )
                    test_action_blocks = aggregate_time_blocks(
                        test_daily_action, block_days
                    )
                    test_resource_blocks = aggregate_time_blocks(
                        test_daily_resource, block_days
                    )
                    row.update(
                        {
                            "test_action_block_risk": float(test_action_blocks.mean()),
                            "test_resource_block_risk": float(test_resource_blocks.mean()),
                            "test_block_contract_pass": (
                                float(test_action_blocks.mean()) <= action_target
                                and float(test_resource_blocks.mean()) <= resource_target
                            ),
                            **{f"test_{key}": value for key, value in metrics.items()},
                        }
                    )
                ltt_rows.append(row)

        if selected_actions is not None:
            for group_column in ["site_id", "gpu_type", "pd_role"]:
                groups = [
                    (value, group)
                    for value, group in selected_actions.groupby(group_column, dropna=False)
                    if len(group) >= 30
                ]
                comparisons = max(1, len(groups))
                for value, group in groups:
                    metrics = action_metrics(
                        group, confidence=confidence, comparisons=comparisons
                    )
                    worst_rows.append(
                        {
                            "horizon": horizon,
                            "group_column": group_column,
                            "group_value": "__NULL__" if pd.isna(value) else str(value),
                            **metrics,
                        }
                    )

    global_results = pd.DataFrame(global_rows)
    guard_results = pd.DataFrame(guard_rows)
    ltt_results = pd.DataFrame(ltt_rows)
    worst_results = pd.DataFrame(worst_rows)
    global_results.to_csv(artifacts / "global_policies.csv", index=False)
    guard_results.to_csv(artifacts / "novelty_guard.csv", index=False)
    ltt_results.to_csv(artifacts / "block_ltt.csv", index=False)
    worst_results.to_csv(artifacts / "worst_groups.csv", index=False)
    report = _render_report(
        global_results,
        guard_results,
        ltt_results,
        worst_results,
        action_target=action_target,
        resource_target=resource_target,
    )
    report_path = artifacts / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _load_frame(path: Path, condition: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT * FROM read_parquet('{_sql_string(path)}') WHERE {condition}"
        ).fetchdf()
    finally:
        con.close()


def _render_report(
    global_results: pd.DataFrame,
    guard_results: pd.DataFrame,
    ltt_results: pd.DataFrame,
    worst_results: pd.DataFrame,
    action_target: float,
    resource_target: float,
) -> str:
    lines = [
        "# ColdGuard Horizon-Specific Operational Evaluation",
        "",
        "## Global policies",
        "",
        "| Horizon | Method | Calibration days/rows | Test days/rows | Eligible | Test contract | Action risk UCB | Resource risk UCB | Reclaimed GiB-hours |",
        "|---:|---|---|---|---|---|---:|---:|---:|",
    ]
    for _, row in global_results.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {int(row['horizon'])} | {row['method']} | {int(row['calibration_rows']):,} | "
                f"{int(row['test_rows']):,} | no | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {int(row['horizon'])} | {row['method']} | {int(row['calibration_rows']):,} | "
                f"{int(row['test_rows']):,} | yes | "
                f"{'PASS' if bool(row['test_dual_contract_pass']) else 'FAIL'} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} | "
                f"{row['test_reclaimed_gib_hours']:,.0f} |"
            )
    lines.extend(
        [
            "",
            "## Novelty guard",
            "",
            "| Horizon | Bucket | Eligible | Test actions | Test contract | Action risk UCB | Resource risk UCB |",
            "|---:|---|---|---:|---|---:|---:|",
        ]
    )
    for _, row in guard_results.iterrows():
        if not bool(row["eligible"]):
            lines.append(f"| {int(row['horizon'])} | {row['novelty_bucket']} | no | 0 | — | — | — |")
        else:
            lines.append(
                f"| {int(row['horizon'])} | {row['novelty_bucket']} | yes | "
                f"{int(row['test_actions']):,} | "
                f"{'PASS' if bool(row['test_dual_contract_pass']) else 'FAIL'} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Block LTT availability",
            "",
            "| Horizon | Block days | Mode | Blocks | Eligible | Test block contract | Test conditional contract | Test actions | Test action block risk | Test resource block risk | Test action UCB | Test resource UCB |",
            "|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in ltt_results.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {int(row['horizon'])} | {int(row['block_days'])} | {row['mode']} | "
                f"{int(row['blocks'])} | no | — | — | 0 | — | — | — | — |"
            )
        else:
            conditional_pass = (
                row["test_unweighted_risk_ucb"] <= action_target
                and row["test_risk_ucb"] <= resource_target
            )
            lines.append(
                f"| {int(row['horizon'])} | {int(row['block_days'])} | {row['mode']} | "
                f"{int(row['blocks'])} | yes | "
                f"{'PASS' if bool(row['test_block_contract_pass']) else 'FAIL'} | "
                f"{'PASS' if conditional_pass else 'FAIL'} | {int(row['test_actions'])} | "
                f"{row['test_action_block_risk']:.2%} | {row['test_resource_block_risk']:.2%} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Worst qualifying groups",
            "",
            "| Horizon | Field | Group | Actions | Action risk UCB | Resource risk UCB |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    if worst_results.empty:
        lines.append("| — | — | — | 0 | — | — |")
    else:
        for _, row in worst_results.sort_values(
            ["unweighted_risk_ucb", "risk_ucb"], ascending=False
        ).head(20).iterrows():
            lines.append(
                f"| {int(row['horizon'])} | {row['group_column']} | {row['group_value']} | "
                f"{int(row['actions']):,} | {row['unweighted_risk_ucb']:.2%} | "
                f"{row['risk_ucb']:.2%} |"
            )
    return "\n".join(lines) + "\n"


def _sql_string(path: Path) -> str:
    return str(path).replace("'", "''")

