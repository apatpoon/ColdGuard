from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from .block_ltt import daily_loss_tables, select_ltt_threshold
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


@dataclass(frozen=True)
class OperationalFold:
    name: str
    calibration_start_day: int
    calibration_end_day: int
    test_start_day: int
    test_end_day: int

    def conditions(self, horizon: int) -> dict[str, str]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not (
            self.calibration_start_day <= self.calibration_end_day
            < self.test_start_day <= self.test_end_day
        ):
            raise ValueError(f"Invalid fold: {self}")
        return {
            "calibration": (
                f"day >= {self.calibration_start_day} "
                f"AND day <= {self.calibration_end_day - horizon}"
            ),
            "test": (
                f"day >= {self.test_start_day} "
                f"AND day <= {self.test_end_day - horizon}"
            ),
        }


def fallback_route(
    global_metrics: dict[str, float | int],
    group_choice_exists: bool,
    minimum_actions: int,
    action_target: float,
    resource_target: float,
) -> str:
    if (
        int(global_metrics["actions"]) >= minimum_actions
        and float(global_metrics["unweighted_risk_ucb"]) <= action_target
        and float(global_metrics["risk_ucb"]) <= resource_target
    ):
        return "global_validated"
    if group_choice_exists:
        return "group_specific"
    return "abstain"


def run_robustness(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(
        config,
        "paths",
        ["landmarks_all", "multihorizon_artifacts_dir", "robustness_artifacts_dir"],
    )
    require_keys(config, "robustness", ["horizons", "folds"])
    require_keys(
        config,
        "multihorizon",
        ["train_end_day", "target_resource_risk", "target_action_risk"],
    )
    require_keys(config, "risk", ["confidence", "threshold_grid_size", "minimum_selected"])
    require_keys(config, "generalization", ["minimum_group_calibration_actions"])
    require_keys(config, "block_ltt", ["confidence", "threshold_grid_size"])

    landmarks = resolve_project_path(project_root, config["paths"]["landmarks_all"])
    model_dir = resolve_project_path(
        project_root, config["paths"]["multihorizon_artifacts_dir"]
    ) / "models"
    artifacts = resolve_project_path(project_root, config["paths"]["robustness_artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(model_dir / "hazard_hierarchical.joblib")
    encoder, model = bundle["encoder"], bundle["model"]
    interval_bundle = joblib.load(model_dir / "next_interval_hierarchical.joblib")
    interval_encoder = interval_bundle["encoder"]
    interval_model = interval_bundle["model"]

    folds = [OperationalFold(**values) for values in config["robustness"]["folds"]]
    horizons = sorted(set(int(value) for value in config["robustness"]["horizons"]))
    train_end = int(config["multihorizon"]["train_end_day"])
    action_target = float(config["multihorizon"]["target_action_risk"])
    resource_target = float(config["multihorizon"]["target_resource_risk"])
    confidence = float(config["risk"]["confidence"])
    grid_size = int(config["risk"]["threshold_grid_size"])
    minimum_selected = int(config["risk"]["minimum_selected"])
    group_minimum = int(config["generalization"]["minimum_group_calibration_actions"])

    entities = _load_frame(landmarks, f"day <= {train_end - 1}")
    seen_deployments = set(deployment_key(entities))
    seen_models = set(normalized_entity(entities["model_id"]))
    seen_bases = set(normalized_entity(entities["base_id"]))

    global_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    ltt_rows: list[dict[str, Any]] = []
    for fold in folds:
        for horizon in horizons:
            conditions = fold.conditions(horizon)
            calibration = attach_horizon_label(
                _load_frame(landmarks, conditions["calibration"]), horizon
            )
            test = attach_horizon_label(_load_frame(landmarks, conditions["test"]), horizon)
            if calibration.empty or test.empty:
                raise ValueError(f"Empty frame for {fold.name}, horizon {horizon}")
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
            for frame in [calibration, test]:
                frame["novelty_bucket"] = assign_novelty_bucket(
                    frame, seen_deployments, seen_models, seen_bases
                )

            global_choice = None
            for method, frontier in {
                "ttl": ttl_frontier(calibration, confidence),
                "hazard_hierarchical": score_frontier(
                    calibration, "score", grid_size, confidence
                ),
                "next_interval_hierarchical": score_frontier(
                    calibration, "score_next_interval", grid_size, confidence
                ),
            }.items():
                choice = choose_policy(
                    frontier,
                    method,
                    resource_target,
                    minimum_selected,
                    target_action_risk=action_target,
                )
                if method == "hazard_hierarchical":
                    global_choice = choice
                row: dict[str, Any] = {
                    "fold": fold.name,
                    "horizon": horizon,
                    "method": method,
                    "eligible": choice is not None,
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                }
                if choice is not None:
                    score_column = (
                        "score_next_interval"
                        if method == "next_interval_hierarchical"
                        else "score"
                    )
                    actions = (
                        first_actions_by_ttl(test, int(round(choice.parameter)))
                        if method == "ttl"
                        else first_actions_by_score(
                            test, score_column, choice.parameter
                        )
                    )
                    metrics = action_metrics(actions, confidence, comparisons=1)
                    row.update(asdict(choice))
                    row.update({f"test_{key}": value for key, value in metrics.items()})
                    row["test_contract_pass"] = _passes(
                        metrics, action_target, resource_target
                    )
                global_rows.append(row)

            buckets = sorted(
                set(calibration["novelty_bucket"].unique())
                | set(test["novelty_bucket"].unique())
            )
            group_confidence = 1.0 - (1.0 - confidence) / max(1, len(buckets))
            combined_actions: list[pd.DataFrame] = []
            for bucket in buckets:
                cal_group = calibration[calibration["novelty_bucket"] == bucket]
                test_group = test[test["novelty_bucket"] == bucket]
                global_actions = (
                    first_actions_by_score(cal_group, "score", global_choice.parameter)
                    if global_choice is not None
                    else cal_group.iloc[0:0].copy()
                )
                global_metrics = action_metrics(
                    global_actions, group_confidence, comparisons=1
                )
                group_frontier = score_frontier(
                    cal_group, "score", grid_size, group_confidence
                )
                group_choice = choose_policy(
                    group_frontier,
                    "hazard_hierarchical",
                    resource_target,
                    group_minimum,
                    target_action_risk=action_target,
                )
                route = fallback_route(
                    global_metrics,
                    group_choice is not None,
                    group_minimum,
                    action_target,
                    resource_target,
                )
                parameter = (
                    global_choice.parameter
                    if route == "global_validated" and global_choice is not None
                    else group_choice.parameter
                    if route == "group_specific" and group_choice is not None
                    else None
                )
                row = {
                    "fold": fold.name,
                    "horizon": horizon,
                    "novelty_bucket": bucket,
                    "route": route,
                    "parameter": parameter,
                    "calibration_rows": len(cal_group),
                    **{f"global_validation_{key}": value for key, value in global_metrics.items()},
                }
                if parameter is not None:
                    actions = first_actions_by_score(test_group, "score", float(parameter))
                    combined_actions.append(actions)
                    metrics = action_metrics(actions, group_confidence, comparisons=1)
                    row.update({f"test_{key}": value for key, value in metrics.items()})
                    row["test_contract_pass"] = _passes(
                        metrics, action_target, resource_target
                    )
                fallback_rows.append(row)
            if combined_actions:
                combined = pd.concat(combined_actions, ignore_index=True)
                combined.sort_values(["episode_id", "day"], inplace=True)
                combined = combined.drop_duplicates("episode_id", keep="first")
                metrics = action_metrics(combined, confidence, comparisons=1)
                fallback_rows.append(
                    {
                        "fold": fold.name,
                        "horizon": horizon,
                        "novelty_bucket": "ALL_ROUTED_BUCKETS",
                        "route": "combined",
                        **{f"test_{key}": value for key, value in metrics.items()},
                        "test_contract_pass": _passes(
                            metrics, action_target, resource_target
                        ),
                    }
                )

            thresholds = np.linspace(
                0.0, 1.0, int(config["block_ltt"]["threshold_grid_size"])
            )
            _, daily_action, daily_resource, counts = daily_loss_tables(
                calibration, "score", thresholds
            )
            choice = select_ltt_threshold(
                daily_action,
                daily_resource,
                thresholds,
                counts,
                action_target,
                resource_target,
                1.0 - float(config["block_ltt"]["confidence"]),
                minimum_selected,
                "fixed_sequence",
            )
            row = {
                "fold": fold.name,
                "horizon": horizon,
                "eligible": choice is not None,
                "calibration_days": len(daily_action),
            }
            if choice is not None:
                actions = first_actions_by_score(test, "score", choice.threshold)
                metrics = action_metrics(actions, confidence, comparisons=1)
                _, test_action, test_resource, _ = daily_loss_tables(
                    test, "score", np.array([choice.threshold])
                )
                row.update(asdict(choice))
                row.update({f"test_{key}": value for key, value in metrics.items()})
                row.update(
                    {
                        "test_daily_action_risk": float(test_action.mean()),
                        "test_daily_resource_risk": float(test_resource.mean()),
                        "test_conditional_contract_pass": _passes(
                            metrics, action_target, resource_target
                        ),
                        "test_daily_contract_pass": (
                            float(test_action.mean()) <= action_target
                            and float(test_resource.mean()) <= resource_target
                        ),
                    }
                )
            ltt_rows.append(row)

    global_results = pd.DataFrame(global_rows)
    fallback_results = pd.DataFrame(fallback_rows)
    ltt_results = pd.DataFrame(ltt_rows)
    global_results.to_csv(artifacts / "rolling_global.csv", index=False)
    fallback_results.to_csv(artifacts / "rolling_fallback.csv", index=False)
    ltt_results.to_csv(artifacts / "rolling_ltt.csv", index=False)
    report = _render_report(global_results, fallback_results, ltt_results)
    report_path = artifacts / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _passes(
    metrics: dict[str, float | int], action_target: float, resource_target: float
) -> bool:
    return (
        float(metrics["unweighted_risk_ucb"]) <= action_target
        and float(metrics["risk_ucb"]) <= resource_target
    )


def _load_frame(path: Path, condition: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        escaped = str(path).replace("'", "''")
        return con.execute(
            f"SELECT * FROM read_parquet('{escaped}') WHERE {condition}"
        ).fetchdf()
    finally:
        con.close()


def _render_report(
    global_results: pd.DataFrame,
    fallback_results: pd.DataFrame,
    ltt_results: pd.DataFrame,
) -> str:
    lines = [
        "# ColdGuard Frozen-Model Rolling Robustness",
        "",
        "The hazard model is frozen after day 60. Only policy thresholds are recalibrated in each chronological window.",
        "",
        "## Global policy replay",
        "",
        "| Fold | Horizon | Method | Eligible | Test actions | Test contract | Action UCB | Resource UCB | Reclaimed GiB-hours |",
        "|---|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for _, row in global_results.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | {row['method']} | no | 0 | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | {row['method']} | yes | "
                f"{int(row['test_actions']):,} | {'PASS' if bool(row['test_contract_pass']) else 'FAIL'} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} | "
                f"{row['test_reclaimed_gib_hours']:,.0f} |"
            )
    lines.extend([
        "", "## Three-level novelty fallback", "",
        "| Fold | Horizon | Bucket | Route | Test actions | Test contract | Action UCB | Resource UCB |",
        "|---|---:|---|---|---:|---|---:|---:|",
    ])
    for _, row in fallback_results.iterrows():
        if row["route"] == "abstain":
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | {row['novelty_bucket']} | abstain | 0 | — | — | — |"
            )
        else:
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | {row['novelty_bucket']} | {row['route']} | "
                f"{int(row['test_actions']):,} | {'PASS' if bool(row['test_contract_pass']) else 'FAIL'} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} |"
            )
    lines.extend([
        "", "## Daily fixed-sequence LTT", "",
        "| Fold | Horizon | Eligible | Test actions | Daily contract | Conditional contract | Action UCB | Resource UCB |",
        "|---|---:|---|---:|---|---|---:|---:|",
    ])
    for _, row in ltt_results.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | no | 0 | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | yes | {int(row['test_actions']):,} | "
                f"{'PASS' if bool(row['test_daily_contract_pass']) else 'FAIL'} | "
                f"{'PASS' if bool(row['test_conditional_contract_pass']) else 'FAIL'} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} |"
            )
    return "\n".join(lines) + "\n"
