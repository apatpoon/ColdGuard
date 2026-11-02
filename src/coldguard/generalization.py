from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import load_config, require_keys, resolve_project_path
from .policy import action_metrics, choose_policy, first_actions_by_score, score_frontier


DEPLOYMENT_KEYS = ["app_id", "model_id", "site_id", "pd_role"]


def normalized_entity(series: pd.Series) -> pd.Series:
    return series.fillna("__NULL__").astype(str)


def deployment_key(frame: pd.DataFrame) -> pd.Series:
    return frame[DEPLOYMENT_KEYS].fillna("__NULL__").astype(str).agg("|".join, axis=1)


def assign_novelty_bucket(
    frame: pd.DataFrame,
    seen_deployments: set[str],
    seen_models: set[str],
    seen_bases: set[str],
) -> pd.Series:
    deployments = deployment_key(frame)
    models = normalized_entity(frame["model_id"])
    bases = normalized_entity(frame["base_id"])
    bucket = np.full(len(frame), "unseen_base", dtype=object)
    bucket[bases.isin(seen_bases).to_numpy()] = "unseen_model_seen_base"
    bucket[models.isin(seen_models).to_numpy()] = "unseen_deployment_seen_model"
    bucket[deployments.isin(seen_deployments).to_numpy()] = "seen_deployment"
    return pd.Series(bucket, index=frame.index, name="novelty_bucket")


def safe_predictive_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    result = {
        "rows": int(len(labels)),
        "prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "brier": float(brier_score_loss(labels, scores)) if len(labels) else float("nan"),
        "average_precision": (
            float(average_precision_score(labels, scores)) if labels.sum() > 0 else float("nan")
        ),
        "roc_auc": (
            float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
        ),
    }
    return result


def run_generalization(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(
        config,
        "paths",
        [
            "landmarks_all",
            "multihorizon_artifacts_dir",
            "generalization_artifacts_dir",
        ],
    )
    require_keys(config, "multihorizon", ["horizons", "train_end_day"])
    require_keys(config, "risk", ["confidence"])
    require_keys(config, "generalization", ["minimum_group_calibration_actions"])

    landmarks = resolve_project_path(project_root, config["paths"]["landmarks_all"])
    multihorizon = resolve_project_path(
        project_root, config["paths"]["multihorizon_artifacts_dir"]
    )
    artifacts = resolve_project_path(
        project_root, config["paths"]["generalization_artifacts_dir"]
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(multihorizon / "test_predictions.parquet")
    calibration = pd.read_parquet(multihorizon / "calibration_predictions.parquet")
    policies = pd.read_csv(multihorizon / "dual_risk_policies.csv")

    train_end = int(config["multihorizon"]["train_end_day"])
    train_entities = _load_entities(landmarks, train_end - 1)
    seen_deployments = set(deployment_key(train_entities))
    seen_models = set(normalized_entity(train_entities["model_id"]))
    seen_bases = set(normalized_entity(train_entities["base_id"]))
    test["novelty_bucket"] = assign_novelty_bucket(
        test, seen_deployments, seen_models, seen_bases
    )
    calibration["novelty_bucket"] = assign_novelty_bucket(
        calibration, seen_deployments, seen_models, seen_bases
    )

    horizons = sorted(int(value) for value in config["multihorizon"]["horizons"])
    methods = sorted(
        {
            column.removeprefix("score_").rsplit("_", 1)[0]
            for column in test.columns
            if column.startswith("score_")
        }
    )
    predictive_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        labels = test[f"label_{horizon}d"].to_numpy(int)
        for method in methods:
            scores = test[f"score_{method}_{horizon}d"].to_numpy(float)
            for bucket, indices in test.groupby("novelty_bucket").groups.items():
                index = np.asarray(list(indices), dtype=int)
                predictive_rows.append(
                    {
                        "horizon": horizon,
                        "method": method,
                        "novelty_bucket": bucket,
                        **safe_predictive_metrics(labels[index], scores[index]),
                    }
                )

    confidence = float(config["risk"]["confidence"])
    policy_rows: list[dict[str, Any]] = []
    worst_group_rows: list[dict[str, Any]] = []
    novelty_guard_rows: list[dict[str, Any]] = []
    eligible = policies.loc[policies["eligible"] == True].copy()  # noqa: E712
    for _, policy in eligible.iterrows():
        horizon = int(policy["horizon"])
        method = str(policy["method"])
        if method == "ttl":
            continue
        frame = test.copy()
        frame["reactivates_within_horizon"] = frame[f"label_{horizon}d"]
        score_column = f"score_{method}_{horizon}d"
        actions = first_actions_by_score(frame, score_column, float(policy["parameter"]))
        for bucket, group in actions.groupby("novelty_bucket"):
            metrics = action_metrics(group, confidence=confidence, comparisons=4)
            policy_rows.append(
                {
                    "horizon": horizon,
                    "method": method,
                    "novelty_bucket": bucket,
                    **metrics,
                }
            )

        if method != "hazard_hierarchical":
            continue
        for group_column in ["site_id", "gpu_type", "pd_role"]:
            qualifying = [
                (value, group)
                for value, group in actions.groupby(group_column, dropna=False)
                if len(group) >= 30
            ]
            comparisons = max(1, len(qualifying))
            for value, group in qualifying:
                metrics = action_metrics(group, confidence=confidence, comparisons=comparisons)
                worst_group_rows.append(
                    {
                        "horizon": horizon,
                        "method": method,
                        "group_column": group_column,
                        "group_value": "__NULL__" if pd.isna(value) else str(value),
                        **metrics,
                    }
                )

    target_resource = float(config["multihorizon"]["target_resource_risk"])
    target_action = float(config["multihorizon"]["target_action_risk"])
    minimum_group_actions = int(
        config["generalization"]["minimum_group_calibration_actions"]
    )
    buckets = sorted(calibration["novelty_bucket"].unique())
    group_confidence = 1.0 - (1.0 - confidence) / max(1, len(buckets))
    grid_size = int(config["risk"].get("threshold_grid_size", 80))
    for horizon in horizons:
        method = "hazard_hierarchical"
        score_column = f"score_{method}_{horizon}d"
        combined_actions = []
        for bucket in buckets:
            cal_group = calibration.loc[calibration["novelty_bucket"] == bucket].copy()
            test_group = test.loc[test["novelty_bucket"] == bucket].copy()
            cal_group["reactivates_within_horizon"] = cal_group[f"label_{horizon}d"]
            test_group["reactivates_within_horizon"] = test_group[f"label_{horizon}d"]
            frontier = score_frontier(
                cal_group, score_column, grid_size, group_confidence
            )
            choice = choose_policy(
                frontier,
                method,
                target_resource,
                minimum_group_actions,
                target_action_risk=target_action,
            )
            if choice is None:
                novelty_guard_rows.append(
                    {
                        "horizon": horizon,
                        "novelty_bucket": bucket,
                        "calibration_eligible": False,
                    }
                )
                continue
            actions = first_actions_by_score(test_group, score_column, choice.parameter)
            combined_actions.append(actions)
            metrics = action_metrics(
                actions, confidence=group_confidence, comparisons=1
            )
            novelty_guard_rows.append(
                {
                    "horizon": horizon,
                    "novelty_bucket": bucket,
                    "calibration_eligible": True,
                    "parameter": choice.parameter,
                    "calibration_actions": choice.calibration_actions,
                    "calibration_action_risk_ucb": choice.calibration_unweighted_risk_ucb,
                    "calibration_resource_risk_ucb": choice.calibration_risk_ucb,
                    **{f"test_{key}": value for key, value in metrics.items()},
                    "test_dual_contract_pass": (
                        float(metrics["unweighted_risk_ucb"]) <= target_action
                        and float(metrics["risk_ucb"]) <= target_resource
                    ),
                }
            )
        if combined_actions:
            combined = pd.concat(combined_actions, ignore_index=True)
            metrics = action_metrics(combined, confidence=confidence, comparisons=1)
            novelty_guard_rows.append(
                {
                    "horizon": horizon,
                    "novelty_bucket": "ALL_ELIGIBLE_BUCKETS",
                    "calibration_eligible": True,
                    **{f"test_{key}": value for key, value in metrics.items()},
                    "test_dual_contract_pass": (
                        float(metrics["unweighted_risk_ucb"]) <= target_action
                        and float(metrics["risk_ucb"]) <= target_resource
                    ),
                }
            )

    predictive = pd.DataFrame(predictive_rows)
    policy_audit = pd.DataFrame(policy_rows)
    worst_groups = pd.DataFrame(worst_group_rows)
    novelty_guard = pd.DataFrame(novelty_guard_rows)
    predictive.to_csv(artifacts / "novelty_predictive_metrics.csv", index=False)
    policy_audit.to_csv(artifacts / "novelty_policy_metrics.csv", index=False)
    worst_groups.to_csv(artifacts / "worst_group_metrics.csv", index=False)
    novelty_guard.to_csv(artifacts / "novelty_guard_policy.csv", index=False)
    distribution = (
        test.groupby("novelty_bucket")
        .agg(rows=("day", "size"), episodes=("episode_id", "nunique"))
        .reset_index()
    )
    distribution.to_csv(artifacts / "novelty_distribution.csv", index=False)
    report = _render_report(
        distribution, predictive, policy_audit, novelty_guard, worst_groups
    )
    report_path = artifacts / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _load_entities(path: Path, end_day: int) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(
            f"""
            SELECT DISTINCT app_id, model_id, base_id, site_id, pd_role
            FROM read_parquet('{_sql_string(path)}')
            WHERE day <= {end_day}
            """
        ).fetchdf()
    finally:
        con.close()


def _render_report(
    distribution: pd.DataFrame,
    predictive: pd.DataFrame,
    policies: pd.DataFrame,
    novelty_guard: pd.DataFrame,
    worst_groups: pd.DataFrame,
) -> str:
    lines = [
        "# ColdGuard Novelty and Worst-Group Audit",
        "",
        "## Test novelty distribution",
        "",
        "| Bucket | Rows | Episodes |",
        "|---|---:|---:|",
    ]
    for _, row in distribution.iterrows():
        lines.append(
            f"| {row['novelty_bucket']} | {int(row['rows']):,} | {int(row['episodes']):,} |"
        )
    lines.extend(
        [
            "",
            "## Hazard-hierarchical prediction by novelty",
            "",
            "| Horizon | Bucket | Rows | Prevalence | Brier | AP | ROC-AUC |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    view = predictive.loc[predictive["method"] == "hazard_hierarchical"]
    for _, row in view.iterrows():
        lines.append(
            f"| {int(row['horizon'])} | {row['novelty_bucket']} | {int(row['rows']):,} | "
            f"{row['prevalence']:.2%} | {row['brier']:.4f} | {row['average_precision']:.4f} | "
            f"{row['roc_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected policies by novelty",
            "",
            "| Horizon | Method | Bucket | Actions | Action risk (UCB) | Resource risk (UCB) | Reclaimed GiB-hours |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in policies.iterrows():
        lines.append(
            f"| {int(row['horizon'])} | {row['method']} | {row['novelty_bucket']} | "
            f"{int(row['actions']):,} | {row['unweighted_risk']:.2%} ({row['unweighted_risk_ucb']:.2%}) | "
            f"{row['weighted_risk']:.2%} ({row['risk_ucb']:.2%}) | {row['reclaimed_gib_hours']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Calibration-selected novelty guard (hazard hierarchical)",
            "",
            "Each novelty bucket receives its own calibration threshold. Confidence is Bonferroni-adjusted across buckets; ineligible buckets abstain.",
            "",
            "| Horizon | Bucket | Calibration eligible | Test actions | Test contract | Action risk UCB | Resource risk UCB | Reclaimed GiB-hours |",
            "|---:|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for _, row in novelty_guard.iterrows():
        if not bool(row["calibration_eligible"]):
            lines.append(
                f"| {int(row['horizon'])} | {row['novelty_bucket']} | no | 0 | — | — | — | 0 |"
            )
        else:
            lines.append(
                f"| {int(row['horizon'])} | {row['novelty_bucket']} | yes | "
                f"{int(row['test_actions']):,} | "
                f"{'PASS' if bool(row['test_dual_contract_pass']) else 'FAIL'} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} | "
                f"{row['test_reclaimed_gib_hours']:,.0f} |"
            )
    lines.extend(
        [
            "",
            "## Worst qualifying hazard-hierarchical groups",
            "",
            "Only groups with at least 30 actions are audited. UCBs are multiplicity-adjusted within each grouping field.",
            "",
            "| Horizon | Field | Group | Actions | Action risk UCB | Resource risk UCB |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    if worst_groups.empty:
        lines.append("| — | — | — | 0 | — | — |")
    else:
        ranked = worst_groups.sort_values(
            ["unweighted_risk_ucb", "risk_ucb"], ascending=False
        ).head(20)
        for _, row in ranked.iterrows():
            lines.append(
                f"| {int(row['horizon'])} | {row['group_column']} | {row['group_value']} | "
                f"{int(row['actions']):,} | {row['unweighted_risk_ucb']:.2%} | {row['risk_ucb']:.2%} |"
            )
    return "\n".join(lines) + "\n"


def _sql_string(path: Path) -> str:
    return str(path).replace("'", "''")

