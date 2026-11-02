from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import load_config, require_keys, resolve_project_path
from .policy import (
    PolicyChoice,
    action_metrics,
    choose_policy,
    first_actions_by_score,
    first_actions_by_ttl,
    score_frontier,
    ttl_frontier,
)


NUMERIC_FEATURES = [
    "cold_age",
    "prior_cold_episodes",
    "days_since_last_active",
    "prior_active_days_30",
    "params_b",
    "instance_count",
    "cluster_count",
    "gpu_type_count",
    "gpu_hours",
    "observed_hours",
    "sm_util",
    "zero_fraction",
    "mem_gib_hours",
    "avg_mem_gib_calendar",
    "avg_gpu_count",
]

SIZE_ONLY_FEATURES = [
    "params_b",
    "instance_count",
    "gpu_hours",
    "observed_hours",
    "mem_gib_hours",
    "avg_mem_gib_calendar",
    "avg_gpu_count",
]

CATEGORICAL_FEATURES = [
    "app_id",
    "model_id",
    "base_id",
    "site_id",
    "pd_role",
    "cluster_id",
    "gpu_type",
    "model_arch",
    "base_arch",
    "variant_arch",
]


class HierarchicalTabularEncoder:
    def __init__(self, numeric: list[str], categorical: list[str], strength: float = 50.0):
        self.numeric = numeric
        self.categorical = categorical
        self.strength = float(strength)
        self.medians: dict[str, float] = {}
        self.frequency_maps: dict[str, dict[str, float]] = {}
        self.target_maps: dict[str, dict[str, float]] = {}
        self.global_target = 0.0
        self.output_names: list[str] = []

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "HierarchicalTabularEncoder":
        target_series = pd.Series(np.asarray(target, dtype=float), index=frame.index)
        self.global_target = float(target_series.mean())
        total = max(1, len(frame))
        for column in self.numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            self.medians[column] = median
        for column in self.categorical:
            values = frame[column].fillna("__NULL__").astype(str)
            counts = values.value_counts(dropna=False)
            sums = target_series.groupby(values).sum()
            self.frequency_maps[column] = (counts / total).to_dict()
            smoothed = (sums + self.strength * self.global_target) / (counts + self.strength)
            self.target_maps[column] = smoothed.to_dict()
        self.output_names = list(self.numeric)
        for column in self.categorical:
            self.output_names.extend([f"{column}__freq", f"{column}__target"])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for column in self.numeric:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column]).to_numpy(float)
            if column in {
                "cold_age",
                "prior_cold_episodes",
                "days_since_last_active",
                "params_b",
                "instance_count",
                "gpu_hours",
                "observed_hours",
                "mem_gib_hours",
                "avg_mem_gib_calendar",
                "avg_gpu_count",
            }:
                values = np.log1p(np.clip(values, 0.0, None))
            columns.append(values)
        for column in self.categorical:
            values = frame[column].fillna("__NULL__").astype(str)
            columns.append(values.map(self.frequency_maps[column]).fillna(0.0).to_numpy(float))
            columns.append(
                values.map(self.target_maps[column]).fillna(self.global_target).to_numpy(float)
            )
        return np.column_stack(columns).astype(np.float32, copy=False)

    def fit_transform(self, frame: pd.DataFrame, target: np.ndarray) -> np.ndarray:
        return self.fit(frame, target).transform(frame)

    def fit_transform_group_oof(
        self,
        frame: pd.DataFrame,
        target: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
    ) -> np.ndarray:
        """Create leakage-resistant encodings, then retain full maps for inference.

        All rows from a logical deployment stay in one fold. This prevents a
        category's outcome from being encoded using another landmark from the
        same deployment during model fitting.
        """
        unique_groups = np.unique(groups)
        splits = min(int(n_splits), int(unique_groups.size))
        if splits < 2:
            return self.fit_transform(frame, target)
        output: np.ndarray | None = None
        splitter = GroupKFold(n_splits=splits)
        for fit_index, holdout_index in splitter.split(frame, target, groups):
            fold_encoder = HierarchicalTabularEncoder(
                numeric=self.numeric,
                categorical=self.categorical,
                strength=self.strength,
            )
            fold_encoder.fit(frame.iloc[fit_index], target[fit_index])
            fold_values = fold_encoder.transform(frame.iloc[holdout_index])
            if output is None:
                output = np.empty((len(frame), fold_values.shape[1]), dtype=np.float32)
            output[holdout_index] = fold_values
        self.fit(frame, target)
        assert output is not None
        return output


def run_pilot(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(config, "paths", ["landmarks", "artifacts_dir"])
    require_keys(config, "model", ["random_seed", "max_rows_per_split"])
    require_keys(
        config,
        "risk",
        [
            "target_false_reclaim_rate",
            "confidence",
            "threshold_grid_size",
            "minimum_selected",
            "minimum_uplift_over_ttl",
        ],
    )

    landmarks = resolve_project_path(project_root, config["paths"]["landmarks"])
    artifacts = resolve_project_path(project_root, config["paths"]["artifacts_dir"])
    if not landmarks.exists():
        raise FileNotFoundError(f"Missing landmarks table: {landmarks}")
    artifacts.mkdir(parents=True, exist_ok=True)

    seed = int(config["model"]["random_seed"])
    max_rows = int(config["model"]["max_rows_per_split"])
    train_end, calibration_end, min_day, max_day = _resolve_splits(landmarks, config)
    horizon = int(config["label"]["horizon_days"])

    split_conditions = {
        "train": f"day <= {train_end - horizon}",
        "calibration": f"day > {train_end} AND day <= {calibration_end - horizon}",
        "test": f"day > {calibration_end}",
    }
    frames = {
        name: _load_complete_episode_sample(landmarks, condition, max_rows, seed + index)
        for index, (name, condition) in enumerate(split_conditions.items())
    }
    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"The {name} split is empty. Check the episode labels and split boundaries.")
        if frame["reactivates_within_horizon"].nunique() < 2:
            raise ValueError(f"The {name} split has only one label class; the pilot is not identifiable.")

    train = frames["train"]
    calibration = frames["calibration"]
    test = frames["test"]
    y_train = train["reactivates_within_horizon"].to_numpy(int)

    model_cfg = config["model"]
    models: dict[str, tuple[HierarchicalTabularEncoder, HistGradientBoostingClassifier]] = {}
    specifications = {
        "age_only": (["cold_age", "days_since_last_active", "prior_active_days_30"], []),
        "size_only": (SIZE_ONLY_FEATURES, []),
        "hierarchical": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in specifications}
    discrimination_rows: list[dict[str, Any]] = []

    for model_index, (name, (numeric, categorical)) in enumerate(specifications.items()):
        encoder = HierarchicalTabularEncoder(
            numeric=numeric,
            categorical=categorical,
            strength=float(model_cfg["target_encoding_strength"]),
        )
        if categorical:
            deployment_groups = (
                train[["app_id", "model_id", "site_id", "pd_role"]]
                .fillna("__NULL__")
                .astype(str)
                .agg("|".join, axis=1)
                .to_numpy()
            )
            x_train = encoder.fit_transform_group_oof(train, y_train, deployment_groups)
        else:
            x_train = encoder.fit_transform(train, y_train)
        model = HistGradientBoostingClassifier(
            learning_rate=float(model_cfg["learning_rate"]),
            max_iter=int(model_cfg["max_iter"]),
            max_leaf_nodes=int(model_cfg["max_leaf_nodes"]),
            min_samples_leaf=int(model_cfg["min_samples_leaf"]),
            l2_regularization=float(model_cfg["l2_regularization"]),
            random_state=seed + model_index,
        )
        model.fit(x_train, y_train)
        models[name] = (encoder, model)
        for split_name, frame in frames.items():
            scores = model.predict_proba(encoder.transform(frame))[:, 1]
            predictions[name][split_name] = scores
            discrimination_rows.append(
                {
                    "model": name,
                    "split": split_name,
                    **_predictive_metrics(frame["reactivates_within_horizon"].to_numpy(int), scores),
                }
            )
        joblib.dump({"encoder": encoder, "model": model}, artifacts / f"{name}.joblib")

    risk_cfg = config["risk"]
    target_risk = float(risk_cfg["target_false_reclaim_rate"])
    confidence = float(risk_cfg["confidence"])
    grid_size = int(risk_cfg["threshold_grid_size"])
    minimum_selected = int(risk_cfg["minimum_selected"])

    frontier_parts: list[pd.DataFrame] = []
    choices: list[PolicyChoice] = []

    ttl = ttl_frontier(calibration, confidence)
    ttl.insert(0, "policy", "ttl")
    frontier_parts.append(ttl)
    ttl_choice = choose_policy(ttl, "ttl", target_risk, minimum_selected)
    if ttl_choice:
        choices.append(ttl_choice)

    for name in specifications:
        score_column = f"score_{name}"
        calibration[score_column] = predictions[name]["calibration"]
        frontier = score_frontier(calibration, score_column, grid_size, confidence)
        frontier.insert(0, "policy", name)
        frontier_parts.append(frontier)
        choice = choose_policy(frontier, name, target_risk, minimum_selected)
        if choice:
            choices.append(choice)

    combined_frontier = pd.concat(frontier_parts, ignore_index=True)
    combined_frontier.to_csv(artifacts / "policy_frontier.csv", index=False)

    for name in specifications:
        test[f"score_{name}"] = predictions[name]["test"]

    test_policy_rows: list[dict[str, Any]] = []
    total_possible = float(
        test.sort_values(["episode_id", "day"])
        .drop_duplicates("episode_id", keep="first")["remaining_mem_gib_hours"]
        .sum()
    )
    for choice in choices:
        if choice.policy == "ttl":
            actions = first_actions_by_ttl(test, int(round(choice.parameter)))
        else:
            score_column = f"score_{choice.policy}"
            actions = first_actions_by_score(test, score_column, choice.parameter)
        metrics = action_metrics(actions, confidence=confidence, comparisons=1)
        test_policy_rows.append(
            {
                **asdict(choice),
                **{f"test_{key}": value for key, value in metrics.items()},
                "test_resource_fraction": (
                    float(metrics["reclaimed_gib_hours"]) / total_possible if total_possible > 0 else 0.0
                ),
            }
        )

    discrimination = pd.DataFrame(discrimination_rows)
    discrimination.to_csv(artifacts / "predictive_metrics.csv", index=False)
    test_policies = pd.DataFrame(test_policy_rows)
    test_policies.to_csv(artifacts / "selected_policies.csv", index=False)

    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_targets = sorted(
        set(float(value) for value in risk_cfg.get("sensitivity_targets", [target_risk]))
    )
    for sensitivity_target in sensitivity_targets:
        for policy_name in ["ttl", *specifications.keys()]:
            policy_frontier = combined_frontier.loc[combined_frontier["policy"] == policy_name]
            choice = choose_policy(
                policy_frontier,
                policy_name,
                sensitivity_target,
                minimum_selected,
            )
            if choice is None:
                sensitivity_rows.append(
                    {
                        "target_risk": sensitivity_target,
                        "policy": policy_name,
                        "eligible": False,
                    }
                )
                continue
            if policy_name == "ttl":
                actions = first_actions_by_ttl(test, int(round(choice.parameter)))
            else:
                actions = first_actions_by_score(test, f"score_{policy_name}", choice.parameter)
            metrics = action_metrics(actions, confidence=confidence, comparisons=1)
            sensitivity_rows.append(
                {
                    "target_risk": sensitivity_target,
                    "policy": policy_name,
                    "eligible": True,
                    "parameter": choice.parameter,
                    "calibration_actions": choice.calibration_actions,
                    "calibration_risk_ucb": choice.calibration_risk_ucb,
                    "calibration_reclaimed_gib_hours": choice.calibration_reclaimed_gib_hours,
                    **{f"test_{key}": value for key, value in metrics.items()},
                    "test_resource_fraction": (
                        float(metrics["reclaimed_gib_hours"]) / total_possible
                        if total_possible > 0
                        else 0.0
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(artifacts / "risk_sensitivity.csv", index=False)

    dual_rows: list[dict[str, Any]] = []
    action_targets = sorted(
        set(float(value) for value in risk_cfg.get("action_risk_sensitivity_targets", [0.10]))
    )
    for resource_target in sensitivity_targets:
        for action_target in action_targets:
            for policy_name in ["ttl", *specifications.keys()]:
                policy_frontier = combined_frontier.loc[combined_frontier["policy"] == policy_name]
                choice = choose_policy(
                    policy_frontier,
                    policy_name,
                    resource_target,
                    minimum_selected,
                    target_action_risk=action_target,
                )
                if choice is None:
                    dual_rows.append(
                        {
                            "target_resource_risk": resource_target,
                            "target_action_risk": action_target,
                            "policy": policy_name,
                            "eligible": False,
                        }
                    )
                    continue
                if policy_name == "ttl":
                    actions = first_actions_by_ttl(test, int(round(choice.parameter)))
                else:
                    actions = first_actions_by_score(test, f"score_{policy_name}", choice.parameter)
                metrics = action_metrics(actions, confidence=confidence, comparisons=1)
                dual_rows.append(
                    {
                        "target_resource_risk": resource_target,
                        "target_action_risk": action_target,
                        "policy": policy_name,
                        "eligible": True,
                        "parameter": choice.parameter,
                        "calibration_actions": choice.calibration_actions,
                        "calibration_resource_risk_ucb": choice.calibration_risk_ucb,
                        "calibration_action_risk_ucb": choice.calibration_unweighted_risk_ucb,
                        "calibration_reclaimed_gib_hours": choice.calibration_reclaimed_gib_hours,
                        **{f"test_{key}": value for key, value in metrics.items()},
                        "test_resource_fraction": (
                            float(metrics["reclaimed_gib_hours"]) / total_possible
                            if total_possible > 0
                            else 0.0
                        ),
                    }
                )
    dual_sensitivity = pd.DataFrame(dual_rows)
    dual_sensitivity.to_csv(artifacts / "dual_risk_sensitivity.csv", index=False)

    decision = _make_decision(test_policies, discrimination, risk_cfg)
    audit = {
        "source": str(landmarks),
        "day_range": [int(min_day), int(max_day)],
        "train_end_day": int(train_end),
        "calibration_end_day": int(calibration_end),
        "splits": {name: _frame_audit(frame) for name, frame in frames.items()},
        "decision": decision,
        "sensitivity_targets": sensitivity_targets,
        "method_warning": (
            "The pilot uses a Bonferroni-adjusted weighted Wilson upper bound. "
            "It is a conservative screening device, not the final grouped temporal conformal guarantee."
        ),
    }
    with (artifacts / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    report = _render_report(
        audit, discrimination, test_policies, sensitivity, dual_sensitivity, config
    )
    report_path = artifacts / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _resolve_splits(landmarks: Path, config: dict[str, Any]) -> tuple[int, int, int, int]:
    connection = duckdb.connect()
    try:
        min_day, max_day = connection.execute(
            f"SELECT MIN(day), MAX(day) FROM read_parquet('{_sql_string(landmarks)}')"
        ).fetchone()
    finally:
        connection.close()
    if min_day is None or max_day is None:
        raise ValueError("Landmark table contains no rows")
    split = config.get("split", {})
    train_end = split.get("train_end_day")
    calibration_end = split.get("calibration_end_day")
    span = int(max_day) - int(min_day)
    if train_end is None:
        train_end = int(min_day) + int(np.floor(0.60 * span))
    if calibration_end is None:
        calibration_end = int(min_day) + int(np.floor(0.80 * span))
    if not int(min_day) <= int(train_end) < int(calibration_end) < int(max_day):
        raise ValueError("Invalid chronological split boundaries")
    return int(train_end), int(calibration_end), int(min_day), int(max_day)


def _load_complete_episode_sample(
    landmarks: Path,
    condition: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        total_rows, total_episodes = connection.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT episode_id)
            FROM read_parquet('{_sql_string(landmarks)}')
            WHERE {condition}
            """
        ).fetchone()
        if total_rows <= max_rows:
            return connection.execute(
                f"SELECT * FROM read_parquet('{_sql_string(landmarks)}') WHERE {condition}"
            ).fetchdf()
        average_rows = total_rows / max(1, total_episodes)
        episode_limit = max(1, int(max_rows / max(1.0, average_rows)))
        return connection.execute(
            f"""
            WITH selected_episodes AS (
              SELECT episode_id
              FROM read_parquet('{_sql_string(landmarks)}')
              WHERE {condition}
              GROUP BY episode_id
              ORDER BY hash(episode_id, {int(seed)})
              LIMIT {episode_limit}
            )
            SELECT l.*
            FROM read_parquet('{_sql_string(landmarks)}') l
            JOIN selected_episodes s USING (episode_id)
            WHERE {condition}
            """
        ).fetchdf()
    finally:
        connection.close()


def _predictive_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "prevalence": float(np.mean(labels)),
    }


def _frame_audit(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "episodes": int(frame["episode_id"].nunique()),
        "deployments": int(frame[["app_id", "model_id", "site_id", "pd_role"]].drop_duplicates().shape[0]),
        "positive_rate": float(frame["reactivates_within_horizon"].mean()),
        "mem_gib_hours": float(frame["mem_gib_hours"].sum()),
        "min_day": int(frame["day"].min()),
        "max_day": int(frame["day"].max()),
    }


def _make_decision(
    policies: pd.DataFrame,
    discrimination: pd.DataFrame,
    risk_cfg: dict[str, Any],
) -> dict[str, Any]:
    target = float(risk_cfg["target_false_reclaim_rate"])
    minimum_uplift = float(risk_cfg["minimum_uplift_over_ttl"])
    minimum_selected = int(risk_cfg["minimum_selected"])
    if policies.empty or "ttl" not in set(policies["policy"]):
        return {"status": "NO_GO", "reason": "No calibrated TTL baseline satisfied the risk contract."}
    ttl = policies.loc[policies["policy"] == "ttl"].iloc[0]
    learned = policies.loc[policies["policy"].isin(["age_only", "size_only", "hierarchical"])]
    if learned.empty:
        return {"status": "NO_GO", "reason": "No learned policy satisfied the calibration risk contract."}
    best = learned.sort_values("test_reclaimed_gib_hours", ascending=False).iloc[0]
    denominator = max(float(ttl["test_reclaimed_gib_hours"]), 1e-12)
    uplift = float(best["test_reclaimed_gib_hours"] - ttl["test_reclaimed_gib_hours"]) / denominator

    metric = discrimination.loc[
        (discrimination["model"] == "hierarchical") & (discrimination["split"] == "test")
    ].iloc[0]
    size_metric = discrimination.loc[
        (discrimination["model"] == "size_only") & (discrimination["split"] == "test")
    ].iloc[0]
    hierarchy_adds_signal = float(metric["brier"]) < float(size_metric["brier"])
    conditions = {
        "test_weighted_risk_within_target": float(best["test_weighted_risk"]) <= target,
        "uplift_over_ttl_meets_gate": uplift >= minimum_uplift,
        "enough_calibration_actions": int(best["calibration_actions"]) >= minimum_selected,
        "hierarchy_beats_size_only_brier": hierarchy_adds_signal,
    }
    return {
        "status": "GO" if all(conditions.values()) else "NO_GO",
        "best_learned_policy": str(best["policy"]),
        "uplift_over_ttl": uplift,
        "conditions": conditions,
        "note": "Do not tune these gates on the test split.",
    }


def _render_report(
    audit: dict[str, Any],
    discrimination: pd.DataFrame,
    policies: pd.DataFrame,
    sensitivity: pd.DataFrame,
    dual_sensitivity: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    decision = audit["decision"]
    lines = [
        "# ColdGuard Go/No-Go Pilot Report",
        "",
        f"**Decision: {decision['status']}**",
        "",
        "## Data audit",
        "",
        "| Split | Rows | Episodes | Deployments | Positive rate | Day range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in audit["splits"].items():
        lines.append(
            f"| {name} | {values['rows']:,} | {values['episodes']:,} | "
            f"{values['deployments']:,} | {values['positive_rate']:.3%} | "
            f"{values['min_day']}–{values['max_day']} |"
        )

    lines.extend(
        [
            "",
            "## Predictive performance",
            "",
            "| Model | Split | Brier ↓ | AP ↑ | ROC-AUC ↑ |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in discrimination.iterrows():
        lines.append(
            f"| {row['model']} | {row['split']} | {row['brier']:.5f} | "
            f"{row['average_precision']:.5f} | {row['roc_auc']:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Calibrated policies evaluated on the later test period",
            "",
            "| Policy | Parameter | Actions | Weighted false-reclaim risk | Reclaimed GiB-hours | Resource fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if policies.empty:
        lines.append("| none | — | 0 | — | 0 | 0 |")
    else:
        for _, row in policies.iterrows():
            lines.append(
                f"| {row['policy']} | {row['parameter']:.5g} | {int(row['test_actions']):,} | "
                f"{row['test_weighted_risk']:.3%} | {row['test_reclaimed_gib_hours']:,.1f} | "
                f"{row['test_resource_fraction']:.3%} |"
            )

    lines.extend(["", "## Gate results", ""])
    if "conditions" in decision:
        for key, passed in decision["conditions"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{key}`")
        lines.append(f"- Learned-policy uplift over TTL: {decision['uplift_over_ttl']:.2%}")
    else:
        lines.append(f"- {decision.get('reason', 'No eligible policy.')}")

    lines.extend(
        [
            "",
            "## Exploratory risk-budget sensitivity",
            "",
            "The 5% contract above remains the primary gate. Higher budgets are diagnostic and are not a post-hoc replacement for it.",
            "",
            "| Target | Policy | Eligible on calibration | Test resource-weighted risk (UCB) | Test action risk (UCB) | Test reclaimed GiB-hours |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for _, row in sensitivity.iterrows():
        if not bool(row["eligible"]):
            lines.append(f"| {row['target_risk']:.1%} | {row['policy']} | no | — | — | — |")
        else:
            lines.append(
                f"| {row['target_risk']:.1%} | {row['policy']} | yes | "
                f"{row['test_weighted_risk']:.3%} ({row['test_risk_ucb']:.3%}) | "
                f"{row['test_unweighted_risk']:.3%} ({row['test_unweighted_risk_ucb']:.3%}) | "
                f"{row['test_reclaimed_gib_hours']:,.1f} |"
            )

    lines.extend(
        [
            "",
            "## Dual-risk sensitivity at a 10% resource-risk budget",
            "",
            "Selection below simultaneously constrains calibration upper bounds for resource-weighted and action-level risk.",
            "",
            "| Action-risk target | Policy | Eligible | Test resource risk (UCB) | Test action risk (UCB) | Test reclaimed GiB-hours |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    dual_view = dual_sensitivity.loc[
        np.isclose(dual_sensitivity["target_resource_risk"], 0.10)
    ]
    for _, row in dual_view.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {row['target_action_risk']:.1%} | {row['policy']} | no | — | — | — |"
            )
        else:
            lines.append(
                f"| {row['target_action_risk']:.1%} | {row['policy']} | yes | "
                f"{row['test_weighted_risk']:.3%} ({row['test_risk_ucb']:.3%}) | "
                f"{row['test_unweighted_risk']:.3%} ({row['test_unweighted_risk_ucb']:.3%}) | "
                f"{row['test_reclaimed_gib_hours']:,.1f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This pilot estimates trace-level reclaimable allocation. It does not establish real reload latency, "
            "SLO safety, or monetary savings. The primary selector controls resource-weighted harm; the action-level "
            "error rate is reported separately and is not certified by that selector. The selector uses a "
            "Bonferroni-adjusted weighted Wilson bound as a screening device; a publishable version must replace it "
            "with deployment-grouped, temporally split conformal risk control. Five-minute consistency validation "
            "is available as a separate audit.",
            "",
            f"Configured target false-reclaim risk: {float(config['risk']['target_false_reclaim_rate']):.1%}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sql_string(path: Path) -> str:
    return str(path).replace("'", "''")

