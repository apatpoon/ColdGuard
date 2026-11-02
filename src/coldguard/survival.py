from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .config import load_config, require_keys, resolve_project_path
from .crc import choose_grouped_crc_threshold, grouped_harm_metrics
from .pilot import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    HierarchicalTabularEncoder,
    _predictive_metrics,
)
from .policy import (
    action_metrics,
    choose_policy,
    first_actions_by_score,
    first_actions_by_ttl,
    score_frontier,
    ttl_frontier,
)


AGE_FEATURES = ["cold_age", "days_since_last_active", "prior_active_days_30"]


@dataclass(frozen=True)
class RollingFold:
    name: str
    train_end_day: int
    calibration_end_day: int
    test_end_day: int

    def conditions(self, horizon: int) -> dict[str, str]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not self.train_end_day < self.calibration_end_day < self.test_end_day:
            raise ValueError(f"Invalid chronological fold: {self}")
        return {
            "train": f"day <= {self.train_end_day - horizon}",
            "calibration": (
                f"day > {self.train_end_day} "
                f"AND day <= {self.calibration_end_day - horizon}"
            ),
            "test": (
                f"day > {self.calibration_end_day} "
                f"AND day <= {self.test_end_day - horizon}"
            ),
        }


def advance_covariates(frame: pd.DataFrame, offset_days: int) -> pd.DataFrame:
    """Return covariates conditional on the deployment remaining cold.

    Only deterministic clocks advance. Other time-varying observations use a
    last-observation-carried-forward assumption and remain unchanged.
    """
    if offset_days < 0:
        raise ValueError("offset_days must be non-negative")
    advanced = frame.copy()
    advanced["cold_age"] = pd.to_numeric(advanced["cold_age"], errors="coerce") + offset_days
    elapsed = pd.to_numeric(advanced["days_since_last_active"], errors="coerce")
    advanced["days_since_last_active"] = elapsed.where(elapsed.isna(), elapsed + offset_days)
    return advanced


def cumulative_incidence_from_hazards(hazards: np.ndarray) -> np.ndarray:
    """Compose discrete conditional hazards into cumulative incidence.

    Parameters
    ----------
    hazards:
        Array shaped ``[n_samples, horizon]`` with values in ``[0, 1]``.
    """
    values = np.asarray(hazards, dtype=float)
    if values.ndim != 2:
        raise ValueError("hazards must have shape [n_samples, horizon]")
    if values.shape[1] < 1:
        raise ValueError("hazards must include at least one future step")
    if not np.isfinite(values).all():
        raise ValueError("hazards contain non-finite values")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("hazards must be probabilities")
    clipped = np.clip(values, 0.0, 1.0 - 1e-12)
    log_survival = np.log1p(-clipped).sum(axis=1)
    return -np.expm1(log_survival)


def predict_cumulative_incidence(
    frame: pd.DataFrame,
    encoder: HierarchicalTabularEncoder,
    model: HistGradientBoostingClassifier,
    horizon: int,
    advance_clocks: bool = True,
) -> np.ndarray:
    hazards = []
    for offset in range(horizon):
        future = advance_covariates(frame, offset) if advance_clocks else frame
        hazards.append(model.predict_proba(encoder.transform(future))[:, 1])
    return cumulative_incidence_from_hazards(np.column_stack(hazards))


def predict_cumulative_incidence_path(
    frame: pd.DataFrame,
    encoder: HierarchicalTabularEncoder,
    model: HistGradientBoostingClassifier,
    horizon: int,
    prior_active_days_path: np.ndarray,
) -> np.ndarray:
    """Predict cumulative incidence along a declared future covariate path.

    ``prior_active_days_path`` must have shape ``[n_samples, horizon]``.  The
    value in column ``k`` is the mechanically updated 30-day active-day count
    at future offset ``k``, conditional on the deployment remaining cold.
    Deterministic clocks advance as in :func:`advance_covariates`; all other
    time-varying measurements retain the declared LOCF scenario.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    path = np.asarray(prior_active_days_path, dtype=float)
    if path.shape != (len(frame), horizon):
        raise ValueError(
            "prior_active_days_path must have shape "
            f"({len(frame)}, {horizon}), got {path.shape}"
        )
    if not np.isfinite(path).all():
        raise ValueError("prior_active_days_path contains non-finite values")
    if (path < 0.0).any():
        raise ValueError("prior_active_days_path must be non-negative")

    hazards = []
    for offset in range(horizon):
        future = advance_covariates(frame, offset)
        future["prior_active_days_30"] = path[:, offset]
        hazards.append(model.predict_proba(encoder.transform(future))[:, 1])
    return cumulative_incidence_from_hazards(np.column_stack(hazards))


def interval_to_horizon_score(
    predicted_days: np.ndarray, horizon: int
) -> np.ndarray:
    """Map a next-invocation interval forecast to a coherent horizon risk score.

    The result is a ranking score, not a claimed probability. It is monotone in
    both the predicted interval and the decision horizon.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = np.asarray(predicted_days, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predicted_days contains non-finite values")
    values = np.clip(values, 0.0, None)
    return np.exp(-values / float(horizon))


def predict_next_interval_score(
    frame: pd.DataFrame,
    encoder: HierarchicalTabularEncoder,
    model: HistGradientBoostingRegressor,
    horizon: int,
) -> np.ndarray:
    predicted_log_days = model.predict(encoder.transform(frame))
    predicted_days = np.expm1(predicted_log_days)
    return interval_to_horizon_score(predicted_days, horizon)


def run_survival(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(config, "paths", ["landmarks", "survival_artifacts_dir"])
    require_keys(config, "model", ["random_seed", "target_encoding_strength"])
    require_keys(
        config,
        "risk",
        ["confidence", "threshold_grid_size", "minimum_selected"],
    )
    require_keys(
        config,
        "survival",
        [
            "horizon_days",
            "target_resource_risk",
            "target_action_risk",
            "folds",
        ],
    )

    landmarks = resolve_project_path(project_root, config["paths"]["landmarks"])
    artifacts = resolve_project_path(project_root, config["paths"]["survival_artifacts_dir"])
    if not landmarks.exists():
        raise FileNotFoundError(landmarks)
    artifacts.mkdir(parents=True, exist_ok=True)
    models_dir = artifacts / "models"
    models_dir.mkdir(exist_ok=True)

    survival_cfg = config["survival"]
    horizon = int(survival_cfg["horizon_days"])
    folds = [RollingFold(**values) for values in survival_cfg["folds"]]
    resource_target = float(survival_cfg["target_resource_risk"])
    action_target = float(survival_cfg["target_action_risk"])
    risk_cfg = config["risk"]
    confidence = float(risk_cfg["confidence"])
    grid_size = int(risk_cfg["threshold_grid_size"])
    minimum_selected = int(risk_cfg["minimum_selected"])
    seed = int(config["model"]["random_seed"])

    metric_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    crc_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(folds):
        frames = {
            split: _load_frame(landmarks, condition)
            for split, condition in fold.conditions(horizon).items()
        }
        for split, frame in frames.items():
            _validate_frame(frame, split, fold)
            split_rows.append(
                {
                    "fold": fold.name,
                    "split": split,
                    "rows": len(frame),
                    "episodes": frame["episode_id"].nunique(),
                    "deployments": _deployment_groups(frame).nunique(),
                    "positive_rate_7d": frame["reactivates_within_horizon"].mean(),
                    "positive_rate_next_day": frame["event_next_day"].mean(),
                    "min_day": frame["day"].min(),
                    "max_day": frame["day"].max(),
                }
            )

        train = frames["train"]
        calibration = frames["calibration"].copy()
        test = frames["test"].copy()
        groups = _deployment_groups(train).to_numpy()
        specifications = {
            "age": (AGE_FEATURES, []),
            "hierarchical": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        }

        scores: dict[str, dict[str, np.ndarray]] = {}
        for feature_index, (feature_name, (numeric, categorical)) in enumerate(
            specifications.items()
        ):
            for target_name, target_column in [
                ("direct", "reactivates_within_horizon"),
                ("hazard", "event_next_day"),
            ]:
                model_name = f"{target_name}_{feature_name}"
                encoder, model = _fit_classifier(
                    train,
                    target_column,
                    numeric,
                    categorical,
                    groups,
                    config["model"],
                    seed + 100 * fold_index + 10 * feature_index + (target_name == "hazard"),
                )
                joblib.dump(
                    {"encoder": encoder, "model": model, "horizon": horizon},
                    models_dir / f"{fold.name}_{model_name}.joblib",
                )
                prediction_modes = (
                    [(model_name, True), (f"hazard_frozen_{feature_name}", False)]
                    if target_name == "hazard"
                    else [(model_name, True)]
                )
                for method, advance_clocks in prediction_modes:
                    scores[method] = {}
                    for split, frame in frames.items():
                        if target_name == "hazard":
                            prediction = predict_cumulative_incidence(
                                frame,
                                encoder,
                                model,
                                horizon,
                                advance_clocks=advance_clocks,
                            )
                        else:
                            prediction = model.predict_proba(encoder.transform(frame))[:, 1]
                        scores[method][split] = prediction
                        metric_rows.append(
                            {
                                "fold": fold.name,
                                "method": method,
                                "split": split,
                                **_predictive_metrics(
                                    frame["reactivates_within_horizon"].to_numpy(int),
                                    prediction,
                                ),
                            }
                        )

            interval_train = train.loc[
                train["event_day"].notna()
                & (pd.to_numeric(train["event_day"]) <= fold.train_end_day)
            ].copy()
            interval_encoder, interval_model = _fit_next_interval_regressor(
                interval_train,
                numeric,
                categorical,
                _deployment_groups(interval_train).to_numpy(),
                config["model"],
                seed + 100 * fold_index + 10 * feature_index + 7,
            )
            method = f"next_interval_{feature_name}"
            joblib.dump(
                {
                    "encoder": interval_encoder,
                    "model": interval_model,
                    "horizon": horizon,
                    "training_contract": (
                        "observed next events ending no later than train boundary"
                    ),
                },
                models_dir / f"{fold.name}_{method}.joblib",
            )
            scores[method] = {}
            for split, frame in frames.items():
                prediction = predict_next_interval_score(
                    frame, interval_encoder, interval_model, horizon
                )
                scores[method][split] = prediction
                metric_rows.append(
                    {
                        "fold": fold.name,
                        "method": method,
                        "split": split,
                        **_predictive_metrics(
                            frame["reactivates_within_horizon"].to_numpy(int),
                            prediction,
                        ),
                    }
                )

        frontiers: dict[str, pd.DataFrame] = {"ttl": ttl_frontier(calibration, confidence)}
        for method in scores:
            column = f"score_{method}"
            calibration[column] = scores[method]["calibration"]
            test[column] = scores[method]["test"]
            frontiers[method] = score_frontier(
                calibration, column, grid_size, confidence
            )

        total_possible = float(
            test.sort_values(["episode_id", "day"])
            .drop_duplicates("episode_id", keep="first")["remaining_mem_gib_hours"]
            .sum()
        )
        for method, frontier in frontiers.items():
            choice = choose_policy(
                frontier,
                method,
                resource_target,
                minimum_selected,
                target_action_risk=action_target,
            )
            if choice is None:
                policy_rows.append(
                    {"fold": fold.name, "method": method, "eligible": False}
                )
                continue
            if method == "ttl":
                actions = first_actions_by_ttl(test, int(round(choice.parameter)))
            else:
                actions = first_actions_by_score(test, f"score_{method}", choice.parameter)
            metrics = action_metrics(actions, confidence=confidence, comparisons=1)
            test_resource_pass = float(metrics["risk_ucb"]) <= resource_target
            test_action_pass = float(metrics["unweighted_risk_ucb"]) <= action_target
            policy_rows.append(
                {
                    "fold": fold.name,
                    "method": method,
                    "eligible": True,
                    **asdict(choice),
                    **{f"test_{key}": value for key, value in metrics.items()},
                    "test_resource_contract_pass": test_resource_pass,
                    "test_action_contract_pass": test_action_pass,
                    "test_dual_contract_pass": test_resource_pass and test_action_pass,
                    "test_resource_fraction": (
                        float(metrics["reclaimed_gib_hours"]) / total_possible
                        if total_possible > 0
                        else 0.0
                    ),
                }
            )

        fixed_crc_thresholds = np.linspace(0.0, 1.0, 101)
        for method in scores:
            choice = choose_grouped_crc_threshold(
                calibration,
                f"score_{method}",
                fixed_crc_thresholds,
                target_action_harm=action_target,
                target_resource_harm=resource_target,
                minimum_actions=minimum_selected,
            )
            if choice is None:
                crc_rows.append(
                    {"fold": fold.name, "method": method, "eligible": False}
                )
                continue
            actions = first_actions_by_score(
                test, f"score_{method}", choice.threshold
            )
            conditional = action_metrics(actions, confidence=confidence, comparisons=1)
            grouped = grouped_harm_metrics(test, actions)
            crc_rows.append(
                {
                    "fold": fold.name,
                    "method": method,
                    "eligible": True,
                    **asdict(choice),
                    **{f"test_{key}": value for key, value in conditional.items()},
                    **{f"test_{key}": value for key, value in grouped.items()},
                    "test_grouped_contract_pass": (
                        float(grouped["deployment_action_harm"]) <= action_target
                        and float(grouped["deployment_resource_harm"]) <= resource_target
                    ),
                    "test_resource_fraction": (
                        float(conditional["reclaimed_gib_hours"]) / total_possible
                        if total_possible > 0
                        else 0.0
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    policies = pd.DataFrame(policy_rows)
    crc_policies = pd.DataFrame(crc_rows)
    splits = pd.DataFrame(split_rows)
    metrics.to_csv(artifacts / "rolling_predictive_metrics.csv", index=False)
    policies.to_csv(artifacts / "rolling_dual_policies.csv", index=False)
    crc_policies.to_csv(artifacts / "rolling_grouped_crc.csv", index=False)
    splits.to_csv(artifacts / "rolling_split_audit.csv", index=False)
    metadata = {
        "horizon_days": horizon,
        "target_resource_risk": resource_target,
        "target_action_risk": action_target,
        "folds": [asdict(fold) for fold in folds],
        "future_covariate_assumption": (
            "cold_age and non-null days_since_last_active advance; all other covariates use LOCF"
        ),
        "risk_warning": (
            "Wilson screening bounds are not a grouped temporal conformal guarantee"
        ),
        "next_interval_baseline": (
            "uncensored next-invocation regression trained only on events observed by the training boundary; "
            "exp(-predicted_days/horizon) is a coherent ranking score, not a calibrated probability"
        ),
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = _render_report(metrics, policies, crc_policies, splits, metadata)
    report_path = artifacts / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _fit_classifier(
    train: pd.DataFrame,
    target_column: str,
    numeric: list[str],
    categorical: list[str],
    groups: np.ndarray,
    model_cfg: dict[str, Any],
    seed: int,
) -> tuple[HierarchicalTabularEncoder, HistGradientBoostingClassifier]:
    target = train[target_column].to_numpy(int)
    if np.unique(target).size < 2:
        raise ValueError(f"Training target {target_column} contains one class")
    encoder = HierarchicalTabularEncoder(
        numeric=numeric,
        categorical=categorical,
        strength=float(model_cfg["target_encoding_strength"]),
    )
    if categorical:
        x_train = encoder.fit_transform_group_oof(train, target, groups)
    else:
        x_train = encoder.fit_transform(train, target)
    model = HistGradientBoostingClassifier(
        learning_rate=float(model_cfg["learning_rate"]),
        max_iter=int(model_cfg["max_iter"]),
        max_leaf_nodes=int(model_cfg["max_leaf_nodes"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        l2_regularization=float(model_cfg["l2_regularization"]),
        random_state=seed,
    )
    model.fit(x_train, target)
    return encoder, model


def _fit_next_interval_regressor(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    groups: np.ndarray,
    model_cfg: dict[str, Any],
    seed: int,
) -> tuple[HierarchicalTabularEncoder, HistGradientBoostingRegressor]:
    """Fit an uncensored next-invocation baseline in the learned-cache style."""
    if train.empty:
        raise ValueError("No observed next-invocation events for interval regression")
    event_day = pd.to_numeric(train["event_day"], errors="raise")
    day = pd.to_numeric(train["day"], errors="raise")
    residual_days = (event_day - day).to_numpy(float)
    if (residual_days <= 0).any():
        raise ValueError("Next-invocation residual days must be positive")
    target = np.log1p(residual_days)
    encoder = HierarchicalTabularEncoder(
        numeric=numeric,
        categorical=categorical,
        strength=float(model_cfg["target_encoding_strength"]),
    )
    if categorical:
        x_train = encoder.fit_transform_group_oof(train, target, groups)
    else:
        x_train = encoder.fit_transform(train, target)
    model = HistGradientBoostingRegressor(
        learning_rate=float(model_cfg["learning_rate"]),
        max_iter=int(model_cfg["max_iter"]),
        max_leaf_nodes=int(model_cfg["max_leaf_nodes"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        l2_regularization=float(model_cfg["l2_regularization"]),
        random_state=seed,
    )
    model.fit(x_train, target)
    return encoder, model


def _load_frame(landmarks: Path, condition: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT * FROM read_parquet('{_sql_string(landmarks)}') WHERE {condition}"
        ).fetchdf()
    finally:
        con.close()


def _deployment_groups(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[["app_id", "model_id", "site_id", "pd_role"]]
        .fillna("__NULL__")
        .astype(str)
        .agg("|".join, axis=1)
    )


def _validate_frame(frame: pd.DataFrame, split: str, fold: RollingFold) -> None:
    if frame.empty:
        raise ValueError(f"Empty {split} split in fold {fold.name}")
    for target in ["reactivates_within_horizon", "event_next_day"]:
        if frame[target].nunique() < 2:
            raise ValueError(f"{fold.name}/{split}/{target} has one class")


def _render_report(
    metrics: pd.DataFrame,
    policies: pd.DataFrame,
    crc_policies: pd.DataFrame,
    splits: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# ColdGuard Rolling Discrete-Survival Report",
        "",
        f"Horizon: {metadata['horizon_days']} days. Calibration contract: "
        f"resource-risk UCB ≤ {metadata['target_resource_risk']:.1%} and "
        f"action-risk UCB ≤ {metadata['target_action_risk']:.1%}.",
        "",
        "## Split audit",
        "",
        "| Fold | Split | Rows | Episodes | Deployments | 7d positive | Next-day event | Days |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in splits.iterrows():
        lines.append(
            f"| {row['fold']} | {row['split']} | {int(row['rows']):,} | "
            f"{int(row['episodes']):,} | {int(row['deployments']):,} | "
            f"{row['positive_rate_7d']:.2%} | {row['positive_rate_next_day']:.2%} | "
            f"{int(row['min_day'])}–{int(row['max_day'])} |"
        )

    test_metrics = metrics.loc[metrics["split"] == "test"]
    lines.extend(
        [
            "",
            "## Frozen test prediction",
            "",
            "| Fold | Method | Brier ↓ | AP ↑ | ROC-AUC ↑ |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in test_metrics.iterrows():
        lines.append(
            f"| {row['fold']} | {row['method']} | {row['brier']:.5f} | "
            f"{row['average_precision']:.5f} | {row['roc_auc']:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Dual-risk policies on frozen test periods",
            "",
            "| Fold | Method | Calibration eligible | Test contract | Resource risk (UCB) | Action risk (UCB) | Reclaimed GiB-hours | Resource share |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in policies.iterrows():
        if not bool(row["eligible"]):
            lines.append(f"| {row['fold']} | {row['method']} | no | — | — | — | — | — |")
        else:
            lines.append(
                f"| {row['fold']} | {row['method']} | yes | "
                f"{'PASS' if bool(row['test_dual_contract_pass']) else 'FAIL'} | "
                f"{row['test_weighted_risk']:.2%} ({row['test_risk_ucb']:.2%}) | "
                f"{row['test_unweighted_risk']:.2%} ({row['test_unweighted_risk_ucb']:.2%}) | "
                f"{row['test_reclaimed_gib_hours']:,.0f} | {row['test_resource_fraction']:.2%} |"
            )

    lines.extend(
        [
            "",
            "## Deployment-grouped conformal risk control prototype",
            "",
            "The bounded CRC losses are (1) any false reclaim within a deployment and "
            "(2) falsely reclaimed resource divided by that deployment's potential resource. "
            "Non-monotone first-action losses are replaced by their conservative cumulative-maximum envelope.",
            "",
            "| Fold | Method | Calibration eligible | Test grouped contract | Deployment action harm | Deployment resource harm | Conditional action risk | Reclaimed GiB-hours |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in crc_policies.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {row['fold']} | {row['method']} | no | — | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {row['fold']} | {row['method']} | yes | "
                f"{'PASS' if bool(row['test_grouped_contract_pass']) else 'FAIL'} | "
                f"{row['test_deployment_action_harm']:.2%} | "
                f"{row['test_deployment_resource_harm']:.2%} | "
                f"{row['test_unweighted_risk']:.2%} | "
                f"{row['test_reclaimed_gib_hours']:,.0f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"Future covariates: {metadata['future_covariate_assumption']}. "
            f"Next-interval baseline: {metadata['next_interval_baseline']}. "
            f"{metadata['risk_warning']}. A policy can pass calibration and fail a later "
            "period; such failures are evidence of drift, not rows to remove. The grouped CRC "
            "guarantee requires exchangeable deployment groups. Repeated deployments and chronological "
            "shift violate exact exchangeability here, so rolling tests are empirical stress tests, not "
            "an unconditional finite-sample guarantee.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sql_string(path: Path) -> str:
    return str(path).replace("'", "''")

