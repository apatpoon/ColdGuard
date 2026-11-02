from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from .config import load_config, require_keys, resolve_project_path
from .pilot import CATEGORICAL_FEATURES, NUMERIC_FEATURES, _predictive_metrics
from .policy import (
    action_metrics,
    choose_policy,
    first_actions_by_score,
    first_actions_by_ttl,
    score_frontier,
    ttl_frontier,
)
from .survival import (
    AGE_FEATURES,
    _deployment_groups,
    _fit_classifier,
    _fit_next_interval_regressor,
    predict_cumulative_incidence,
    predict_next_interval_score,
)


def horizon_is_observed(frame: pd.DataFrame, horizon: int) -> pd.Series:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    event_day = pd.to_numeric(frame["event_day"], errors="coerce")
    day = pd.to_numeric(frame["day"], errors="raise")
    end_day = pd.to_numeric(frame["end_cold_day"], errors="raise")
    return ((event_day.notna()) & (event_day <= day + horizon)) | (
        end_day >= day + horizon
    )


def attach_horizon_label(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    observed = horizon_is_observed(frame, horizon)
    result = frame.loc[observed].copy()
    event_day = pd.to_numeric(result["event_day"], errors="coerce")
    result["reactivates_within_horizon"] = (
        event_day.notna() & (event_day <= result["day"] + horizon)
    ).astype(int)
    return result


def monotonicity_audit(
    predictions: dict[int, np.ndarray], tolerance: float = 1e-12
) -> dict[str, float | int]:
    horizons = sorted(predictions)
    matrix = np.column_stack([np.asarray(predictions[h], dtype=float) for h in horizons])
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("at least two horizons are required")
    differences = np.diff(matrix, axis=1)
    violation = np.any(differences < -tolerance, axis=1)
    magnitude = np.maximum(-differences, 0.0).sum(axis=1)
    return {
        "rows": int(len(matrix)),
        "violation_rows": int(violation.sum()),
        "violation_rate": float(violation.mean()),
        "mean_violation_magnitude": float(magnitude.mean()),
        "max_violation_magnitude": float(magnitude.max(initial=0.0)),
    }


def run_multihorizon(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(
        config,
        "paths",
        ["landmarks_all", "multihorizon_artifacts_dir"],
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

    landmarks = resolve_project_path(project_root, config["paths"]["landmarks_all"])
    artifacts = resolve_project_path(
        project_root, config["paths"]["multihorizon_artifacts_dir"]
    )
    if not landmarks.exists():
        raise FileNotFoundError(landmarks)
    artifacts.mkdir(parents=True, exist_ok=True)
    models_dir = artifacts / "models"
    models_dir.mkdir(exist_ok=True)

    cfg = config["multihorizon"]
    horizons = sorted(set(int(value) for value in cfg["horizons"]))
    if not horizons or horizons[0] < 1:
        raise ValueError("multihorizon.horizons must be positive")
    max_horizon = max(horizons)
    train_end = int(cfg["train_end_day"])
    calibration_end = int(cfg["calibration_end_day"])
    test_end = int(cfg["test_end_day"])
    if not train_end < calibration_end < test_end:
        raise ValueError("Invalid multihorizon chronological boundaries")

    common_calibration = attach_horizon_label(
        _load_frame(
            landmarks,
            f"day > {train_end} AND day <= {calibration_end - max_horizon}",
        ),
        max_horizon,
    )
    common_test = attach_horizon_label(
        _load_frame(
            landmarks,
            f"day > {calibration_end} AND day <= {test_end - max_horizon}",
        ),
        max_horizon,
    )
    if common_calibration.empty or common_test.empty:
        raise ValueError("Common multi-horizon calibration or test frame is empty")

    model_cfg = config["model"]
    seed = int(model_cfg["random_seed"])
    specifications = {
        "age": (AGE_FEATURES, []),
        "hierarchical": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    hazard_train = attach_horizon_label(
        _load_frame(landmarks, f"day <= {train_end - 1}"), 1
    )
    hazard_train["event_next_day"] = hazard_train["reactivates_within_horizon"]
    hazard_groups = _deployment_groups(hazard_train).to_numpy()

    predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for feature_index, (feature_name, (numeric, categorical)) in enumerate(
        specifications.items()
    ):
        encoder, model = _fit_classifier(
            hazard_train,
            "event_next_day",
            numeric,
            categorical,
            hazard_groups,
            model_cfg,
            seed + 1000 + feature_index,
        )
        joblib.dump(
            {"encoder": encoder, "model": model, "horizons": horizons},
            models_dir / f"hazard_{feature_name}.joblib",
        )
        hazard_method = f"hazard_{feature_name}"
        predictions[hazard_method] = {}
        for horizon in horizons:
            predictions[hazard_method][horizon] = {
                "calibration": predict_cumulative_incidence(
                    common_calibration, encoder, model, horizon
                ),
                "test": predict_cumulative_incidence(common_test, encoder, model, horizon),
            }

        for horizon in horizons:
            direct_train = attach_horizon_label(
                _load_frame(landmarks, f"day <= {train_end - horizon}"), horizon
            )
            groups = _deployment_groups(direct_train).to_numpy()
            direct_encoder, direct_model = _fit_classifier(
                direct_train,
                "reactivates_within_horizon",
                numeric,
                categorical,
                groups,
                model_cfg,
                seed + 100 * horizon + feature_index,
            )
            direct_method = f"direct_{feature_name}"
            predictions.setdefault(direct_method, {})[horizon] = {
                "calibration": direct_model.predict_proba(
                    direct_encoder.transform(common_calibration)
                )[:, 1],
                "test": direct_model.predict_proba(
                    direct_encoder.transform(common_test)
                )[:, 1],
            }
            joblib.dump(
                {
                    "encoder": direct_encoder,
                    "model": direct_model,
                    "horizon": horizon,
                },
                models_dir / f"direct_{feature_name}_{horizon}d.joblib",
            )
            audit_rows.append(
                {
                    "horizon": horizon,
                    "feature_set": feature_name,
                    "direct_train_rows": len(direct_train),
                    "direct_train_positive_rate": direct_train[
                        "reactivates_within_horizon"
                    ].mean(),
                    "hazard_train_rows": len(hazard_train),
                    "hazard_train_next_day_rate": hazard_train["event_next_day"].mean(),
                    "calibration_rows": len(common_calibration),
                    "test_rows": len(common_test),
                }
            )

        interval_train = _load_frame(
            landmarks,
            f"day <= {train_end - 1} AND event_day IS NOT NULL AND event_day <= {train_end}",
        )
        interval_encoder, interval_model = _fit_next_interval_regressor(
            interval_train,
            numeric,
            categorical,
            _deployment_groups(interval_train).to_numpy(),
            model_cfg,
            seed + 2000 + feature_index,
        )
        interval_method = f"next_interval_{feature_name}"
        predictions[interval_method] = {}
        for horizon in horizons:
            predictions[interval_method][horizon] = {
                "calibration": predict_next_interval_score(
                    common_calibration, interval_encoder, interval_model, horizon
                ),
                "test": predict_next_interval_score(
                    common_test, interval_encoder, interval_model, horizon
                ),
            }
        joblib.dump(
            {
                "encoder": interval_encoder,
                "model": interval_model,
                "horizons": horizons,
                "training_contract": (
                    "observed next events ending no later than train boundary"
                ),
            },
            models_dir / f"{interval_method}.joblib",
        )

    labels: dict[int, dict[str, np.ndarray]] = {}
    for horizon in horizons:
        labels[horizon] = {}
        for split, frame in [
            ("calibration", common_calibration),
            ("test", common_test),
        ]:
            event_day = pd.to_numeric(frame["event_day"], errors="coerce")
            label = (event_day.notna() & (event_day <= frame["day"] + horizon)).astype(int)
            labels[horizon][split] = label.to_numpy(int)
            for method in predictions:
                metric_rows.append(
                    {
                        "horizon": horizon,
                        "method": method,
                        "split": split,
                        **_predictive_metrics(
                            labels[horizon][split], predictions[method][horizon][split]
                        ),
                    }
                )

    consistency_rows = []
    for method in predictions:
        for split in ["calibration", "test"]:
            values = {
                horizon: predictions[method][horizon][split] for horizon in horizons
            }
            consistency_rows.append(
                {"method": method, "split": split, **monotonicity_audit(values)}
            )

    risk_cfg = config["risk"]
    confidence = float(risk_cfg["confidence"])
    grid_size = int(risk_cfg["threshold_grid_size"])
    minimum_selected = int(risk_cfg["minimum_selected"])
    resource_target = float(cfg["target_resource_risk"])
    action_target = float(cfg["target_action_risk"])
    policy_rows = []
    for horizon in horizons:
        calibration = common_calibration.copy()
        test = common_test.copy()
        calibration["reactivates_within_horizon"] = labels[horizon]["calibration"]
        test["reactivates_within_horizon"] = labels[horizon]["test"]
        total_possible = float(
            test.sort_values(["episode_id", "day"])
            .drop_duplicates("episode_id", keep="first")["remaining_mem_gib_hours"]
            .sum()
        )
        frontiers: dict[str, pd.DataFrame] = {"ttl": ttl_frontier(calibration, confidence)}
        for method in predictions:
            column = f"score_{method}"
            calibration[column] = predictions[method][horizon]["calibration"]
            test[column] = predictions[method][horizon]["test"]
            frontiers[method] = score_frontier(calibration, column, grid_size, confidence)
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
                    {"horizon": horizon, "method": method, "eligible": False}
                )
                continue
            if method == "ttl":
                actions = first_actions_by_ttl(test, int(round(choice.parameter)))
            else:
                actions = first_actions_by_score(test, f"score_{method}", choice.parameter)
            metrics = action_metrics(actions, confidence=confidence, comparisons=1)
            policy_rows.append(
                {
                    "horizon": horizon,
                    "method": method,
                    "eligible": True,
                    **asdict(choice),
                    **{f"test_{key}": value for key, value in metrics.items()},
                    "test_dual_contract_pass": (
                        float(metrics["risk_ucb"]) <= resource_target
                        and float(metrics["unweighted_risk_ucb"]) <= action_target
                    ),
                    "test_resource_fraction": (
                        float(metrics["reclaimed_gib_hours"]) / total_possible
                        if total_possible > 0
                        else 0.0
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    policies = pd.DataFrame(policy_rows)
    consistency = pd.DataFrame(consistency_rows)
    audit = pd.DataFrame(audit_rows)
    prediction_columns = [
        "day",
        "app_id",
        "model_id",
        "base_id",
        "site_id",
        "pd_role",
        "cluster_id",
        "gpu_type",
        "episode_id",
        "cold_age",
        "remaining_mem_gib_hours",
    ]
    calibration_predictions = common_calibration[prediction_columns].copy()
    test_predictions = common_test[prediction_columns].copy()
    for horizon in horizons:
        calibration_predictions[f"label_{horizon}d"] = labels[horizon]["calibration"]
        test_predictions[f"label_{horizon}d"] = labels[horizon]["test"]
        for method in predictions:
            calibration_predictions[f"score_{method}_{horizon}d"] = predictions[method][horizon][
                "calibration"
            ]
            test_predictions[f"score_{method}_{horizon}d"] = predictions[method][horizon]["test"]
    metrics.to_csv(artifacts / "predictive_metrics.csv", index=False)
    policies.to_csv(artifacts / "dual_risk_policies.csv", index=False)
    consistency.to_csv(artifacts / "probability_consistency.csv", index=False)
    audit.to_csv(artifacts / "data_audit.csv", index=False)
    calibration_predictions.to_parquet(
        artifacts / "calibration_predictions.parquet", index=False
    )
    test_predictions.to_parquet(artifacts / "test_predictions.parquet", index=False)
    metadata = {
        "horizons": horizons,
        "train_end_day": train_end,
        "calibration_end_day": calibration_end,
        "test_end_day": test_end,
        "common_calibration_days": [
            int(common_calibration["day"].min()),
            int(common_calibration["day"].max()),
        ],
        "common_test_days": [int(common_test["day"].min()), int(common_test["day"].max())],
        "target_resource_risk": resource_target,
        "target_action_risk": action_target,
        "next_interval_baseline": (
            "uncensored next-invocation regression; exp(-predicted_days/horizon) is a coherent ranking score"
        ),
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = _render_report(metrics, policies, consistency, metadata)
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
    metrics: pd.DataFrame,
    policies: pd.DataFrame,
    consistency: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# ColdGuard Multi-Horizon Survival Report",
        "",
        f"Horizons: {metadata['horizons']}. Common calibration days: "
        f"{metadata['common_calibration_days'][0]}–{metadata['common_calibration_days'][1]}; "
        f"common test days: {metadata['common_test_days'][0]}–{metadata['common_test_days'][1]}.",
        f"Next-interval baseline: {metadata['next_interval_baseline']}.",
        "",
        "## Frozen test prediction",
        "",
        "| Horizon | Method | Brier ↓ | AP ↑ | ROC-AUC ↑ |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, row in metrics.loc[metrics["split"] == "test"].iterrows():
        lines.append(
            f"| {int(row['horizon'])} | {row['method']} | {row['brier']:.5f} | "
            f"{row['average_precision']:.5f} | {row['roc_auc']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-horizon probability consistency",
            "",
            "| Method | Split | Violation rate | Mean magnitude | Maximum magnitude |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in consistency.iterrows():
        lines.append(
            f"| {row['method']} | {row['split']} | {row['violation_rate']:.2%} | "
            f"{row['mean_violation_magnitude']:.5f} | {row['max_violation_magnitude']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## 10%/10% conditional dual-risk policies",
            "",
            "| Horizon | Method | Calibration eligible | Test contract | Resource risk UCB | Action risk UCB | Reclaimed GiB-hours |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in policies.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {int(row['horizon'])} | {row['method']} | no | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {int(row['horizon'])} | {row['method']} | yes | "
                f"{'PASS' if bool(row['test_dual_contract_pass']) else 'FAIL'} | "
                f"{row['test_risk_ucb']:.2%} | {row['test_unweighted_risk_ucb']:.2%} | "
                f"{row['test_reclaimed_gib_hours']:,.0f} |"
            )
    return "\n".join(lines) + "\n"


def _sql_string(path: Path) -> str:
    return str(path).replace("'", "''")

