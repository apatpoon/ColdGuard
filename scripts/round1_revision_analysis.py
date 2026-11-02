from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from coldguard.clustered_risk import DualRiskBound, dual_clustered_risk_bound
from coldguard.config import load_config, resolve_project_path
from coldguard.multihorizon import attach_horizon_label
from coldguard.policy import first_actions_by_score, first_actions_by_ttl
from coldguard.survival import predict_cumulative_incidence, predict_next_interval_score


ROOT = Path(__file__).resolve().parents[1]


def _flatten_bound(prefix: str, bound: DualRiskBound) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for risk_name, risk in [("action", bound.action), ("resource", bound.resource)]:
        for key, value in asdict(risk).items():
            row[f"{prefix}_{risk_name}_{key}"] = value
    return row


def _candidate_scores(frame: pd.DataFrame, score_column: str, grid_size: int) -> np.ndarray:
    values = frame[score_column].dropna().to_numpy(float)
    quantiles = np.linspace(0.01, 0.99, max(3, grid_size))
    return np.unique(np.quantile(values, quantiles))


def _select_score_policy(
    calibration: pd.DataFrame,
    score_column: str,
    grid_size: int,
    minimum_actions: int,
    target_action: float,
    target_resource: float,
    family_confidence: float,
    exposure_column: str,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    audit: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    testing_started = False
    for index, threshold in enumerate(_candidate_scores(calibration, score_column, grid_size)):
        actions = first_actions_by_score(calibration, score_column, float(threshold))
        if len(actions) < minimum_actions:
            continue
        testing_started = True
        bound = dual_clustered_risk_bound(
            actions, family_confidence, exposure_column=exposure_column
        )
        passed = bound.action.upper <= target_action and bound.resource.upper <= target_resource
        row = {
            "candidate_index": index,
            "parameter": float(threshold),
            "actions": len(actions),
            "trace_exposure_gib_hours": float(actions[exposure_column].sum()),
            "passed": passed,
            **_flatten_bound("calibration", bound),
        }
        audit.append(row)
        if not passed:
            break
        selected = row
    if not testing_started:
        return None, audit
    return selected, audit


def _select_ttl_policy(
    calibration: pd.DataFrame,
    minimum_actions: int,
    target_action: float,
    target_resource: float,
    family_confidence: float,
    exposure_column: str,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    audit: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for index, ttl in enumerate(range(int(calibration["cold_age"].max()), 0, -1)):
        actions = first_actions_by_ttl(calibration, ttl)
        if len(actions) < minimum_actions:
            continue
        bound = dual_clustered_risk_bound(
            actions, family_confidence, exposure_column=exposure_column
        )
        passed = bound.action.upper <= target_action and bound.resource.upper <= target_resource
        row = {
            "candidate_index": index,
            "parameter": float(ttl),
            "actions": len(actions),
            "trace_exposure_gib_hours": float(actions[exposure_column].sum()),
            "passed": passed,
            **_flatten_bound("calibration", bound),
        }
        audit.append(row)
        if not passed:
            break
        selected = row
    return selected, audit


def _evaluate_selected(
    test: pd.DataFrame,
    method: str,
    score_column: str | None,
    selected: dict[str, object] | None,
    target_action: float,
    target_resource: float,
    family_confidence: float,
    exposure_column: str,
) -> dict[str, object]:
    if selected is None:
        return {
            "calibration_qualified": False,
            "test_route": "ABSTAIN_NO_CALIBRATION_SUPPORT",
        }
    parameter = float(selected["parameter"])
    actions = (
        first_actions_by_ttl(test, int(round(parameter)))
        if method == "ttl"
        else first_actions_by_score(test, str(score_column), parameter)
    )
    bound = dual_clustered_risk_bound(
        actions, family_confidence, exposure_column=exposure_column
    )
    passed = bound.action.upper <= target_action and bound.resource.upper <= target_resource
    return {
        "calibration_qualified": True,
        "selected_parameter": parameter,
        "calibration_actions": int(selected["actions"]),
        "calibration_trace_exposure_gib_hours": float(selected["trace_exposure_gib_hours"]),
        **{
            key: value
            for key, value in selected.items()
            if str(key).startswith("calibration_")
        },
        "test_actions": int(len(actions)),
        "test_false_actions": int(actions["reactivates_within_horizon"].sum()),
        "test_action_days": int(actions["day"].nunique()),
        "test_trace_exposure_gib_hours": float(actions[exposure_column].sum()),
        **_flatten_bound("test", bound),
        "test_dual_pass": passed,
        "test_route": "FORWARD_SUPPORTED" if passed else "EXPIRED_RECALIBRATE",
    }


def _reliability_rows(
    frame: pd.DataFrame,
    definition: str,
    split: str,
    horizon: int,
    method: str,
    score_column: str,
    bins: int = 10,
) -> list[dict[str, object]]:
    values = frame[[score_column, "reactivates_within_horizon"]].dropna().copy()
    if values.empty:
        return []
    values["bin"] = pd.qcut(values[score_column], q=bins, duplicates="drop")
    rows = []
    for index, (_, group) in enumerate(values.groupby("bin", observed=True, sort=True)):
        rows.append(
            {
                "cold_definition": definition,
                "split": split,
                "horizon": horizon,
                "method": method,
                "bin": index + 1,
                "count": len(group),
                "mean_score": float(group[score_column].mean()),
                "event_rate": float(group["reactivates_within_horizon"].mean()),
            }
        )
    return rows


def _load_operational_frames(config_path: Path, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    config, project_root = load_config(config_path)
    cfg = config["multihorizon"]
    train_end = int(cfg["train_end_day"])
    calibration_end = int(cfg["calibration_end_day"])
    test_end = int(cfg["test_end_day"])
    landmarks_path = resolve_project_path(project_root, config["paths"]["landmarks_all"])
    landmarks = pd.read_parquet(landmarks_path).sort_values(["episode_id", "day"]).copy()
    exposure = np.zeros(len(landmarks), dtype=float)
    episode_groups = landmarks.groupby("episode_id", observed=True, sort=False)["mem_gib_hours"]
    for offset in range(horizon):
        exposure += episode_groups.shift(-offset).fillna(0.0).to_numpy(float)
    landmarks["horizon_mem_gib_hours"] = exposure
    calibration = attach_horizon_label(
        landmarks.loc[
            (landmarks["day"] > train_end)
            & (landmarks["day"] <= calibration_end - horizon)
        ],
        horizon,
    )
    test = attach_horizon_label(
        landmarks.loc[
            (landmarks["day"] > calibration_end)
            & (landmarks["day"] <= test_end - horizon)
        ],
        horizon,
    )
    models = resolve_project_path(project_root, config["paths"]["multihorizon_artifacts_dir"]) / "models"
    hazard = joblib.load(models / "hazard_hierarchical.joblib")
    hazard_age = joblib.load(models / "hazard_age.joblib")
    interval = joblib.load(models / "next_interval_hierarchical.joblib")
    direct_horizons = [value for value in [1, 3, 7] if value <= horizon]

    for frame in [calibration, test]:
        frame["score_hazard_hierarchical"] = predict_cumulative_incidence(
            frame, hazard["encoder"], hazard["model"], horizon
        )
        frame["score_hazard_age"] = predict_cumulative_incidence(
            frame, hazard_age["encoder"], hazard_age["model"], horizon
        )
        frame["score_next_interval_hierarchical"] = predict_next_interval_score(
            frame, interval["encoder"], interval["model"], horizon
        )
        direct_scores = []
        for direct_horizon in direct_horizons:
            direct = joblib.load(models / f"direct_hierarchical_{direct_horizon}d.joblib")
            scores = direct["model"].predict_proba(direct["encoder"].transform(frame))[:, 1]
            if direct_horizon == horizon:
                frame["score_direct_hierarchical"] = scores
            direct_scores.append(scores)
        frame["score_direct_monotone"] = np.maximum.reduce(direct_scores)
        frame["score_recency_frequency"] = (
            frame["prior_active_days_30"].fillna(0.0) + 1.0
        ) / (
            frame["prior_active_days_30"].fillna(0.0)
            + frame["cold_age"].clip(lower=1.0)
            + 2.0
        )
    return calibration, test


def run(config_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for config_path in config_paths:
        config, _ = load_config(config_path)
        definition = f"sm_le_{config['cold_definition']['sm_util_max']}_pct"
        risk_cfg = config["risk"]
        grid_size = int(risk_cfg["threshold_grid_size"])
        minimum_actions = int(risk_cfg["minimum_selected"])
        family_confidence = float(risk_cfg["confidence"])
        action_target = float(config["multihorizon"]["target_action_risk"])
        resource_target = float(config["multihorizon"]["target_resource_risk"])
        methods = {
            "hazard_hierarchical": "score_hazard_hierarchical",
            "hazard_age": "score_hazard_age",
            "direct_hierarchical": "score_direct_hierarchical",
            "direct_monotone": "score_direct_monotone",
            "next_interval_hierarchical": "score_next_interval_hierarchical",
            "recency_frequency": "score_recency_frequency",
        }
        for horizon in [1, 3, 7]:
            calibration, test = _load_operational_frames(config_path, horizon)
            for method, score_column in {"ttl": None, **methods}.items():
                if method == "ttl":
                    selected, audit = _select_ttl_policy(
                        calibration,
                        minimum_actions,
                        action_target,
                        resource_target,
                        family_confidence,
                        "horizon_mem_gib_hours",
                    )
                else:
                    selected, audit = _select_score_policy(
                        calibration,
                        str(score_column),
                        grid_size,
                        minimum_actions,
                        action_target,
                        resource_target,
                        family_confidence,
                        "horizon_mem_gib_hours",
                    )
                for row in audit:
                    candidate_rows.append(
                        {
                            "cold_definition": definition,
                            "horizon": horizon,
                            "method": method,
                            **row,
                        }
                    )
                policy_rows.append(
                    {
                        "cold_definition": definition,
                        "horizon": horizon,
                        "method": method,
                        "calibration_start_day": int(calibration["day"].min()),
                        "calibration_end_day": int(calibration["day"].max()),
                        "test_start_day": int(test["day"].min()),
                        "test_end_day": int(test["day"].max()),
                        "calibration_risk_set_rows": len(calibration),
                        "test_risk_set_rows": len(test),
                        **_evaluate_selected(
                            test,
                            method,
                            score_column,
                            selected,
                            action_target,
                            resource_target,
                            family_confidence,
                            "horizon_mem_gib_hours",
                        ),
                    }
                )
                if score_column is None:
                    continue
                for split, frame in [("calibration", calibration), ("test", test)]:
                    labels = frame["reactivates_within_horizon"].to_numpy(int)
                    scores = frame[str(score_column)].to_numpy(float)
                    metric_rows.append(
                        {
                            "cold_definition": definition,
                            "split": split,
                            "horizon": horizon,
                            "method": method,
                            "rows": len(frame),
                            "events": int(labels.sum()),
                            "brier": float(brier_score_loss(labels, scores)),
                            "roc_auc": float(roc_auc_score(labels, scores)),
                        }
                    )
                    reliability_rows.extend(
                        _reliability_rows(
                            frame,
                            definition,
                            split,
                            horizon,
                            method,
                            str(score_column),
                        )
                    )

    policies = pd.DataFrame(policy_rows)
    candidates = pd.DataFrame(candidate_rows)
    reliability = pd.DataFrame(reliability_rows)
    metrics = pd.DataFrame(metric_rows)
    policies.to_csv(output_dir / "operational_policy_audit.csv", index=False)
    candidates.to_csv(output_dir / "candidate_sequence_audit.csv", index=False)
    reliability.to_csv(output_dir / "reliability_bins.csv", index=False)
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    summary = {
        "method": "pooled-ratio two-axis cluster stress bound",
        "selection": "predeclared conservative-to-permissive fixed sequence",
        "joint_control": "Bonferroni split over action and resource risk",
        "scope": "empirical dependence-stressed qualification under chronological replay",
        "policies": len(policies),
        "calibration_qualified": int(policies["calibration_qualified"].sum()),
        "forward_supported": int((policies["test_route"] == "FORWARD_SUPPORTED").sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["config/pilot.yaml", "config/pilot_strict.yaml"],
    )
    parser.add_argument("--output-dir", default="artifacts/revision_round1")
    args = parser.parse_args()
    run(
        [ROOT / value for value in args.configs],
        ROOT / args.output_dir,
    )


if __name__ == "__main__":
    main()
