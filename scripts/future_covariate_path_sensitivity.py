from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from coldguard.clustered_risk import DualRiskBound, dual_clustered_risk_bound
from coldguard.config import load_config, resolve_project_path
from coldguard.multihorizon import attach_horizon_label, horizon_is_observed
from coldguard.partial_identification import audit_proposal_stream
from coldguard.policy import first_actions_by_score
from coldguard.survival import (
    _deployment_groups,
    predict_cumulative_incidence_path,
)


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["app_id", "model_id", "site_id", "pd_role"]
PATHS = ["frozen", "rolling_exact", "fast_decay", "robust_max"]


def _horizon_exposure(frame: pd.DataFrame, horizon: int) -> pd.Series:
    ordered = frame.sort_values(["episode_id", "day"]).copy()
    exposure = np.zeros(len(ordered), dtype=float)
    grouped = ordered.groupby("episode_id", observed=True, sort=False)["mem_gib_hours"]
    for offset in range(horizon):
        exposure += grouped.shift(-offset).fillna(0.0).to_numpy(float)
    return pd.Series(exposure, index=ordered.index).reindex(frame.index)


def _active_lookup(daily: pd.DataFrame) -> pd.DataFrame:
    active = daily.loc[~daily["is_cold"].astype(bool), KEYS + ["day"]].copy()
    active["was_active"] = 1.0
    if active.duplicated(KEYS + ["day"]).any():
        raise ValueError("deployment_daily is not unique by logical deployment and day")
    return active


def _prior_active_paths(
    frame: pd.DataFrame,
    active_lookup: pd.DataFrame,
    horizon: int,
) -> dict[str, np.ndarray]:
    current = (
        pd.to_numeric(frame["prior_active_days_30"], errors="raise")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(float)
    )
    frozen = np.repeat(current[:, None], horizon, axis=1)
    exact = frozen.copy()
    fast = frozen.copy()
    for offset in range(1, horizon):
        leaving = frame[KEYS].copy()
        leaving["day"] = pd.to_numeric(frame["day"], errors="raise").to_numpy(int) + offset - 30
        indicator = (
            leaving.merge(
                active_lookup,
                on=KEYS + ["day"],
                how="left",
                validate="many_to_one",
                sort=False,
            )["was_active"]
            .fillna(0.0)
            .to_numpy(float)
        )
        exact[:, offset] = np.maximum(0.0, exact[:, offset - 1] - indicator)
        fast[:, offset] = np.maximum(0.0, current - offset)
    if not (
        (fast <= exact + 1e-12).all()
        and (exact <= frozen + 1e-12).all()
    ):
        raise AssertionError("rolling active-day path escaped its feature envelope")
    return {"frozen": frozen, "rolling_exact": exact, "fast_decay": fast}


def _path_scores(
    frame: pd.DataFrame,
    active_lookup: pd.DataFrame,
    horizon: int,
    bundle: dict[str, object],
) -> dict[str, np.ndarray]:
    paths = _prior_active_paths(frame, active_lookup, horizon)
    scores = {
        name: predict_cumulative_incidence_path(
            frame,
            bundle["encoder"],
            bundle["model"],
            horizon,
            values,
        )
        for name, values in paths.items()
    }
    scores["robust_max"] = np.maximum.reduce(list(scores.values()))
    return scores


def _flatten_bound(prefix: str, bound: DualRiskBound) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for risk_name, risk in [("action", bound.action), ("resource", bound.resource)]:
        for key, value in asdict(risk).items():
            row[f"{prefix}_{risk_name}_{key}"] = value
    return row


def _candidate_scores(frame: pd.DataFrame, column: str, grid_size: int) -> np.ndarray:
    quantiles = np.linspace(0.01, 0.99, max(3, grid_size))
    return np.unique(np.quantile(frame[column].dropna().to_numpy(float), quantiles))


def _select_policy(
    calibration: pd.DataFrame,
    column: str,
    *,
    grid_size: int,
    minimum_actions: int,
    family_confidence: float,
    action_target: float,
    resource_target: float,
) -> dict[str, object] | None:
    selected: dict[str, object] | None = None
    for index, threshold in enumerate(_candidate_scores(calibration, column, grid_size)):
        actions = first_actions_by_score(calibration, column, float(threshold))
        if len(actions) < minimum_actions:
            continue
        bound = dual_clustered_risk_bound(
            actions,
            family_confidence,
            exposure_column="horizon_mem_gib_hours",
        )
        passed = bound.action.upper <= action_target and bound.resource.upper <= resource_target
        if not passed:
            break
        selected = {
            "candidate_index": index,
            "selected_parameter": float(threshold),
            "calibration_actions": int(len(actions)),
            **_flatten_bound("calibration", bound),
        }
    return selected


def _fit_logistic_calibration(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    x = np.column_stack([np.ones(len(p)), np.log(p / (1.0 - p))])
    w = np.ones(len(p), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if np.unique(y[w > 0]).size < 2:
        raise ValueError("calibration fit requires both outcome classes")
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(60):
        fitted = expit(x @ beta)
        score = x.T @ (w * (y - fitted))
        information = x.T @ ((w * fitted * (1.0 - fitted))[:, None] * x)
        step = np.linalg.solve(information + np.eye(2) * 1e-8, score)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def _cluster_bootstrap_calibration(
    frame: pd.DataFrame,
    score_column: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    labels = frame["reactivates_within_horizon"].to_numpy(int)
    scores = frame[score_column].to_numpy(float)
    cluster_codes, cluster_values = pd.factorize(_deployment_groups(frame), sort=True)
    cluster_count = len(cluster_values)
    intercept, slope = _fit_logistic_calibration(labels, scores)
    rng = np.random.default_rng(seed)
    boot: list[tuple[float, float]] = []
    for _ in range(repetitions):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        counts = np.bincount(sampled, minlength=cluster_count).astype(float)
        weights = counts[cluster_codes]
        try:
            boot.append(_fit_logistic_calibration(labels, scores, weights))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    values = np.asarray(boot, dtype=float)
    if len(values) < max(50, int(0.8 * repetitions)):
        raise RuntimeError(f"only {len(values)}/{repetitions} calibration bootstraps converged")
    return {
        "clusters": cluster_count,
        "bootstrap_repetitions_requested": repetitions,
        "bootstrap_repetitions_valid": len(values),
        "calibration_intercept": intercept,
        "calibration_intercept_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "calibration_intercept_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "calibration_slope": slope,
        "calibration_slope_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "calibration_slope_ci_high": float(np.quantile(values[:, 1], 0.975)),
    }


def _prepare_frames(config_path: Path, horizon: int):
    config, project_root = load_config(config_path)
    cfg = config["multihorizon"]
    train_end = int(cfg["train_end_day"])
    calibration_end = int(cfg["calibration_end_day"])
    test_end = int(cfg["test_end_day"])
    landmarks = pd.read_parquet(
        resolve_project_path(project_root, config["paths"]["landmarks_all"])
    ).sort_values(["episode_id", "day"])
    landmarks["horizon_mem_gib_hours"] = _horizon_exposure(landmarks, horizon)
    calibration = attach_horizon_label(
        landmarks.loc[
            (landmarks["day"] > train_end)
            & (landmarks["day"] <= calibration_end - horizon)
        ],
        horizon,
    )
    full_test = landmarks.loc[
        (landmarks["day"] > calibration_end)
        & (landmarks["day"] <= test_end - horizon)
    ].copy()
    full_test["outcome_observed"] = horizon_is_observed(full_test, horizon)
    event_day = pd.to_numeric(full_test["event_day"], errors="coerce")
    full_test["observed_label"] = (
        event_day.notna() & (event_day <= full_test["day"] + horizon)
    ).astype(int)
    persisted = (
        full_test["avg_mem_gib_calendar"].fillna(0.0).clip(lower=0.0)
        * 24.0
        * horizon
    )
    full_test["persistent_horizon_mem_gib_hours"] = np.maximum(
        full_test["horizon_mem_gib_hours"].to_numpy(float),
        persisted.to_numpy(float),
    )
    test = full_test.loc[full_test["outcome_observed"]].copy()
    test["reactivates_within_horizon"] = test["observed_label"].to_numpy(int)
    daily = pd.read_parquet(
        resolve_project_path(project_root, config["paths"]["deployment_daily"]),
        columns=KEYS + ["day", "is_cold"],
    )
    model_dir = (
        resolve_project_path(project_root, config["paths"]["multihorizon_artifacts_dir"])
        / "models"
    )
    bundle = joblib.load(model_dir / "hazard_hierarchical.joblib")
    return config, calibration, test, full_test, _active_lookup(daily), bundle


def run(
    config_paths: list[Path],
    output_dir: Path,
    *,
    bootstrap_repetitions: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []

    for config_index, config_path in enumerate(config_paths):
        for horizon in [1, 3, 7]:
            config, calibration, test, full_test, active, bundle = _prepare_frames(
                config_path, horizon
            )
            definition = f"sm_le_{config['cold_definition']['sm_util_max']}_pct"
            risk = config["risk"]
            mh = config["multihorizon"]
            family_confidence = float(risk["confidence"])
            action_target = float(mh["target_action_risk"])
            resource_target = float(mh["target_resource_risk"])
            scored: dict[str, dict[str, np.ndarray]] = {}
            for split_name, frame in [
                ("calibration", calibration),
                ("test", test),
                ("full_test", full_test),
            ]:
                scored[split_name] = _path_scores(frame, active, horizon, bundle)
                for path_name, values in scored[split_name].items():
                    frame[f"score_{path_name}"] = values

            for split_name, frame in [("calibration", calibration), ("test", test)]:
                labels = frame["reactivates_within_horizon"].to_numpy(int)
                for path_index, path_name in enumerate(PATHS):
                    scores = frame[f"score_{path_name}"].to_numpy(float)
                    metric_rows.append(
                        {
                            "cold_definition": definition,
                            "horizon": horizon,
                            "split": split_name,
                            "path": path_name,
                            "rows": len(frame),
                            "events": int(labels.sum()),
                            "brier": float(brier_score_loss(labels, scores)),
                            "average_precision": float(average_precision_score(labels, scores)),
                            "roc_auc": float(roc_auc_score(labels, scores)),
                        }
                    )
                    if split_name == "test":
                        calibration_rows.append(
                            {
                                "cold_definition": definition,
                                "horizon": horizon,
                                "split": split_name,
                                "path": path_name,
                                **_cluster_bootstrap_calibration(
                                    frame,
                                    f"score_{path_name}",
                                    repetitions=bootstrap_repetitions,
                                    seed=(
                                        int(config["model"]["random_seed"])
                                        + 1000 * config_index
                                        + 100 * horizon
                                        + path_index
                                    ),
                                ),
                            }
                        )

            for split_name, frame in [("calibration", calibration), ("test", test)]:
                frozen = frame["score_frozen"].to_numpy(float)
                for path_name in ["rolling_exact", "fast_decay", "robust_max"]:
                    values = frame[f"score_{path_name}"].to_numpy(float)
                    difference = values - frozen
                    delta_rows.append(
                        {
                            "cold_definition": definition,
                            "horizon": horizon,
                            "split": split_name,
                            "path": path_name,
                            "mean_delta": float(difference.mean()),
                            "mean_absolute_delta": float(np.abs(difference).mean()),
                            "p95_absolute_delta": float(np.quantile(np.abs(difference), 0.95)),
                            "max_absolute_delta": float(np.abs(difference).max(initial=0.0)),
                            "fraction_changed": float(np.mean(np.abs(difference) > 1e-12)),
                            "fraction_higher_risk": float(np.mean(difference > 1e-12)),
                        }
                    )

            for path_name in PATHS:
                selected = _select_policy(
                    calibration,
                    f"score_{path_name}",
                    grid_size=int(risk["threshold_grid_size"]),
                    minimum_actions=int(risk["minimum_selected"]),
                    family_confidence=family_confidence,
                    action_target=action_target,
                    resource_target=resource_target,
                )
                base = {
                    "cold_definition": definition,
                    "horizon": horizon,
                    "path": path_name,
                }
                if selected is None:
                    policy_rows.append(
                        {
                            **base,
                            "calibration_qualified": False,
                            "complete_case_route": "ABSTAIN_NO_CALIBRATION_SUPPORT",
                            "full_stream_route": "ABSTAIN_NO_CALIBRATION_SUPPORT",
                        }
                    )
                    continue
                threshold = float(selected["selected_parameter"])
                test_actions = first_actions_by_score(
                    test, f"score_{path_name}", threshold
                )
                test_bound = dual_clustered_risk_bound(
                    test_actions,
                    family_confidence,
                    exposure_column="horizon_mem_gib_hours",
                )
                complete_pass = (
                    test_bound.action.upper <= action_target
                    and test_bound.resource.upper <= resource_target
                )
                proposal_actions = first_actions_by_score(
                    full_test, f"score_{path_name}", threshold
                )
                stream = audit_proposal_stream(
                    proposal_actions,
                    family_confidence=family_confidence,
                    action_target=action_target,
                    resource_target=resource_target,
                )
                policy_rows.append(
                    {
                        **base,
                        "calibration_qualified": True,
                        **selected,
                        "test_actions": int(len(test_actions)),
                        "test_false_actions": int(
                            test_actions["reactivates_within_horizon"].sum()
                        ),
                        **_flatten_bound("test", test_bound),
                        "complete_case_route": (
                            "FORWARD_SUPPORTED" if complete_pass else "EXPIRED_RECALIBRATE"
                        ),
                        "proposed_actions": stream.proposed_actions,
                        "observed_actions": stream.observed_actions,
                        "unresolved_actions": stream.unresolved_actions,
                        **_flatten_bound("full_lower", stream.lower),
                        **_flatten_bound("full_upper", stream.upper),
                        "full_stream_route": stream.route,
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["cold_definition", "horizon", "split", "path"]
    )
    coefficients = pd.DataFrame(calibration_rows).sort_values(
        ["cold_definition", "horizon", "path"]
    )
    deltas = pd.DataFrame(delta_rows).sort_values(
        ["cold_definition", "horizon", "split", "path"]
    )
    policies = pd.DataFrame(policy_rows).sort_values(
        ["cold_definition", "horizon", "path"]
    )
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    coefficients.to_csv(output_dir / "calibration_coefficients.csv", index=False)
    deltas.to_csv(output_dir / "score_path_deltas.csv", index=False)
    policies.to_csv(output_dir / "policy_path_audit.csv", index=False)

    qualified = policies.loc[policies["calibration_qualified"].astype(bool)]
    summary = {
        "analysis": "future-covariate path sensitivity for discrete hazards",
        "paths": {
            "frozen": "clocks advance; prior_active_days_30 and other measurements use LOCF",
            "rolling_exact": "prior_active_days_30 mechanically rolls using observed pre-landmark active days under continued inactivity",
            "fast_decay": "feature-path lower envelope max(0, current_count-offset)",
            "robust_max": "row-wise maximum predicted risk over frozen, rolling_exact, and fast_decay paths",
        },
        "bootstrap_unit": "logical deployment",
        "bootstrap_repetitions": bootstrap_repetitions,
        "policy_rows": int(len(policies)),
        "calibration_qualified": int(len(qualified)),
        "complete_case_route_counts": {
            str(k): int(v) for k, v in policies["complete_case_route"].value_counts().items()
        },
        "full_stream_route_counts": {
            str(k): int(v) for k, v in policies["full_stream_route"].value_counts().items()
        },
        "full_stream_forward_supported": int(
            (policies["full_stream_route"] == "FORWARD_SUPPORTED_FULL_BOUND").sum()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["config/pilot.yaml", "config/pilot_strict.yaml"],
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/revision_round2/future_covariate_path",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    args = parser.parse_args()
    run(
        [ROOT / value for value in args.configs],
        ROOT / args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )


if __name__ == "__main__":
    main()
