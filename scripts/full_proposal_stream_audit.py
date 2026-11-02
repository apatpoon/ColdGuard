from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from coldguard.config import load_config, resolve_project_path
from coldguard.multihorizon import horizon_is_observed
from coldguard.partial_identification import audit_proposal_stream, tipping_curve
from coldguard.policy import first_actions_by_score, first_actions_by_ttl
from coldguard.survival import predict_cumulative_incidence, predict_next_interval_score


ROOT = Path(__file__).resolve().parents[1]


def _horizon_exposure(frame: pd.DataFrame, horizon: int) -> pd.Series:
    ordered = frame.sort_values(["episode_id", "day"]).copy()
    exposure = np.zeros(len(ordered), dtype=float)
    grouped = ordered.groupby("episode_id", observed=True, sort=False)["mem_gib_hours"]
    for offset in range(horizon):
        exposure += grouped.shift(-offset).fillna(0.0).to_numpy(float)
    return pd.Series(exposure, index=ordered.index).reindex(frame.index)


def _flatten(prefix: str, bound) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for risk_name, value in [("action", bound.action), ("resource", bound.resource)]:
        for key, item in asdict(value).items():
            row[f"{prefix}_{risk_name}_{key}"] = item
    return row


def _score_full_test_frame(config_path: Path, horizon: int) -> pd.DataFrame:
    config, project_root = load_config(config_path)
    cfg = config["multihorizon"]
    calibration_end = int(cfg["calibration_end_day"])
    test_end = int(cfg["test_end_day"])
    landmarks = pd.read_parquet(
        resolve_project_path(project_root, config["paths"]["landmarks_all"])
    ).sort_values(["episode_id", "day"])
    landmarks["horizon_mem_gib_hours"] = _horizon_exposure(landmarks, horizon)
    # Preserve the frozen chronological evaluation boundary. The final h days
    # are outside the eligible proposal period because their outcomes would
    # require evidence beyond the predeclared test window. Unresolved actions
    # below are therefore due to within-window episode disappearance, not an
    # extension of the action period into the horizon-censored tail.
    frame = landmarks.loc[
        (landmarks["day"] > calibration_end)
        & (landmarks["day"] <= test_end - horizon)
    ].copy()
    frame["outcome_observed"] = horizon_is_observed(frame, horizon)
    event_day = pd.to_numeric(frame["event_day"], errors="coerce")
    frame["observed_label"] = (
        event_day.notna() & (event_day <= frame["day"] + horizon)
    ).astype(int)
    persisted = (
        frame["avg_mem_gib_calendar"].fillna(0.0).clip(lower=0.0)
        * 24.0
        * horizon
    )
    frame["persistent_horizon_mem_gib_hours"] = np.maximum(
        frame["horizon_mem_gib_hours"].to_numpy(float),
        persisted.to_numpy(float),
    )

    model_dir = (
        resolve_project_path(project_root, config["paths"]["multihorizon_artifacts_dir"])
        / "models"
    )
    hazard = joblib.load(model_dir / "hazard_hierarchical.joblib")
    hazard_age = joblib.load(model_dir / "hazard_age.joblib")
    interval = joblib.load(model_dir / "next_interval_hierarchical.joblib")
    frame["score_hazard_hierarchical"] = predict_cumulative_incidence(
        frame, hazard["encoder"], hazard["model"], horizon
    )
    frame["score_hazard_age"] = predict_cumulative_incidence(
        frame, hazard_age["encoder"], hazard_age["model"], horizon
    )
    frame["score_next_interval_hierarchical"] = predict_next_interval_score(
        frame, interval["encoder"], interval["model"], horizon
    )
    direct_scores: list[np.ndarray] = []
    for direct_horizon in [value for value in [1, 3, 7] if value <= horizon]:
        direct = joblib.load(
            model_dir / f"direct_hierarchical_{direct_horizon}d.joblib"
        )
        score = direct["model"].predict_proba(direct["encoder"].transform(frame))[:, 1]
        direct_scores.append(score)
        if direct_horizon == horizon:
            frame["score_direct_hierarchical"] = score
    frame["score_direct_monotone"] = np.maximum.reduce(direct_scores)
    frame["score_recency_frequency"] = (
        frame["prior_active_days_30"].fillna(0.0) + 1.0
    ) / (
        frame["prior_active_days_30"].fillna(0.0)
        + frame["cold_age"].clip(lower=1.0)
        + 2.0
    )
    return frame


def _actions(frame: pd.DataFrame, method: str, parameter: float) -> pd.DataFrame:
    if method == "ttl":
        return first_actions_by_ttl(frame, int(round(parameter)))
    return first_actions_by_score(frame, f"score_{method}", parameter)


def _first_crossing(curve: pd.DataFrame, column: str, target: float) -> float:
    crossed = curve.loc[curve[column] > target, "unresolved_revival_fraction"]
    return float(crossed.iloc[0]) if not crossed.empty else float("nan")


def run(config_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = pd.read_csv(ROOT / "artifacts/revision_round1/operational_policy_audit.csv")
    fraction_grid = np.linspace(0.0, 1.0, 101)
    policy_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    action_frames: list[pd.DataFrame] = []

    for config_path in config_paths:
        config, _ = load_config(config_path)
        definition = f"sm_le_{config['cold_definition']['sm_util_max']}_pct"
        family_confidence = float(config["risk"]["confidence"])
        action_target = float(config["multihorizon"]["target_action_risk"])
        resource_target = float(config["multihorizon"]["target_resource_risk"])
        for horizon in [1, 3, 7]:
            frame = _score_full_test_frame(config_path, horizon)
            selected_rows = policy.loc[
                (policy["cold_definition"] == definition)
                & (policy["horizon"] == horizon)
                & policy["calibration_qualified"].astype(bool)
            ]
            for selected in selected_rows.itertuples(index=False):
                method = str(selected.method)
                parameter = float(selected.selected_parameter)
                actions = _actions(frame, method, parameter).copy()
                audit = audit_proposal_stream(
                    actions,
                    family_confidence=family_confidence,
                    action_target=action_target,
                    resource_target=resource_target,
                )
                observed = actions["outcome_observed"].astype(bool)
                observed_count_matches = audit.observed_actions == int(selected.test_actions)
                curve = tipping_curve(
                    actions,
                    fraction_grid,
                    family_confidence=family_confidence,
                    action_target=action_target,
                    resource_target=resource_target,
                )
                dual_crossed = curve.loc[
                    (curve["action_upper"] > action_target)
                    | (curve["resource_upper"] > resource_target),
                    "unresolved_revival_fraction",
                ]
                observed_events = int(actions.loc[observed, "observed_label"].sum())
                unresolved = audit.unresolved_actions
                action_point_tipping = (
                    float(np.clip((action_target * len(actions) - observed_events) / unresolved, 0.0, 1.0))
                    if unresolved > 0
                    else float("nan")
                )
                policy_rows.append(
                    {
                        "cold_definition": definition,
                        "horizon": horizon,
                        "method": method,
                        "selected_parameter": parameter,
                        "original_complete_case_route": str(selected.test_route),
                        "complete_case_actions": int(selected.test_actions),
                        "action_target": action_target,
                        "resource_target": resource_target,
                        "proposed_actions": audit.proposed_actions,
                        "observed_actions": audit.observed_actions,
                        "unresolved_actions": audit.unresolved_actions,
                        "unresolved_share": (
                            audit.unresolved_actions / audit.proposed_actions
                            if audit.proposed_actions
                            else 0.0
                        ),
                        "observed_events": observed_events,
                        "observed_count_matches_complete_case": observed_count_matches,
                        "full_stream_route": audit.route,
                        "action_point_tipping_fraction": action_point_tipping,
                        "action_bound_first_crossing_fraction": _first_crossing(
                            curve, "action_upper", action_target
                        ),
                        "resource_bound_first_crossing_fraction": _first_crossing(
                            curve, "resource_upper", resource_target
                        ),
                        "dual_bound_first_crossing_fraction": (
                            float(dual_crossed.iloc[0])
                            if not dual_crossed.empty
                            else float("nan")
                        ),
                        **_flatten("lower", audit.lower),
                        **_flatten("upper", audit.upper),
                    }
                )
                curve.insert(0, "method", method)
                curve.insert(0, "horizon", horizon)
                curve.insert(0, "cold_definition", definition)
                curve_rows.extend(curve.to_dict("records"))

                keep = actions.loc[:, [
                    "day", "episode_id", "app_id", "model_id", "site_id", "pd_role",
                    "outcome_observed", "observed_label", "horizon_mem_gib_hours",
                    "persistent_horizon_mem_gib_hours",
                ]].copy()
                keep.insert(0, "method", method)
                keep.insert(0, "horizon", horizon)
                keep.insert(0, "cold_definition", definition)
                action_frames.append(keep)

    policies = pd.DataFrame(policy_rows).sort_values(
        ["cold_definition", "horizon", "method"]
    )
    curves = pd.DataFrame(curve_rows).sort_values(
        ["cold_definition", "horizon", "method", "unresolved_revival_fraction"]
    )
    actions = pd.concat(action_frames, ignore_index=True)
    policies.to_csv(output_dir / "policy_bounds.csv", index=False)
    curves.to_csv(output_dir / "tipping_curves.csv", index=False)
    actions.to_parquet(output_dir / "selected_actions.parquet", index=False)

    changed = policies.loc[
        policies["original_complete_case_route"] == "FORWARD_SUPPORTED"
    ]
    summary = {
        "analysis": "full proposal stream extreme bounds and coupled tipping sensitivity",
        "fraction_grid_step": 0.01,
        "policies_audited": int(len(policies)),
        "originally_forward_supported": int(len(changed)),
        "full_bound_forward_supported": int(
            (changed["full_stream_route"] == "FORWARD_SUPPORTED_FULL_BOUND").sum()
        ),
        "shadow_only_unresolved_bound": int(
            (changed["full_stream_route"] == "SHADOW_ONLY_UNRESOLVED_BOUND").sum()
        ),
        "expired_recalibrate": int(
            (changed["full_stream_route"] == "EXPIRED_RECALIBRATE").sum()
        ),
        "route_counts_all_calibration_qualified": {
            str(key): int(value)
            for key, value in policies["full_stream_route"].value_counts().items()
        },
        "interpretation": (
            "Endpoints are extreme-scenario sensitivity bounds. Intermediate q values "
            "form a coupled fractional sensitivity path, not an identified probability model."
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
        default="artifacts/revision_round2/full_proposal_stream",
    )
    args = parser.parse_args()
    run([ROOT / value for value in args.configs], ROOT / args.output_dir)


if __name__ == "__main__":
    main()

