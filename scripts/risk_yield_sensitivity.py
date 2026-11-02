from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from coldguard.config import load_config, resolve_project_path
from coldguard.multihorizon import attach_horizon_label
from coldguard.operational import _load_frame, operational_conditions
from coldguard.policy import (
    action_metrics,
    choose_policy,
    first_actions_by_score,
    score_frontier,
)
from coldguard.survival import predict_cumulative_incidence


def evaluate_config(config_path: Path, cold_definition: str) -> pd.DataFrame:
    config, project_root = load_config(config_path)
    landmarks = resolve_project_path(project_root, config["paths"]["landmarks_all"])
    trained = resolve_project_path(
        project_root, config["paths"]["multihorizon_artifacts_dir"]
    )
    bundle = joblib.load(trained / "models" / "hazard_hierarchical.joblib")
    encoder, model = bundle["encoder"], bundle["model"]

    cfg = config["multihorizon"]
    train_end = int(cfg["train_end_day"])
    calibration_end = int(cfg["calibration_end_day"])
    test_end = int(cfg["test_end_day"])
    confidence = float(config["risk"]["confidence"])
    grid_size = int(config["risk"]["threshold_grid_size"])
    minimum_selected = int(config["risk"]["minimum_selected"])
    budgets = [0.075, 0.10, 0.125, 0.15, 0.20]
    rows: list[dict[str, object]] = []

    for horizon in [1, 3, 7]:
        conditions = operational_conditions(
            train_end, calibration_end, test_end, horizon
        )
        calibration = attach_horizon_label(
            _load_frame(landmarks, conditions["calibration"]), horizon
        )
        test = attach_horizon_label(
            _load_frame(landmarks, conditions["test"]), horizon
        )
        calibration["score"] = predict_cumulative_incidence(
            calibration, encoder, model, horizon
        )
        test["score"] = predict_cumulative_incidence(test, encoder, model, horizon)
        frontier = score_frontier(calibration, "score", grid_size, confidence)

        for budget in budgets:
            choice = choose_policy(
                frontier,
                "hazard_hierarchical",
                target_risk=budget,
                minimum_selected=minimum_selected,
                target_action_risk=budget,
            )
            base = {
                "cold_definition": cold_definition,
                "horizon_days": horizon,
                "risk_budget": budget,
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "minimum_actions": minimum_selected,
            }
            if choice is None:
                rows.append({**base, "calibration_eligible": False})
                continue
            actions = first_actions_by_score(test, "score", choice.parameter)
            metrics = action_metrics(actions, confidence=confidence, comparisons=1)
            rows.append(
                {
                    **base,
                    "calibration_eligible": True,
                    "threshold": choice.parameter,
                    "calibration_actions": choice.calibration_actions,
                    "calibration_action_ucb": choice.calibration_unweighted_risk_ucb,
                    "calibration_resource_ucb": choice.calibration_risk_ucb,
                    "calibration_reclaimed_gib_hours": (
                        choice.calibration_reclaimed_gib_hours
                    ),
                    "test_actions": metrics["actions"],
                    "test_action_ucb": metrics["unweighted_risk_ucb"],
                    "test_resource_ucb": metrics["risk_ucb"],
                    "test_reclaimed_gib_hours": metrics["reclaimed_gib_hours"],
                    "test_contract_pass": (
                        float(metrics["unweighted_risk_ucb"]) <= budget
                        and float(metrics["risk_ucb"]) <= budget
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/risk_yield_sensitivity.csv",
        help="Project-relative output CSV.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    frames = [
        evaluate_config(project_root / "config/pilot.yaml", "SM <= 1%"),
        evaluate_config(project_root / "config/pilot_strict.yaml", "SM <= 0.1%"),
    ]
    result = pd.concat(frames, ignore_index=True)
    output = project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
