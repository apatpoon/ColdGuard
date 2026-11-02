from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd

from .config import load_config, require_keys, resolve_project_path
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
class EvaluationFold:
    name: str
    test_start_day: int
    test_end_day: int

    def test_condition(self, horizon: int) -> str:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if self.test_start_day > self.test_end_day:
            raise ValueError(f"Invalid test fold: {self}")
        return (
            f"day >= {self.test_start_day} "
            f"AND day <= {self.test_end_day - horizon}"
        )


@dataclass(frozen=True)
class CalibrationWindow:
    calendar_days: int
    start_day: int
    end_day: int
    mature_end_day: int

    @property
    def mature_calendar_days(self) -> int:
        return max(0, self.mature_end_day - self.start_day + 1)

    def condition(self) -> str:
        return f"day >= {self.start_day} AND day <= {self.mature_end_day}"


def calibration_window(
    fold: EvaluationFold,
    calendar_days: int,
    horizon: int,
    earliest_day: int,
) -> CalibrationWindow | None:
    if calendar_days < 1:
        raise ValueError("calendar_days must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    end_day = fold.test_start_day - 1
    start_day = end_day - calendar_days + 1
    if start_day < earliest_day:
        return None
    return CalibrationWindow(
        calendar_days=calendar_days,
        start_day=start_day,
        end_day=end_day,
        mature_end_day=end_day - horizon,
    )


def run_calibration_window(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(
        config,
        "paths",
        [
            "landmarks_all",
            "multihorizon_artifacts_dir",
            "calibration_window_artifacts_dir",
        ],
    )
    require_keys(
        config,
        "calibration_window",
        ["horizons", "window_days", "earliest_day", "test_folds"],
    )
    require_keys(
        config,
        "multihorizon",
        ["target_resource_risk", "target_action_risk"],
    )
    require_keys(config, "risk", ["confidence", "threshold_grid_size", "minimum_selected"])

    landmarks = resolve_project_path(project_root, config["paths"]["landmarks_all"])
    model_dir = resolve_project_path(
        project_root, config["paths"]["multihorizon_artifacts_dir"]
    ) / "models"
    artifacts = resolve_project_path(
        project_root, config["paths"]["calibration_window_artifacts_dir"]
    )
    artifacts.mkdir(parents=True, exist_ok=True)

    hazard_bundle = joblib.load(model_dir / "hazard_hierarchical.joblib")
    hazard_encoder, hazard_model = hazard_bundle["encoder"], hazard_bundle["model"]
    interval_bundle = joblib.load(model_dir / "next_interval_hierarchical.joblib")
    interval_encoder, interval_model = interval_bundle["encoder"], interval_bundle["model"]

    cfg = config["calibration_window"]
    horizons = sorted(set(int(value) for value in cfg["horizons"]))
    window_days = sorted(set(int(value) for value in cfg["window_days"]))
    earliest_day = int(cfg["earliest_day"])
    folds = [EvaluationFold(**values) for values in cfg["test_folds"]]
    action_target = float(config["multihorizon"]["target_action_risk"])
    resource_target = float(config["multihorizon"]["target_resource_risk"])
    confidence = float(config["risk"]["confidence"])
    grid_size = int(config["risk"]["threshold_grid_size"])
    minimum_selected = int(config["risk"]["minimum_selected"])

    rows: list[dict[str, Any]] = []
    for fold in folds:
        for horizon in horizons:
            test = attach_horizon_label(
                _load_frame(landmarks, fold.test_condition(horizon)), horizon
            )
            if test.empty:
                raise ValueError(f"Empty test frame for {fold.name}, horizon {horizon}")
            test["score_hazard"] = predict_cumulative_incidence(
                test, hazard_encoder, hazard_model, horizon
            )
            test["score_next_interval"] = predict_next_interval_score(
                test, interval_encoder, interval_model, horizon
            )

            for days in window_days:
                window = calibration_window(fold, days, horizon, earliest_day)
                if window is None:
                    continue
                calibration = attach_horizon_label(
                    _load_frame(landmarks, window.condition()), horizon
                )
                if calibration.empty:
                    raise ValueError(
                        f"Empty calibration frame for {fold.name}, horizon {horizon}, window {days}"
                    )
                calibration["score_hazard"] = predict_cumulative_incidence(
                    calibration, hazard_encoder, hazard_model, horizon
                )
                calibration["score_next_interval"] = predict_next_interval_score(
                    calibration, interval_encoder, interval_model, horizon
                )

                frontiers = {
                    "ttl": ttl_frontier(calibration, confidence),
                    "hazard_hierarchical": score_frontier(
                        calibration, "score_hazard", grid_size, confidence
                    ),
                    "next_interval_hierarchical": score_frontier(
                        calibration, "score_next_interval", grid_size, confidence
                    ),
                }
                for method, frontier in frontiers.items():
                    choice = choose_policy(
                        frontier,
                        method,
                        resource_target,
                        minimum_selected,
                        target_action_risk=action_target,
                    )
                    row: dict[str, Any] = {
                        "fold": fold.name,
                        "horizon": horizon,
                        "calibration_window_days": days,
                        "calibration_start_day": window.start_day,
                        "calibration_end_day": window.end_day,
                        "calibration_mature_end_day": window.mature_end_day,
                        "calibration_mature_calendar_days": window.mature_calendar_days,
                        "test_start_day": fold.test_start_day,
                        "test_end_day": fold.test_end_day,
                        "method": method,
                        "eligible": choice is not None,
                        "calibration_rows": len(calibration),
                        "test_rows": len(test),
                    }
                    if choice is not None:
                        if method == "ttl":
                            actions = first_actions_by_ttl(
                                test, int(round(choice.parameter))
                            )
                        else:
                            score_column = (
                                "score_next_interval"
                                if method == "next_interval_hierarchical"
                                else "score_hazard"
                            )
                            actions = first_actions_by_score(
                                test, score_column, choice.parameter
                            )
                        metrics = action_metrics(actions, confidence, comparisons=1)
                        row.update(asdict(choice))
                        row.update({f"test_{key}": value for key, value in metrics.items()})
                        row["test_contract_pass"] = (
                            float(metrics["unweighted_risk_ucb"]) <= action_target
                            and float(metrics["risk_ucb"]) <= resource_target
                        )
                    rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(artifacts / "calibration_window_results.csv", index=False)
    report = _render_report(results, action_target, resource_target)
    report_path = artifacts / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


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
    results: pd.DataFrame, action_target: float, resource_target: float
) -> str:
    lines = [
        "# ColdGuard Calibration-Window Ablation",
        "",
        "The day-60 hierarchical model is frozen. Only the amount of chronological calibration history changes.",
        f"The operational contract is action-risk UCB <= {action_target:.0%} and resource-risk UCB <= {resource_target:.0%}.",
        "",
        "| Fold | Horizon | History | Method | Eligible | Test contract | Action UCB | Resource UCB | Reclaimed GiB-hours |",
        "|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    ordered = results.sort_values(
        ["fold", "horizon", "calibration_window_days", "method"]
    )
    for _, row in ordered.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | {int(row['calibration_window_days'])}d | "
                f"{row['method']} | no | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | {int(row['calibration_window_days'])}d | "
                f"{row['method']} | yes | {'PASS' if bool(row['test_contract_pass']) else 'FAIL'} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} | "
                f"{row['test_reclaimed_gib_hours']:,.0f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation rule: a longer window is useful only when it changes abstention into a passing policy or increases reclaimed memory while preserving the same held-out contract. Calibration eligibility alone is not treated as evidence of future safety.",
        ]
    )
    return "\n".join(lines) + "\n"
