from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd

from .calibration_window import EvaluationFold, calibration_window
from .config import load_config, require_keys, resolve_project_path
from .multihorizon import attach_horizon_label
from .policy import (
    PolicyChoice,
    action_metrics,
    choose_policy,
    first_actions_by_score,
    first_actions_by_ttl,
    score_frontier,
    ttl_frontier,
)
from .survival import predict_cumulative_incidence, predict_next_interval_score


@dataclass(frozen=True)
class WindowCandidate:
    window_days: int
    calibration_start_day: int
    calibration_end_day: int
    calibration_mature_end_day: int
    calibration_rows: int
    choice: PolicyChoice


def familywise_confidence(base_confidence: float, candidate_count: int) -> float:
    if not 0.0 < base_confidence < 1.0:
        raise ValueError("base_confidence must be between zero and one")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    return 1.0 - (1.0 - base_confidence) / candidate_count


def select_window_candidate(
    candidates: list[WindowCandidate],
) -> WindowCandidate | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.choice.calibration_reclaimed_gib_hours,
            candidate.window_days,
        ),
    )[0]


def run_window_selector(config_path: str | Path) -> Path:
    config, project_root = load_config(config_path)
    require_keys(
        config,
        "paths",
        [
            "landmarks_all",
            "multihorizon_artifacts_dir",
            "window_selector_artifacts_dir",
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
        project_root, config["paths"]["window_selector_artifacts_dir"]
    )
    artifacts.mkdir(parents=True, exist_ok=True)

    hazard_bundle = joblib.load(model_dir / "hazard_hierarchical.joblib")
    hazard_encoder, hazard_model = hazard_bundle["encoder"], hazard_bundle["model"]
    interval_bundle = joblib.load(model_dir / "next_interval_hierarchical.joblib")
    interval_encoder, interval_model = interval_bundle["encoder"], interval_bundle["model"]

    cfg = config["calibration_window"]
    horizons = sorted(set(int(value) for value in cfg["horizons"]))
    configured_windows = sorted(set(int(value) for value in cfg["window_days"]))
    earliest_day = int(cfg["earliest_day"])
    folds = [EvaluationFold(**values) for values in cfg["test_folds"]]
    action_target = float(config["multihorizon"]["target_action_risk"])
    resource_target = float(config["multihorizon"]["target_resource_risk"])
    base_confidence = float(config["risk"]["confidence"])
    grid_size = int(config["risk"]["threshold_grid_size"])
    minimum_selected = int(config["risk"]["minimum_selected"])

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for fold in folds:
        for horizon in horizons:
            windows = [
                window
                for days in configured_windows
                if (window := calibration_window(fold, days, horizon, earliest_day))
                is not None
            ]
            if not windows:
                raise ValueError(f"No candidate windows for {fold.name}, horizon {horizon}")
            selection_confidence = familywise_confidence(
                base_confidence, len(windows)
            )

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

            candidates_by_method: dict[str, list[WindowCandidate]] = {
                "ttl": [],
                "hazard_hierarchical": [],
                "next_interval_hierarchical": [],
            }
            for window in windows:
                calibration = attach_horizon_label(
                    _load_frame(landmarks, window.condition()), horizon
                )
                if calibration.empty:
                    raise ValueError(
                        f"Empty calibration frame for {fold.name}, horizon {horizon}, "
                        f"window {window.calendar_days}"
                    )
                calibration["score_hazard"] = predict_cumulative_incidence(
                    calibration, hazard_encoder, hazard_model, horizon
                )
                calibration["score_next_interval"] = predict_next_interval_score(
                    calibration, interval_encoder, interval_model, horizon
                )
                frontiers = {
                    "ttl": ttl_frontier(calibration, selection_confidence),
                    "hazard_hierarchical": score_frontier(
                        calibration,
                        "score_hazard",
                        grid_size,
                        selection_confidence,
                    ),
                    "next_interval_hierarchical": score_frontier(
                        calibration,
                        "score_next_interval",
                        grid_size,
                        selection_confidence,
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
                    candidate_rows.append(
                        {
                            "fold": fold.name,
                            "horizon": horizon,
                            "method": method,
                            "candidate_window_days": window.calendar_days,
                            "candidate_count": len(windows),
                            "selection_confidence": selection_confidence,
                            "eligible": choice is not None,
                            "calibration_rows": len(calibration),
                            **(
                                {f"choice_{key}": value for key, value in asdict(choice).items()}
                                if choice is not None
                                else {}
                            ),
                        }
                    )
                    if choice is not None:
                        candidates_by_method[method].append(
                            WindowCandidate(
                                window_days=window.calendar_days,
                                calibration_start_day=window.start_day,
                                calibration_end_day=window.end_day,
                                calibration_mature_end_day=window.mature_end_day,
                                calibration_rows=len(calibration),
                                choice=choice,
                            )
                        )

            for method, candidates in candidates_by_method.items():
                selected = select_window_candidate(candidates)
                row: dict[str, Any] = {
                    "fold": fold.name,
                    "horizon": horizon,
                    "method": method,
                    "candidate_windows": ",".join(
                        str(window.calendar_days) for window in windows
                    ),
                    "candidate_count": len(windows),
                    "selection_confidence": selection_confidence,
                    "eligible": selected is not None,
                    "test_rows": len(test),
                }
                if selected is not None:
                    if method == "ttl":
                        actions = first_actions_by_ttl(
                            test, int(round(selected.choice.parameter))
                        )
                    else:
                        score_column = (
                            "score_next_interval"
                            if method == "next_interval_hierarchical"
                            else "score_hazard"
                        )
                        actions = first_actions_by_score(
                            test, score_column, selected.choice.parameter
                        )
                    metrics = action_metrics(actions, base_confidence, comparisons=1)
                    row.update(
                        {
                            "selected_window_days": selected.window_days,
                            "calibration_start_day": selected.calibration_start_day,
                            "calibration_end_day": selected.calibration_end_day,
                            "calibration_mature_end_day": selected.calibration_mature_end_day,
                            "calibration_rows": selected.calibration_rows,
                            **asdict(selected.choice),
                            **{f"test_{key}": value for key, value in metrics.items()},
                            "test_contract_pass": (
                                float(metrics["unweighted_risk_ucb"])
                                <= action_target
                                and float(metrics["risk_ucb"]) <= resource_target
                            ),
                        }
                    )
                rows.append(row)

    results = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    results.to_csv(artifacts / "selected_window_results.csv", index=False)
    candidates.to_csv(artifacts / "candidate_window_audit.csv", index=False)
    report = _render_report(results, base_confidence, action_target, resource_target)
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
    results: pd.DataFrame,
    base_confidence: float,
    action_target: float,
    resource_target: float,
) -> str:
    lines = [
        "# ColdGuard Search-Corrected Calibration-Window Selector",
        "",
        "Window and threshold selection use calibration data only. Bonferroni correction covers all available windows and all threshold candidates; the selected policy is evaluated once on the following test block.",
        f"Held-out risk UCBs are reported at {base_confidence:.0%}; the operational contract is action <= {action_target:.0%} and resource <= {resource_target:.0%}.",
        "",
        "| Fold | Horizon | Method | Candidate windows | Selected | Eligible | Test contract | Action UCB | Resource UCB | Reclaimed GiB-hours |",
        "|---|---:|---|---|---:|---|---|---:|---:|---:|",
    ]
    for _, row in results.sort_values(["fold", "horizon", "method"]).iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | {row['method']} | "
                f"{row['candidate_windows']} | — | no | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {row['fold']} | {int(row['horizon'])} | {row['method']} | "
                f"{row['candidate_windows']} | {int(row['selected_window_days'])}d | yes | "
                f"{'PASS' if bool(row['test_contract_pass']) else 'FAIL'} | "
                f"{row['test_unweighted_risk_ucb']:.2%} | {row['test_risk_ucb']:.2%} | "
                f"{row['test_reclaimed_gib_hours']:,.0f} |"
            )
    return "\n".join(lines) + "\n"
