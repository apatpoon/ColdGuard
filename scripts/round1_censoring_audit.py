from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from coldguard.config import load_config, resolve_project_path
from coldguard.multihorizon import attach_horizon_label


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_COLUMNS = ["app_id", "model_id", "site_id", "pd_role"]


def _deployment_id(frame: pd.DataFrame) -> pd.Series:
    return frame[DEPLOYMENT_COLUMNS].astype("string").fillna("<NA>").agg("|".join, axis=1)


def _episode_end_audit(episodes: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    episodes = episodes.copy()
    daily = daily.copy()
    episodes["deployment_id"] = _deployment_id(episodes)
    daily["deployment_id"] = _deployment_id(daily)
    last_observed = daily.groupby("deployment_id", observed=True)["day"].max()
    episodes["deployment_last_observed_day"] = episodes["deployment_id"].map(last_observed)
    trace_end = int(daily["day"].max())
    reason = pd.Series("reactivated", index=episodes.index, dtype="string")
    censored = ~episodes["reactivated"].astype(bool)
    reason.loc[censored & (episodes["end_cold_day"] >= trace_end)] = "administrative_trace_end"
    reason.loc[
        censored
        & (episodes["end_cold_day"] < trace_end)
        & (episodes["deployment_last_observed_day"] > episodes["end_cold_day"] + 1)
    ] = "telemetry_gap_then_reappears"
    reason.loc[
        censored
        & (episodes["end_cold_day"] < trace_end)
        & (episodes["deployment_last_observed_day"] <= episodes["end_cold_day"])
    ] = "deployment_disappearance"
    unresolved = censored & (reason == "reactivated")
    reason.loc[unresolved] = "unresolved_gap"
    episodes["resolved_end_reason"] = reason
    return episodes


def _risk_set_rows(
    landmarks: pd.DataFrame,
    definition: str,
    horizon: int,
    split: str,
    start_exclusive: int,
    end_inclusive: int,
) -> dict[str, object]:
    raw = landmarks.loc[
        (landmarks["day"] > start_exclusive)
        & (landmarks["day"] <= end_inclusive - horizon)
    ].copy()
    mature = attach_horizon_label(raw, horizon)
    deployments = _deployment_id(mature) if not mature.empty else pd.Series(dtype="string")
    episode_counts = mature.groupby(deployments, observed=True)["episode_id"].nunique()
    return {
        "cold_definition": definition,
        "split": split,
        "horizon": horizon,
        "calendar_start_day": start_exclusive + 1,
        "calendar_end_day_before_embargo": end_inclusive,
        "latest_decision_day": end_inclusive - horizon,
        "candidate_rows": len(raw),
        "mature_risk_set_rows": len(mature),
        "excluded_unmatured_or_gap_rows": len(raw) - len(mature),
        "revival_events": int(mature["reactivates_within_horizon"].sum()),
        "unique_episodes": int(mature["episode_id"].nunique()),
        "unique_deployments": int(deployments.nunique()),
        "deployments_with_repeated_episodes": int((episode_counts > 1).sum()),
        "decision_days": int(mature["day"].nunique()),
    }


def run(config_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    end_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    exposure_examples: list[dict[str, object]] = []
    for config_path in config_paths:
        config, project_root = load_config(config_path)
        definition = f"sm_le_{config['cold_definition']['sm_util_max']}_pct"
        episodes = pd.read_parquet(resolve_project_path(project_root, config["paths"]["episodes"]))
        daily = pd.read_parquet(resolve_project_path(project_root, config["paths"]["deployment_daily"]))
        landmarks = pd.read_parquet(resolve_project_path(project_root, config["paths"]["landmarks_all"]))
        audited = _episode_end_audit(episodes, daily)
        counts = audited["resolved_end_reason"].value_counts(dropna=False)
        for reason, count in counts.items():
            end_rows.append(
                {
                    "cold_definition": definition,
                    "end_reason": reason,
                    "episodes": int(count),
                    "share": float(count / len(audited)),
                }
            )
        cfg = config["multihorizon"]
        train_end = int(cfg["train_end_day"])
        calibration_end = int(cfg["calibration_end_day"])
        test_end = int(cfg["test_end_day"])
        for horizon in [1, 3, 7]:
            risk_rows.append(
                _risk_set_rows(
                    landmarks,
                    definition,
                    horizon,
                    "calibration",
                    train_end,
                    calibration_end,
                )
            )
            risk_rows.append(
                _risk_set_rows(
                    landmarks,
                    definition,
                    horizon,
                    "forward_test",
                    calibration_end,
                    test_end,
                )
            )

            eligible = attach_horizon_label(
                landmarks.loc[
                    (landmarks["day"] > calibration_end)
                    & (landmarks["day"] <= test_end - horizon)
                ],
                horizon,
            ).sort_values(["episode_id", "day"])
            if eligible.empty:
                continue
            example = eligible.iloc[len(eligible) // 2]
            episode = landmarks.loc[
                (landmarks["episode_id"] == example["episode_id"])
                & (landmarks["day"] >= example["day"])
                & (landmarks["day"] <= example["day"] + horizon - 1)
            ]
            reconstructed = float(episode["mem_gib_hours"].sum())
            exposure_examples.append(
                {
                    "cold_definition": definition,
                    "horizon": horizon,
                    "episode_id": int(example["episode_id"]),
                    "decision_day": int(example["day"]),
                    "exposure_end_day": int(min(example["end_cold_day"], example["day"] + horizon - 1)),
                    "source_rows": len(episode),
                    "horizon_capped_allocated_memory_gib_hours": reconstructed,
                    "decision_uses_exposure": False,
                    "observability": "matured audit outcome; online action uses only score, threshold, and current identifiers",
                }
            )

    pd.DataFrame(end_rows).to_csv(output_dir / "episode_end_reasons.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output_dir / "risk_set_censoring_audit.csv", index=False)
    pd.DataFrame(exposure_examples).to_csv(output_dir / "exposure_worked_examples.csv", index=False)
    estimand_example = {
        "actions_by_day": {"day_1": [1], "day_2": [0] * 9, "day_3": []},
        "pooled_action_risk": 0.1,
        "active_day_mean_risk": 0.5,
        "zero_filled_controller_day_mean": 1.0 / 3.0,
        "conclusion": "The three estimands differ; the revision controls the pooled ratio and does not encode no-action days as zero-risk observations.",
    }
    (output_dir / "estimand_counterexample.json").write_text(
        json.dumps(estimand_example, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["config/pilot.yaml", "config/pilot_strict.yaml"],
    )
    parser.add_argument("--output-dir", default="artifacts/revision_round1/censoring")
    args = parser.parse_args()
    run([ROOT / value for value in args.configs], ROOT / args.output_dir)


if __name__ == "__main__":
    main()
