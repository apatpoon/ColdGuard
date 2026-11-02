from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from coldguard.clustered_risk import dual_clustered_risk_bound
from coldguard.config import load_config, resolve_project_path
from coldguard.multihorizon import horizon_is_observed
from coldguard.policy import first_actions_by_score
from coldguard.survival import predict_cumulative_incidence


ROOT = Path(__file__).resolve().parents[1]


def _horizon_exposure(frame: pd.DataFrame, horizon: int) -> pd.Series:
    ordered = frame.sort_values(["episode_id", "day"]).copy()
    exposure = np.zeros(len(ordered), dtype=float)
    grouped = ordered.groupby("episode_id", observed=True, sort=False)["mem_gib_hours"]
    for offset in range(horizon):
        exposure += grouped.shift(-offset).fillna(0.0).to_numpy(float)
    result = pd.Series(exposure, index=ordered.index)
    return result.reindex(frame.index)


def _flatten(prefix: str, value) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for risk_name, bound in [("action", value.action), ("resource", value.resource)]:
        for key, item in asdict(bound).items():
            row[f"{prefix}_{risk_name}_{key}"] = item
    return row


def run(config_paths: list[Path], output_path: Path) -> None:
    policy = pd.read_csv(ROOT / "artifacts/revision_round1/operational_policy_audit.csv")
    rows: list[dict[str, object]] = []
    for config_path in config_paths:
        config, project_root = load_config(config_path)
        definition = f"sm_le_{config['cold_definition']['sm_util_max']}_pct"
        cfg = config["multihorizon"]
        calibration_end = int(cfg["calibration_end_day"])
        test_end = int(cfg["test_end_day"])
        landmarks = pd.read_parquet(resolve_project_path(project_root, config["paths"]["landmarks_all"]))
        model_dir = resolve_project_path(project_root, config["paths"]["multihorizon_artifacts_dir"]) / "models"
        hazard = joblib.load(model_dir / "hazard_hierarchical.joblib")
        for horizon in [1, 3, 7]:
            selected = policy[
                (policy["cold_definition"] == definition)
                & (policy["horizon"] == horizon)
                & (policy["method"] == "hazard_hierarchical")
            ].iloc[0]
            if not bool(selected["calibration_qualified"]):
                continue
            raw = landmarks.loc[
                (landmarks["day"] > calibration_end)
                & (landmarks["day"] <= test_end - horizon)
            ].copy()
            raw["score"] = predict_cumulative_incidence(
                raw, hazard["encoder"], hazard["model"], horizon
            )
            raw["horizon_mem_gib_hours"] = _horizon_exposure(raw, horizon)
            raw["outcome_observed"] = horizon_is_observed(raw, horizon)
            event_day = pd.to_numeric(raw["event_day"], errors="coerce")
            raw["observed_label"] = (
                event_day.notna() & (event_day <= raw["day"] + horizon)
            ).astype(int)
            actions = first_actions_by_score(raw, "score", float(selected["selected_parameter"]))
            observed_actions = actions["outcome_observed"].astype(bool)
            action_key = actions[["episode_id", "day"]].astype(str).agg("|".join, axis=1)
            permuted_raw = raw.copy()
            permuted_raw["horizon_mem_gib_hours"] = np.random.default_rng(
                20261028 + horizon
            ).permutation(
                permuted_raw["horizon_mem_gib_hours"].to_numpy()
            )
            permuted_actions = first_actions_by_score(
                permuted_raw, "score", float(selected["selected_parameter"])
            )
            permuted_key = permuted_actions[["episode_id", "day"]].astype(str).agg("|".join, axis=1)
            no_lookahead_pass = set(action_key) == set(permuted_key)
            base = {
                "cold_definition": definition,
                "horizon": horizon,
                "selected_parameter": float(selected["selected_parameter"]),
                "all_proposed_actions": len(actions),
                "observed_actions": int(observed_actions.sum()),
                "unresolved_actions": int((~observed_actions).sum()),
                "unresolved_share": float((~observed_actions).mean()) if len(actions) else 0.0,
                "no_lookahead_permutation_pass": no_lookahead_pass,
            }
            for scenario, unresolved_label in [("optimistic_no_revival", 0), ("pessimistic_revival", 1)]:
                imputed = actions.copy()
                imputed["reactivates_within_horizon"] = imputed["observed_label"]
                imputed.loc[~observed_actions, "reactivates_within_horizon"] = unresolved_label
                if unresolved_label == 1:
                    persisted = (
                        imputed["avg_mem_gib_calendar"].fillna(0.0).clip(lower=0.0)
                        * 24.0
                        * horizon
                    )
                    imputed.loc[~observed_actions, "horizon_mem_gib_hours"] = np.maximum(
                        imputed.loc[~observed_actions, "horizon_mem_gib_hours"],
                        persisted.loc[~observed_actions],
                    )
                bound = dual_clustered_risk_bound(
                    imputed,
                    family_confidence=float(config["risk"]["confidence"]),
                    exposure_column="horizon_mem_gib_hours",
                )
                rows.append({**base, "scenario": scenario, **_flatten("test", bound)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["config/pilot.yaml", "config/pilot_strict.yaml"],
    )
    parser.add_argument(
        "--output",
        default="artifacts/revision_round1/censoring/disappearance_sensitivity.csv",
    )
    args = parser.parse_args()
    run([ROOT / value for value in args.configs], ROOT / args.output)


if __name__ == "__main__":
    main()
