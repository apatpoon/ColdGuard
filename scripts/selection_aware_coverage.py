from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from coldguard.clustered_risk import dual_clustered_risk_bound
from coldguard.policy import first_actions_by_score


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "iid": {"day_sd": 0.0, "deployment_sd": 0.0, "weight_sd": 0.25, "test_shift": 0.0},
    "day_shock": {"day_sd": 0.75, "deployment_sd": 0.0, "weight_sd": 0.25, "test_shift": 0.0},
    "repeated_deployment": {"day_sd": 0.0, "deployment_sd": 0.90, "weight_sd": 0.25, "test_shift": 0.0},
    "two_axis_heavy_weight": {"day_sd": 0.75, "deployment_sd": 0.90, "weight_sd": 1.20, "test_shift": 0.0},
    "two_axis_test_shift": {"day_sd": 0.75, "deployment_sd": 0.90, "weight_sd": 1.20, "test_shift": 0.90},
}
PROTOCOLS = {
    "fixed_sequence_pointwise": lambda family, count: family,
    "threshold_bonferroni": lambda family, count: 1.0 - (1.0 - family) / count,
}


def _panel(
    rng: np.random.Generator,
    *,
    days: int,
    deployments: int,
    episodes: int,
    landmarks: int,
    day_sd: float,
    deployment_sd: float,
    weight_sd: float,
    deployment_effect: np.ndarray,
    logit_shift: float,
) -> pd.DataFrame:
    episode = np.arange(episodes)
    deployment = rng.integers(0, deployments, size=episodes)
    base_day = rng.integers(0, days - landmarks + 1, size=episodes)
    difficulty = rng.normal(0.0, 1.0, size=episodes)
    repeated_episode = np.repeat(episode, landmarks)
    repeated_deployment = np.repeat(deployment, landmarks)
    age = np.tile(np.arange(landmarks), episodes)
    day = np.repeat(base_day, landmarks) + age
    day_effect = rng.normal(0.0, day_sd, size=days)
    log_weight = (
        rng.normal(0.0, weight_sd, size=episodes)
        + 0.20 * difficulty
        + 0.15 * deployment_effect[deployment]
    )
    repeated_weight = np.exp(np.repeat(log_weight, landmarks))
    latent = (
        -4.15
        + np.repeat(difficulty, landmarks)
        + day_effect[day]
        + deployment_effect[repeated_deployment]
        + 0.10 * np.repeat(log_weight, landmarks)
        - 0.22 * age
        + logit_shift
    )
    probability = expit(latent)
    score_latent = (
        -4.10
        + 0.90 * np.repeat(difficulty, landmarks)
        + 0.55 * day_effect[day]
        + 0.55 * deployment_effect[repeated_deployment]
        + 0.08 * np.repeat(log_weight, landmarks)
        - 0.22 * age
        + rng.normal(0.0, 0.38, size=len(age))
    )
    score = expit(score_latent)
    label = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "episode_id": repeated_episode.astype(str),
            "day": day,
            "app_id": pd.Series(repeated_deployment).astype(str),
            "model_id": "m",
            "site_id": "s",
            "pd_role": "r",
            "score": score,
            "reactivates_within_horizon": label,
            "remaining_mem_gib_hours": repeated_weight,
            "true_probability": probability,
        }
    )


def _oracle(actions: pd.DataFrame) -> tuple[float, float]:
    if actions.empty:
        return np.nan, np.nan
    probability = actions["true_probability"].to_numpy(float)
    weight = actions["remaining_mem_gib_hours"].to_numpy(float)
    return float(probability.mean()), float(np.dot(weight, probability) / weight.sum())


def _choose(
    calibration: pd.DataFrame,
    thresholds: np.ndarray,
    *,
    confidence: float,
    minimum_actions: int,
    budget: float,
) -> dict[str, object]:
    selected: dict[str, object] | None = None
    tested = 0
    for index, threshold in enumerate(thresholds):
        actions = first_actions_by_score(calibration, "score", float(threshold))
        if len(actions) < minimum_actions:
            continue
        tested += 1
        bound = dual_clustered_risk_bound(actions, family_confidence=confidence)
        passed = bound.action.upper <= budget and bound.resource.upper <= budget
        if not passed:
            break
        action_oracle, resource_oracle = _oracle(actions)
        selected = {
            "threshold_index": index,
            "threshold": float(threshold),
            "calibration_actions": int(len(actions)),
            "calibration_action_upper": bound.action.upper,
            "calibration_resource_upper": bound.resource.upper,
            "calibration_action_oracle": action_oracle,
            "calibration_resource_oracle": resource_oracle,
            "calibration_joint_coverage": bool(
                bound.action.upper >= action_oracle
                and bound.resource.upper >= resource_oracle
            ),
            "calibration_true_safe": bool(
                action_oracle <= budget and resource_oracle <= budget
            ),
        }
    return {"selected": selected, "thresholds_tested": tested}


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials == 0:
        return np.nan, np.nan
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z / denominator * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return float(max(0, center - radius)), float(min(1, center + radius))


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, protocol), group in detail.groupby(["scenario", "protocol"], sort=True):
        selected = group.loc[group["selected"]].copy()
        n = len(group)
        n_selected = len(selected)
        metrics = {
            "selection_rate": ("selected", n),
            "calibration_joint_coverage": ("calibration_joint_coverage", n_selected),
            "calibration_true_safe": ("calibration_true_safe", n_selected),
            "forward_joint_coverage": ("forward_joint_coverage", n_selected),
            "forward_true_safe": ("forward_true_safe", n_selected),
            "forward_bound_pass": ("forward_bound_pass", n_selected),
        }
        row: dict[str, object] = {
            "scenario": scenario,
            "protocol": protocol,
            "replications": n,
            "selected_replications": n_selected,
            "median_thresholds_tested": float(group["thresholds_tested"].median()),
            "median_selected_actions": float(selected["calibration_actions"].median()) if n_selected else np.nan,
        }
        for name, (column, denominator) in metrics.items():
            numerator = int(group[column].sum()) if name == "selection_rate" else int(selected[column].sum())
            estimate = numerator / denominator if denominator else np.nan
            low, high = _wilson_interval(numerator, denominator)
            row[name] = estimate
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        row["false_qualification"] = 1.0 - row["forward_true_safe"] if n_selected else np.nan
        if n_selected:
            unsafe = n_selected - int(selected["forward_true_safe"].sum())
            low, high = _wilson_interval(unsafe, n_selected)
            row["false_qualification_ci_low"] = low
            row["false_qualification_ci_high"] = high
        else:
            row["false_qualification_ci_low"] = np.nan
            row["false_qualification_ci_high"] = np.nan
        unsafe = selected.loc[~selected["forward_true_safe"]]
        unsafe_count = int(len(unsafe))
        unsafe_caught = int((~unsafe["forward_bound_pass"]).sum()) if unsafe_count else 0
        false_promotions = int(
            ((~selected["forward_true_safe"]) & selected["forward_bound_pass"]).sum()
        ) if n_selected else 0
        row["unsafe_forward_selected"] = unsafe_count
        row["unsafe_caught_by_forward_audit"] = (
            unsafe_caught / unsafe_count if unsafe_count else np.nan
        )
        row["false_promotion"] = false_promotions / n_selected if n_selected else np.nan
        low, high = _wilson_interval(false_promotions, n_selected)
        row["false_promotion_ci_low"] = low
        row["false_promotion_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    output_dir: Path,
    *,
    replications: int,
    seed: int,
    days: int,
    deployments: int,
    episodes: int,
    landmarks: int,
    thresholds: np.ndarray,
    minimum_actions: int,
    family_confidence: float,
    budget: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for scenario, parameters in SCENARIOS.items():
        for replication in range(replications):
            deployment_effect = rng.normal(
                0.0, parameters["deployment_sd"], size=deployments
            )
            calibration = _panel(
                rng,
                days=days,
                deployments=deployments,
                episodes=episodes,
                landmarks=landmarks,
                day_sd=parameters["day_sd"],
                deployment_sd=parameters["deployment_sd"],
                weight_sd=parameters["weight_sd"],
                deployment_effect=deployment_effect,
                logit_shift=0.0,
            )
            test = _panel(
                rng,
                days=days,
                deployments=deployments,
                episodes=episodes,
                landmarks=landmarks,
                day_sd=parameters["day_sd"],
                deployment_sd=parameters["deployment_sd"],
                weight_sd=parameters["weight_sd"],
                deployment_effect=deployment_effect,
                logit_shift=parameters["test_shift"],
            )
            for protocol, confidence_rule in PROTOCOLS.items():
                selection_confidence = confidence_rule(family_confidence, len(thresholds))
                choice = _choose(
                    calibration,
                    thresholds,
                    confidence=selection_confidence,
                    minimum_actions=minimum_actions,
                    budget=budget,
                )
                selected = choice["selected"]
                row: dict[str, object] = {
                    "scenario": scenario,
                    "protocol": protocol,
                    "replication": replication,
                    "selection_confidence": selection_confidence,
                    "thresholds_tested": choice["thresholds_tested"],
                    "selected": selected is not None,
                    "calibration_actions": np.nan,
                    "calibration_joint_coverage": False,
                    "calibration_true_safe": False,
                    "forward_joint_coverage": False,
                    "forward_true_safe": False,
                    "forward_bound_pass": False,
                }
                if selected is not None:
                    row.update(selected)
                    threshold = float(selected["threshold"])
                    test_actions = first_actions_by_score(test, "score", threshold)
                    forward_bound = dual_clustered_risk_bound(
                        test_actions, family_confidence=family_confidence
                    )
                    action_oracle, resource_oracle = _oracle(test_actions)
                    row.update(
                        {
                            "forward_actions": int(len(test_actions)),
                            "forward_action_oracle": action_oracle,
                            "forward_resource_oracle": resource_oracle,
                            "forward_action_upper": forward_bound.action.upper,
                            "forward_resource_upper": forward_bound.resource.upper,
                            "forward_joint_coverage": bool(
                                forward_bound.action.upper >= action_oracle
                                and forward_bound.resource.upper >= resource_oracle
                            ),
                            "forward_true_safe": bool(
                                action_oracle <= budget and resource_oracle <= budget
                            ),
                            "forward_bound_pass": bool(
                                forward_bound.action.upper <= budget
                                and forward_bound.resource.upper <= budget
                            ),
                        }
                    )
                rows.append(row)
    detail = pd.DataFrame(rows)
    summary = _summarize(detail)
    detail.to_csv(output_dir / "selection_replications.csv", index=False)
    summary.to_csv(output_dir / "selection_summary.csv", index=False)
    design = {
        "seed": seed,
        "replications_per_scenario": replications,
        "scenarios": SCENARIOS,
        "protocols": list(PROTOCOLS),
        "days_per_panel": days,
        "deployments": deployments,
        "episodes_per_panel": episodes,
        "landmarks_per_episode": landmarks,
        "thresholds": thresholds.tolist(),
        "minimum_actions": minimum_actions,
        "family_confidence": family_confidence,
        "dual_risk_budget": budget,
        "estimand": "conditional expected pooled action and exposure ratios of the selected policy",
        "scope": "empirical Monte Carlo stress test of the entire selection-and-forward-audit procedure; not a theorem",
    }
    (output_dir / "design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/revision_round2/selection_aware_coverage")
    parser.add_argument("--replications", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--deployments", type=int, default=160)
    parser.add_argument("--episodes", type=int, default=320)
    parser.add_argument("--landmarks", type=int, default=3)
    parser.add_argument("--minimum-actions", type=int, default=80)
    parser.add_argument("--family-confidence", type=float, default=0.95)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--threshold-count", type=int, default=12)
    args = parser.parse_args()
    thresholds = np.linspace(0.015, 0.18, args.threshold_count)
    run(
        ROOT / args.output_dir,
        replications=args.replications,
        seed=args.seed,
        days=args.days,
        deployments=args.deployments,
        episodes=args.episodes,
        landmarks=args.landmarks,
        thresholds=thresholds,
        minimum_actions=args.minimum_actions,
        family_confidence=args.family_confidence,
        budget=args.budget,
    )


if __name__ == "__main__":
    main()
