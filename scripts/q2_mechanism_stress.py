from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from coldguard.clustered_risk import dual_clustered_risk_bound
from coldguard.policy import action_metrics, first_actions_by_score


ROOT = Path(__file__).resolve().parents[1]


SCENARIOS: dict[str, dict[str, float]] = {
    "iid": {
        "day_sd": 0.0,
        "deployment_sd": 0.0,
        "weight_sd": 0.25,
        "outcome_shift": 0.0,
        "domain_fraction": 0.0,
        "domain_shift": 0.0,
        "weight_outcome_coef": 0.10,
        "weight_score_coef": 0.08,
        "day_score_coef": 0.55,
        "deployment_score_coef": 0.55,
        "observation_intercept": 20.0,
        "observation_risk_coef": 0.0,
        "observation_weight_coef": 0.0,
    },
    "two_axis_heavy": {
        "day_sd": 0.75,
        "deployment_sd": 0.90,
        "weight_sd": 1.20,
        "outcome_shift": 0.0,
        "domain_fraction": 0.0,
        "domain_shift": 0.0,
        "weight_outcome_coef": 0.10,
        "weight_score_coef": 0.08,
        "day_score_coef": 0.55,
        "deployment_score_coef": 0.55,
        "observation_intercept": 20.0,
        "observation_risk_coef": 0.0,
        "observation_weight_coef": 0.0,
    },
    "resource_concentration": {
        "day_sd": 0.45,
        "deployment_sd": 0.55,
        "weight_sd": 1.35,
        "outcome_shift": 0.0,
        "domain_fraction": 0.0,
        "domain_shift": 0.0,
        "weight_outcome_coef": 0.65,
        "weight_score_coef": 0.02,
        "day_score_coef": 0.55,
        "deployment_score_coef": 0.55,
        "observation_intercept": 20.0,
        "observation_risk_coef": 0.0,
        "observation_weight_coef": 0.0,
    },
    "abrupt_forward_shift": {
        "day_sd": 0.75,
        "deployment_sd": 0.90,
        "weight_sd": 1.20,
        "outcome_shift": 1.20,
        "domain_fraction": 0.0,
        "domain_shift": 0.0,
        "weight_outcome_coef": 0.10,
        "weight_score_coef": 0.08,
        "day_score_coef": 0.55,
        "deployment_score_coef": 0.55,
        "observation_intercept": 20.0,
        "observation_risk_coef": 0.0,
        "observation_weight_coef": 0.0,
    },
    "unseen_domain_shift": {
        "day_sd": 0.55,
        "deployment_sd": 0.70,
        "weight_sd": 0.90,
        "outcome_shift": 0.0,
        "domain_fraction": 0.40,
        "domain_shift": 1.60,
        "weight_outcome_coef": 0.10,
        "weight_score_coef": 0.08,
        "day_score_coef": 0.55,
        "deployment_score_coef": 0.55,
        "observation_intercept": 20.0,
        "observation_risk_coef": 0.0,
        "observation_weight_coef": 0.0,
    },
    "informative_endpoint_loss": {
        "day_sd": 0.55,
        "deployment_sd": 0.70,
        "weight_sd": 0.45,
        "outcome_shift": 1.35,
        "domain_fraction": 0.0,
        "domain_shift": 0.0,
        "weight_outcome_coef": 0.10,
        "weight_score_coef": 0.08,
        "day_score_coef": 0.55,
        "deployment_score_coef": 0.55,
        "observation_intercept": 4.20,
        "observation_risk_coef": 18.00,
        "observation_weight_coef": 0.70,
    },
    "residual_cluster_dependence": {
        "day_sd": 1.25,
        "deployment_sd": 1.35,
        "weight_sd": 1.00,
        "outcome_shift": 0.0,
        "domain_fraction": 0.0,
        "domain_shift": 0.0,
        "weight_outcome_coef": 0.15,
        "weight_score_coef": 0.05,
        "day_score_coef": 0.12,
        "deployment_score_coef": 0.12,
        "observation_intercept": 20.0,
        "observation_risk_coef": 0.0,
        "observation_weight_coef": 0.0,
    },
}

PROTOCOLS = [
    "coldguard_full",
    "no_forward_expiry",
    "complete_case_only",
    "action_risk_only",
    "rowwise_naive",
]


@dataclass(frozen=True)
class Candidate:
    threshold: float
    calibration_actions: int
    calibration_observed_actions: int
    calibration_action_upper: float
    calibration_resource_upper: float


def _panel(
    rng: np.random.Generator,
    *,
    parameters: dict[str, float],
    days: int,
    deployments: int,
    episodes: int,
    landmarks: int,
    deployment_effect: np.ndarray,
    forward: bool,
) -> pd.DataFrame:
    episode = np.arange(episodes)
    deployment = rng.integers(0, deployments, size=episodes)
    base_day = rng.integers(0, days - landmarks + 1, size=episodes)
    difficulty = rng.normal(0.0, 1.0, size=episodes)
    repeated_episode = np.repeat(episode, landmarks)
    repeated_deployment = np.repeat(deployment, landmarks)
    age = np.tile(np.arange(landmarks), episodes)
    day = np.repeat(base_day, landmarks) + age
    day_effect = rng.normal(0.0, parameters["day_sd"], size=days)
    log_weight = (
        rng.normal(0.0, parameters["weight_sd"], size=episodes)
        + 0.20 * difficulty
        + 0.15 * deployment_effect[deployment]
    )
    repeated_log_weight = np.repeat(log_weight, landmarks)
    weight = np.exp(repeated_log_weight)

    domain_flag = np.zeros(deployments, dtype=float)
    if forward and parameters["domain_fraction"] > 0.0:
        count = max(1, int(round(deployments * parameters["domain_fraction"])))
        chosen = rng.choice(deployments, size=count, replace=False)
        domain_flag[chosen] = 1.0
    domain_outcome_shift = (
        parameters["domain_shift"] * domain_flag[repeated_deployment]
    )
    forward_shift = parameters["outcome_shift"] if forward else 0.0
    latent = (
        -4.15
        + np.repeat(difficulty, landmarks)
        + day_effect[day]
        + deployment_effect[repeated_deployment]
        + parameters["weight_outcome_coef"] * repeated_log_weight
        - 0.22 * age
        + forward_shift
        + domain_outcome_shift
    )
    probability = expit(latent)
    score_latent = (
        -4.10
        + 0.90 * np.repeat(difficulty, landmarks)
        + parameters["day_score_coef"] * day_effect[day]
        + parameters["deployment_score_coef"]
        * deployment_effect[repeated_deployment]
        + parameters["weight_score_coef"] * repeated_log_weight
        - 0.22 * age
        + rng.normal(0.0, 0.38, size=len(age))
    )
    score = expit(score_latent)
    label = rng.binomial(1, probability)

    standardized_log_weight = (
        repeated_log_weight - float(repeated_log_weight.mean())
    ) / max(float(repeated_log_weight.std()), 1e-8)
    observation_logit = (
        parameters["observation_intercept"]
        - parameters["observation_risk_coef"] * probability
        - parameters["observation_weight_coef"] * standardized_log_weight
    )
    observation_probability = expit(observation_logit)
    observed = rng.binomial(1, observation_probability).astype(bool)

    return pd.DataFrame(
        {
            "episode_id": repeated_episode.astype(str),
            "day": day,
            "app_id": pd.Series(repeated_deployment).astype(str),
            "model_id": "m",
            "site_id": np.where(
                domain_flag[repeated_deployment] > 0.0, "shifted", "reference"
            ),
            "pd_role": "r",
            "score": score,
            "reactivates_within_horizon": label,
            "remaining_mem_gib_hours": weight,
            "true_probability": probability,
            "outcome_observed": observed,
        }
    )


def _oracle(actions: pd.DataFrame) -> tuple[float, float]:
    if actions.empty:
        return np.nan, np.nan
    probability = actions["true_probability"].to_numpy(float)
    weight = actions["remaining_mem_gib_hours"].to_numpy(float)
    return float(probability.mean()), float(np.dot(weight, probability) / weight.sum())


def _bound(
    proposals: pd.DataFrame,
    *,
    protocol: str,
    family_confidence: float,
    maturity_upper: bool,
) -> tuple[float, float, int]:
    frame = proposals.copy()
    if maturity_upper:
        frame.loc[~frame["outcome_observed"], "reactivates_within_horizon"] = 1
    else:
        frame = frame.loc[frame["outcome_observed"]].copy()
    if frame.empty:
        return 1.0, 1.0, 0
    if protocol == "rowwise_naive":
        metrics = action_metrics(frame, confidence=family_confidence, comparisons=2)
        return (
            float(metrics["unweighted_risk_ucb"]),
            float(metrics["risk_ucb"]),
            len(frame),
        )
    bound = dual_clustered_risk_bound(frame, family_confidence=family_confidence)
    return bound.action.upper, bound.resource.upper, len(frame)


def _choose(
    calibration: pd.DataFrame,
    thresholds: np.ndarray,
    *,
    protocol: str,
    family_confidence: float,
    minimum_actions: int,
    budget: float,
) -> tuple[Candidate | None, int]:
    selected: Candidate | None = None
    tested = 0
    for threshold in thresholds:
        proposals = first_actions_by_score(calibration, "score", float(threshold))
        observed_count = int(proposals["outcome_observed"].sum())
        if observed_count < minimum_actions:
            continue
        tested += 1
        action_upper, resource_upper, _ = _bound(
            proposals,
            protocol=protocol,
            family_confidence=family_confidence,
            maturity_upper=False,
        )
        if protocol == "action_risk_only":
            passed = action_upper <= budget
        else:
            passed = action_upper <= budget and resource_upper <= budget
        if not passed:
            break
        selected = Candidate(
            threshold=float(threshold),
            calibration_actions=len(proposals),
            calibration_observed_actions=observed_count,
            calibration_action_upper=action_upper,
            calibration_resource_upper=resource_upper,
        )
    return selected, tested


def _wilson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials == 0:
        return np.nan, np.nan
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z / denominator * np.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    )
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario, protocol), group in detail.groupby(
        ["scenario", "protocol"], sort=True
    ):
        selected = group.loc[group["selected"]]
        promoted = group.loc[group["promoted"]]
        unsafe_selected = selected.loc[~selected["forward_true_safe"]]
        false_promotions = promoted.loc[~promoted["forward_true_safe"]]
        row: dict[str, object] = {
            "scenario": scenario,
            "protocol": protocol,
            "replications": len(group),
            "selected_replications": len(selected),
            "promoted_replications": len(promoted),
            "unsafe_selected": len(unsafe_selected),
            "false_promotions": len(false_promotions),
            "median_selected_actions": (
                float(selected["forward_actions"].median()) if len(selected) else np.nan
            ),
            "median_observed_share": (
                float(selected["forward_observed_share"].median())
                if len(selected)
                else np.nan
            ),
        }
        rate_specs = {
            "selection_rate": (len(selected), len(group)),
            "promotion_rate": (len(promoted), len(group)),
            "false_promotion_rate": (len(false_promotions), len(group)),
            "unsafe_catch_rate": (
                len(unsafe_selected) - len(false_promotions),
                len(unsafe_selected),
            ),
        }
        for name, (numerator, denominator) in rate_specs.items():
            row[name] = numerator / denominator if denominator else np.nan
            low, high = _wilson(numerator, denominator)
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _report(summary: pd.DataFrame, replications: int) -> str:
    lines = [
        "# ColdGuard Q2 mechanism stress suite",
        "",
        (
            "This experiment regenerates calibration and independent forward panels, "
            "selects a threshold, and then removes one safety mechanism at a time. "
            f"Each scenario contains {replications} complete Monte Carlo replications."
        ),
        "",
        "| Scenario | Protocol | Selected | Promoted | Unsafe selected | False promotions | Catch rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "coldguard_full": "Full contract",
        "no_forward_expiry": "No expiry",
        "complete_case_only": "Complete case",
        "action_risk_only": "Action risk only",
        "rowwise_naive": "Rowwise naive",
    }
    for row in summary.itertuples(index=False):
        catch = "—" if pd.isna(row.unsafe_catch_rate) else f"{row.unsafe_catch_rate:.1%}"
        lines.append(
            f"| {row.scenario} | {labels[row.protocol]} | "
            f"{row.selected_replications}/{row.replications} | "
            f"{row.promoted_replications}/{row.replications} | "
            f"{row.unsafe_selected} | {row.false_promotions} | {catch} |"
        )
    lines.extend(
        [
            "",
            "`coldguard_full` applies both pooled risks, the conservative unresolved-outcome endpoint, and an independent forward expiry audit. `no_forward_expiry` promotes a calibration-selected policy without the forward audit. `complete_case_only` audits observed outcomes but ignores unresolved proposals. `action_risk_only` ignores exposure concentration. `rowwise_naive` replaces the day/deployment stress bound with Wilson/Kish rowwise bounds.",
            "",
            "All results are empirical stress evidence for the declared generators, not a distribution-free theorem or production intervention.",
        ]
    )
    return "\n".join(lines) + "\n"


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
                parameters=parameters,
                days=days,
                deployments=deployments,
                episodes=episodes,
                landmarks=landmarks,
                deployment_effect=deployment_effect,
                forward=False,
            )
            forward = _panel(
                rng,
                parameters=parameters,
                days=days,
                deployments=deployments,
                episodes=episodes,
                landmarks=landmarks,
                deployment_effect=deployment_effect,
                forward=True,
            )
            for protocol in PROTOCOLS:
                selected, tested = _choose(
                    calibration,
                    thresholds,
                    protocol=protocol,
                    family_confidence=family_confidence,
                    minimum_actions=minimum_actions,
                    budget=budget,
                )
                row: dict[str, object] = {
                    "scenario": scenario,
                    "protocol": protocol,
                    "replication": replication,
                    "thresholds_tested": tested,
                    "selected": selected is not None,
                    "promoted": False,
                    "forward_true_safe": False,
                    "forward_actions": np.nan,
                    "forward_observed_share": np.nan,
                }
                if selected is not None:
                    proposals = first_actions_by_score(
                        forward, "score", selected.threshold
                    )
                    action_oracle, resource_oracle = _oracle(proposals)
                    forward_true_safe = bool(
                        action_oracle <= budget and resource_oracle <= budget
                    )
                    maturity_upper = protocol in {
                        "coldguard_full",
                        "action_risk_only",
                        "rowwise_naive",
                    }
                    action_upper, resource_upper, audit_actions = _bound(
                        proposals,
                        protocol=protocol,
                        family_confidence=family_confidence,
                        maturity_upper=maturity_upper,
                    )
                    if protocol == "no_forward_expiry":
                        promoted = True
                    elif protocol == "action_risk_only":
                        promoted = action_upper <= budget
                    else:
                        promoted = (
                            action_upper <= budget and resource_upper <= budget
                        )
                    row.update(
                        {
                            "threshold": selected.threshold,
                            "calibration_actions": selected.calibration_actions,
                            "calibration_observed_actions": (
                                selected.calibration_observed_actions
                            ),
                            "calibration_action_upper": (
                                selected.calibration_action_upper
                            ),
                            "calibration_resource_upper": (
                                selected.calibration_resource_upper
                            ),
                            "forward_actions": len(proposals),
                            "forward_audit_actions": audit_actions,
                            "forward_observed_share": (
                                float(proposals["outcome_observed"].mean())
                                if len(proposals)
                                else 0.0
                            ),
                            "forward_action_oracle": action_oracle,
                            "forward_resource_oracle": resource_oracle,
                            "forward_action_upper": action_upper,
                            "forward_resource_upper": resource_upper,
                            "forward_true_safe": forward_true_safe,
                            "promoted": bool(promoted),
                        }
                    )
                rows.append(row)

    detail = pd.DataFrame(rows)
    summary = _summarize(detail)
    detail.to_csv(output_dir / "mechanism_replications.csv", index=False)
    summary.to_csv(output_dir / "mechanism_summary.csv", index=False)
    (output_dir / "report.md").write_text(
        _report(summary, replications), encoding="utf-8"
    )
    design = {
        "seed": seed,
        "replications_per_scenario": replications,
        "scenarios": SCENARIOS,
        "protocols": PROTOCOLS,
        "days_per_panel": days,
        "deployments": deployments,
        "episodes_per_panel": episodes,
        "landmarks_per_episode": landmarks,
        "thresholds": thresholds.tolist(),
        "minimum_observed_actions": minimum_actions,
        "family_confidence": family_confidence,
        "dual_risk_budget": budget,
        "estimand": (
            "conditional expected pooled action and exposure ratios over all "
            "proposals from the selected policy"
        ),
        "scope": (
            "mechanism-removal Monte Carlo stress test; empirical evidence, "
            "not a distribution-free guarantee"
        ),
    }
    (output_dir / "design.json").write_text(
        json.dumps(design, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="artifacts/q2_upgrade/mechanism_stress"
    )
    parser.add_argument("--replications", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--deployments", type=int, default=180)
    parser.add_argument("--episodes", type=int, default=360)
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
