from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from coldguard.clustered_risk import dual_clustered_risk_bound
from coldguard.policy import effective_sample_size, wilson_upper_bound


ROOT = Path(__file__).resolve().parents[1]


SCENARIOS = {
    "iid": {"day_sd": 0.0, "deployment_sd": 0.0, "weight_sd": 0.25},
    "day_shock": {"day_sd": 0.9, "deployment_sd": 0.0, "weight_sd": 0.25},
    "repeated_deployment": {"day_sd": 0.0, "deployment_sd": 1.0, "weight_sd": 0.25},
    "two_axis_heavy_weight": {"day_sd": 0.9, "deployment_sd": 1.0, "weight_sd": 1.5},
}


def _one_replication(
    rng: np.random.Generator,
    day_sd: float,
    deployment_sd: float,
    weight_sd: float,
    days: int,
    deployments: int,
    actions_per_day: int,
    confidence: float,
) -> dict[str, bool | float]:
    n = days * actions_per_day
    day = np.repeat(np.arange(days), actions_per_day)
    deployment = rng.integers(0, deployments, size=n)
    day_effect = rng.normal(0.0, day_sd, size=days)
    deployment_effect = rng.normal(0.0, deployment_sd, size=deployments)
    log_weight = rng.normal(0.0, weight_sd, size=n) + 0.25 * day_effect[day]
    weight = np.exp(log_weight)
    logits = -3.0 + day_effect[day] + deployment_effect[deployment] + 0.12 * log_weight
    probability = expit(logits)
    label = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "day": day,
            "app_id": pd.Series(deployment).astype(str),
            "model_id": "m",
            "site_id": "s",
            "pd_role": "r",
            "reactivates_within_horizon": label,
            "remaining_mem_gib_hours": weight,
        }
    )
    bound = dual_clustered_risk_bound(frame, family_confidence=confidence)
    oracle_action = float(probability.mean())
    oracle_resource = float(np.dot(weight, probability) / weight.sum())

    per_risk_confidence = 1.0 - (1.0 - confidence) / 2.0
    naive_action = wilson_upper_bound(
        float(label.sum()), float(n), per_risk_confidence, comparisons=1
    )
    n_eff = effective_sample_size(weight)
    observed_resource = float(np.dot(weight, label) / weight.sum())
    naive_resource = wilson_upper_bound(
        observed_resource * n_eff,
        n_eff,
        per_risk_confidence,
        comparisons=1,
    )
    return {
        "oracle_action": oracle_action,
        "oracle_resource": oracle_resource,
        "cluster_action_cover": bound.action.upper >= oracle_action,
        "cluster_resource_cover": bound.resource.upper >= oracle_resource,
        "cluster_joint_cover": (
            bound.action.upper >= oracle_action and bound.resource.upper >= oracle_resource
        ),
        "naive_action_cover": naive_action >= oracle_action,
        "naive_resource_cover": naive_resource >= oracle_resource,
        "naive_joint_cover": naive_action >= oracle_action and naive_resource >= oracle_resource,
        "cluster_action_width": bound.action.upper - bound.action.estimate,
        "cluster_resource_width": bound.resource.upper - bound.resource.estimate,
        "naive_action_width": naive_action - float(label.mean()),
        "naive_resource_width": naive_resource - observed_resource,
    }


def run(
    output_dir: Path,
    replications: int,
    seed: int,
    days: int,
    deployments: int,
    actions_per_day: int,
    confidence: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for scenario, parameters in SCENARIOS.items():
        for replication in range(replications):
            rows.append(
                {
                    "scenario": scenario,
                    "replication": replication,
                    **_one_replication(
                        rng,
                        **parameters,
                        days=days,
                        deployments=deployments,
                        actions_per_day=actions_per_day,
                        confidence=confidence,
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "coverage_replications.csv", index=False)
    summary = (
        detail.groupby("scenario", observed=True)
        .agg(
            replications=("replication", "count"),
            cluster_action_coverage=("cluster_action_cover", "mean"),
            cluster_resource_coverage=("cluster_resource_cover", "mean"),
            cluster_joint_coverage=("cluster_joint_cover", "mean"),
            naive_action_coverage=("naive_action_cover", "mean"),
            naive_resource_coverage=("naive_resource_cover", "mean"),
            naive_joint_coverage=("naive_joint_cover", "mean"),
            cluster_action_mean_width=("cluster_action_width", "mean"),
            cluster_resource_mean_width=("cluster_resource_width", "mean"),
            naive_action_mean_width=("naive_action_width", "mean"),
            naive_resource_mean_width=("naive_resource_width", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "coverage_summary.csv", index=False)
    metadata = {
        "seed": seed,
        "replications": replications,
        "days": days,
        "deployments": deployments,
        "actions_per_day": actions_per_day,
        "family_confidence": confidence,
        "per_risk_confidence": 1.0 - (1.0 - confidence) / 2.0,
        "oracle": "conditional expected pooled action/resource ratios for each simulated panel",
    }
    (output_dir / "coverage_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/revision_round1/coverage")
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--deployments", type=int, default=200)
    parser.add_argument("--actions-per-day", type=int, default=40)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    run(
        ROOT / args.output_dir,
        args.replications,
        args.seed,
        args.days,
        args.deployments,
        args.actions_per_day,
        args.confidence,
    )


if __name__ == "__main__":
    main()
