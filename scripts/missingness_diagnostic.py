from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from coldguard.config import load_config, resolve_project_path
from coldguard.survival import _deployment_groups, predict_cumulative_incidence


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["app_id", "model_id", "site_id", "pd_role"]
REASONS = ["reactivated", "telemetry_gap_then_reappears", "deployment_disappearance"]
FEATURES = {
    "log1p_cold_age": "Cold age, log1p(days)",
    "log1p_days_since_last_active": "Time since activity, log1p(days)",
    "log1p_prior_active_days_30": "Prior active days, log1p(count)",
    "log1p_mem_gib_hours": "Daily allocation, log1p(GiB-hours)",
    "log1p_avg_mem_gib_calendar": "Average allocated GiB, log1p",
    "log1p_params_b": "Model parameters, log1p(billion)",
    "sm_util": "SM utilization (%)",
    "zero_fraction": "Zero-utilization fraction",
    "predicted_risk_3d": "Predicted 3-day revival risk",
}


def _deployment_id(frame: pd.DataFrame) -> pd.Series:
    return frame[KEYS].astype("string").fillna("<NA>").agg("|".join, axis=1)


def _episode_end_audit(episodes: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    episodes = episodes.copy()
    episodes["deployment_id"] = _deployment_id(episodes)
    daily_ids = _deployment_id(daily)
    last_observed = daily.assign(deployment_id=daily_ids).groupby(
        "deployment_id", observed=True
    )["day"].max()
    episodes["deployment_last_observed_day"] = episodes["deployment_id"].map(last_observed)
    trace_end = int(daily["day"].max())
    reason = pd.Series("reactivated", index=episodes.index, dtype="string")
    censored = ~episodes["reactivated"].astype(bool)
    reason.loc[censored & episodes["end_cold_day"].ge(trace_end)] = "administrative_trace_end"
    reason.loc[
        censored
        & episodes["end_cold_day"].lt(trace_end)
        & episodes["deployment_last_observed_day"].gt(episodes["end_cold_day"] + 1)
    ] = "telemetry_gap_then_reappears"
    reason.loc[
        censored
        & episodes["end_cold_day"].lt(trace_end)
        & episodes["deployment_last_observed_day"].le(episodes["end_cold_day"])
    ] = "deployment_disappearance"
    reason.loc[censored & reason.eq("reactivated")] = "unresolved_gap"
    episodes["end_reason"] = reason
    return episodes


def _transform_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source in [
        "cold_age",
        "days_since_last_active",
        "prior_active_days_30",
        "mem_gib_hours",
        "avg_mem_gib_calendar",
        "params_b",
    ]:
        values = pd.to_numeric(result[source], errors="coerce").clip(lower=0.0)
        result[f"log1p_{source}"] = np.log1p(values)
    return result


def _weighted_mean_variance(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(values) & (weights > 0)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    x = values[valid]
    w = weights[valid]
    mean = float(np.average(x, weights=w))
    variance = float(np.average((x - mean) ** 2, weights=w))
    return mean, variance


def _smd(
    values: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    w = np.ones(len(values), dtype=float) if weights is None else weights
    mean_1, var_1 = _weighted_mean_variance(values[target == 1], w[target == 1])
    mean_0, var_0 = _weighted_mean_variance(values[target == 0], w[target == 0])
    pooled = np.sqrt((var_1 + var_0) / 2.0)
    return float((mean_1 - mean_0) / pooled) if pooled > 0 else float("nan")


def _cluster_bootstrap_smd(
    frame: pd.DataFrame,
    feature: str,
    target: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
    codes, clusters = pd.factorize(frame["deployment_id"], sort=True)
    estimate = _smd(values, target)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        counts = np.bincount(sampled, minlength=len(clusters)).astype(float)
        value = _smd(values, target, counts[codes])
        if np.isfinite(value):
            boot.append(value)
    if len(boot) < int(0.8 * repetitions):
        raise RuntimeError(f"Only {len(boot)}/{repetitions} SMD bootstraps were valid")
    return {
        "smd": estimate,
        "smd_ci_low": float(np.quantile(boot, 0.025)),
        "smd_ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_valid": len(boot),
    }


def _descriptive_rows(frame: pd.DataFrame, definition: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reason in REASONS:
        group = frame.loc[frame["end_reason"].eq(reason)]
        for feature, label in FEATURES.items():
            values = pd.to_numeric(group[feature], errors="coerce")
            rows.append(
                {
                    "cold_definition": definition,
                    "end_reason": reason,
                    "feature": feature,
                    "feature_label": label,
                    "episodes": len(group),
                    "deployments": int(group["deployment_id"].nunique()),
                    "nonmissing": int(values.notna().sum()),
                    "missing_share": float(values.isna().mean()),
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                }
            )
    return rows


def _cv_missingness_model(
    frame: pd.DataFrame,
    definition: str,
    *,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    analysis = frame.loc[
        frame["end_reason"].isin(["reactivated", "deployment_disappearance"])
    ].copy()
    target = analysis["end_reason"].eq("deployment_disappearance").to_numpy(int)
    groups = analysis["deployment_id"].to_numpy(str)
    x = analysis[list(FEATURES)].copy()
    pipeline = Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            list(FEATURES),
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    splitter = GroupKFold(n_splits=5)
    probability = np.full(len(analysis), np.nan, dtype=float)
    fold = np.full(len(analysis), -1, dtype=int)
    for fold_index, (train, test) in enumerate(splitter.split(x, target, groups)):
        pipeline.fit(x.iloc[train], target[train])
        probability[test] = pipeline.predict_proba(x.iloc[test])[:, 1]
        fold[test] = fold_index
    if not np.isfinite(probability).all() or (fold < 0).any():
        raise AssertionError("grouped cross-validation did not score every episode")

    codes, cluster_values = pd.factorize(analysis["deployment_id"], sort=True)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(cluster_values), size=len(cluster_values))
        counts = np.bincount(sampled, minlength=len(cluster_values)).astype(float)
        weights = counts[codes]
        if np.unique(target[weights > 0]).size < 2:
            continue
        boot.append(
            (
                roc_auc_score(target, probability, sample_weight=weights),
                average_precision_score(target, probability, sample_weight=weights),
                brier_score_loss(target, probability, sample_weight=weights),
            )
        )
    values = np.asarray(boot, dtype=float)
    if len(values) < int(0.8 * repetitions):
        raise RuntimeError(f"Only {len(values)}/{repetitions} model bootstraps were valid")
    metrics = {
        "cold_definition": definition,
        "episodes": len(analysis),
        "deployments": int(analysis["deployment_id"].nunique()),
        "disappearances": int(target.sum()),
        "prevalence": float(target.mean()),
        "folds": 5,
        "roc_auc": float(roc_auc_score(target, probability)),
        "roc_auc_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "roc_auc_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "average_precision": float(average_precision_score(target, probability)),
        "average_precision_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "average_precision_ci_high": float(np.quantile(values[:, 1], 0.975)),
        "brier": float(brier_score_loss(target, probability)),
        "brier_ci_low": float(np.quantile(values[:, 2], 0.025)),
        "brier_ci_high": float(np.quantile(values[:, 2], 0.975)),
        "bootstrap_valid": len(values),
    }
    predictions = analysis[
        ["day", "episode_id", "deployment_id", "end_reason"] + KEYS
    ].copy()
    predictions["target_disappearance"] = target
    predictions["predicted_disappearance"] = probability
    predictions["fold"] = fold
    predictions.insert(0, "cold_definition", definition)
    return metrics, predictions


def run(
    config_paths: list[Path],
    output_dir: Path,
    *,
    bootstrap_repetitions: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_frames = []
    count_rows = []
    descriptive_rows = []
    effect_rows = []
    model_rows = []
    prediction_frames = []

    for config_index, config_path in enumerate(config_paths):
        config, project_root = load_config(config_path)
        definition = f"sm_le_{config['cold_definition']['sm_util_max']}_pct"
        episodes = pd.read_parquet(resolve_project_path(project_root, config["paths"]["episodes"]))
        daily = pd.read_parquet(
            resolve_project_path(project_root, config["paths"]["deployment_daily"]),
            columns=KEYS + ["day", "is_cold"],
        )
        landmarks = pd.read_parquet(resolve_project_path(project_root, config["paths"]["landmarks_all"]))
        audited = _episode_end_audit(episodes, daily)
        terminal = landmarks.sort_values(["episode_id", "day"]).groupby(
            "episode_id", observed=True, as_index=False
        ).tail(1)
        terminal = terminal.merge(
            audited[["episode_id", "end_reason", "deployment_id"]],
            on="episode_id",
            how="left",
            validate="one_to_one",
        )
        terminal = terminal.loc[
            terminal["day"].between(121, 177, inclusive="both")
            & terminal["end_reason"].isin(REASONS)
        ].copy()
        model_dir = (
            resolve_project_path(project_root, config["paths"]["multihorizon_artifacts_dir"])
            / "models"
        )
        hazard = joblib.load(model_dir / "hazard_hierarchical.joblib")
        terminal["predicted_risk_3d"] = predict_cumulative_incidence(
            terminal, hazard["encoder"], hazard["model"], 3
        )
        terminal = _transform_features(terminal)
        terminal.insert(0, "cold_definition", definition)
        cohort_frames.append(terminal)

        counts = terminal["end_reason"].value_counts()
        for reason in REASONS:
            group = terminal.loc[terminal["end_reason"].eq(reason)]
            count_rows.append(
                {
                    "cold_definition": definition,
                    "end_reason": reason,
                    "episodes": int(counts.get(reason, 0)),
                    "share": float(counts.get(reason, 0) / len(terminal)),
                    "deployments": int(group["deployment_id"].nunique()),
                }
            )
        descriptive_rows.extend(_descriptive_rows(terminal, definition))

        for contrast_index, comparison in enumerate(
            ["deployment_disappearance", "telemetry_gap_then_reappears"]
        ):
            contrast = terminal.loc[
                terminal["end_reason"].isin(["reactivated", comparison])
            ].copy()
            target = contrast["end_reason"].eq(comparison).to_numpy(int)
            for feature_index, (feature, label) in enumerate(FEATURES.items()):
                effect_rows.append(
                    {
                        "cold_definition": definition,
                        "comparison": f"{comparison}_vs_reactivated",
                        "feature": feature,
                        "feature_label": label,
                        "episodes": len(contrast),
                        "comparison_episodes": int(target.sum()),
                        "reactivated_episodes": int((target == 0).sum()),
                        **_cluster_bootstrap_smd(
                            contrast,
                            feature,
                            target,
                            repetitions=bootstrap_repetitions,
                            seed=(
                                int(config["model"]["random_seed"])
                                + 10000 * config_index
                                + 1000 * contrast_index
                                + feature_index
                            ),
                        ),
                    }
                )

        model_metrics, predictions = _cv_missingness_model(
            terminal,
            definition,
            repetitions=bootstrap_repetitions,
            seed=int(config["model"]["random_seed"]) + 50000 + config_index,
        )
        model_rows.append(model_metrics)
        prediction_frames.append(predictions)

    cohort = pd.concat(cohort_frames, ignore_index=True)
    counts = pd.DataFrame(count_rows)
    descriptives = pd.DataFrame(descriptive_rows)
    effects = pd.DataFrame(effect_rows)
    models = pd.DataFrame(model_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    cohort.to_parquet(output_dir / "terminal_episode_cohort.parquet", index=False)
    counts.to_csv(output_dir / "end_reason_counts.csv", index=False)
    descriptives.to_csv(output_dir / "feature_descriptives.csv", index=False)
    effects.to_csv(output_dir / "standardized_differences.csv", index=False)
    models.to_csv(output_dir / "missingness_model_metrics.csv", index=False)
    predictions.to_parquet(output_dir / "missingness_oof_predictions.parquet", index=False)

    primary = effects.loc[
        effects["comparison"].eq("deployment_disappearance_vs_reactivated")
    ]
    summary = {
        "analysis": "observable informative-disappearance diagnostic",
        "cohort": "one terminal cold landmark per episode, decision days 121-177",
        "primary_contrast": "deployment disappearance versus observed next-day reactivation",
        "bootstrap_unit": "logical deployment",
        "bootstrap_repetitions": bootstrap_repetitions,
        "features": list(FEATURES),
        "largest_absolute_smd_by_definition": {
            str(definition): float(group["smd"].abs().max())
            for definition, group in primary.groupby("cold_definition", observed=True)
        },
        "grouped_cv_roc_auc": {
            str(row.cold_definition): float(row.roc_auc)
            for row in models.itertuples(index=False)
        },
        "interpretation": (
            "Observable associations and grouped-CV predictability diagnose non-random endpoint selection; "
            "they do not identify the missingness mechanism or recover unresolved outcomes."
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
        "--output-dir", default="artifacts/revision_round2/missingness_diagnostic"
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    args = parser.parse_args()
    run(
        [ROOT / value for value in args.configs],
        ROOT / args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )


if __name__ == "__main__":
    main()
