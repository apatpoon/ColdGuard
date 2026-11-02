#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coldguard.policy import wilson_upper_bound


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "freeinference_7d"
FIGURES = ROOT / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fleet = pd.read_csv(ARTIFACTS / "fleet_audit.csv")
    landmarks = pd.read_parquet(ARTIFACTS / "cold_landmarks.parquet")
    metrics = pd.read_csv(ARTIFACTS / "predictive_metrics.csv")
    test = metrics.loc[
        (metrics["split"] == "test")
        & metrics["method"].isin(["age_hazard", "hierarchical_hazard"])
    ].copy()

    labels = ["1", "2–3", "4–7", "8–11", "12–23", "24+"]
    landmarks["age_bin"] = pd.cut(
        landmarks["cold_age"],
        bins=[0, 1, 3, 7, 11, 23, np.inf],
        labels=labels,
    )
    risk_rows = []
    for label, frame in landmarks.groupby("age_bin", observed=True):
        positives = int(frame["reactivates_within_horizon"].sum())
        trials = int(len(frame))
        risk_rows.append(
            {
                "age_bin": str(label),
                "landmarks": trials,
                "revival_rate": positives / trials,
                "wilson_upper_95": wilson_upper_bound(positives, trials, 0.95),
            }
        )
    risk = pd.DataFrame(risk_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), constrained_layout=True)

    ordered = fleet.sort_values("idle_hours", ascending=True)
    y = np.arange(len(ordered))
    axes[0].barh(y, ordered["active_hours"], color="#2878B5", label="Active")
    axes[0].barh(
        y,
        ordered["idle_hours"],
        left=ordered["active_hours"],
        color="#F2A104",
        label="Idle",
    )
    axes[0].barh(
        y,
        ordered["missing_hours"],
        left=ordered["active_hours"] + ordered["idle_hours"],
        color="#D9D9D9",
        label="Not observed",
    )
    axes[0].set_yticks(y, [value.replace("rtx6000b-", "r6-").replace("qwen36-nvfp4", "qwen36") for value in ordered["server_id"]])
    axes[0].set_xlim(0, 168)
    axes[0].set_xlabel("Hours in the 7-day window")
    axes[0].set_title("a  Independent serving coverage", loc="left", fontweight="bold")
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34), frameon=False)

    x = np.arange(len(risk))
    lower = risk["revival_rate"]
    upper_error = risk["wilson_upper_95"] - lower
    axes[1].bar(x, lower, color="#F2A104", width=0.72)
    axes[1].errorbar(
        x,
        lower,
        yerr=np.vstack([np.zeros(len(risk)), upper_error]),
        fmt="none",
        ecolor="#343434",
        capsize=3,
        linewidth=1,
    )
    for position, row in risk.iterrows():
        axes[1].text(
            position,
            min(1.03, row["wilson_upper_95"] + 0.04),
            f"n={int(row['landmarks'])}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[1].set_xticks(x, risk["age_bin"])
    axes[1].set_ylim(0, 1.18)
    axes[1].set_ylabel("Revival within 6 h")
    axes[1].set_xlabel("Consecutive idle age (h)")
    axes[1].set_title("b  Risk declines after prolonged idle", loc="left", fontweight="bold")

    metric_names = ["roc_auc", "average_precision"]
    metric_labels = ["ROC-AUC", "PR-AUC"]
    positions = np.arange(len(metric_names))
    width = 0.34
    for offset, (method, label, color) in enumerate(
        [
            ("age_hazard", "Cold-age hazard", "#8B8B8B"),
            ("hierarchical_hazard", "ColdGuard hierarchical", "#2878B5"),
        ]
    ):
        row = test.loc[test["method"] == method].iloc[0]
        axes[2].bar(
            positions + (offset - 0.5) * width,
            [row[value] for value in metric_names],
            width,
            label=label,
            color=color,
        )
        for j, value in enumerate([row[name] for name in metric_names]):
            axes[2].text(
                positions[j] + (offset - 0.5) * width,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    prevalence = float(test["prevalence"].iloc[0])
    axes[2].plot(
        [positions[1] - 0.48, positions[1] + 0.48],
        [prevalence, prevalence],
        linestyle="--",
        color="#B33A3A",
        linewidth=1,
        label=f"PR no-skill = {prevalence:.3f}",
    )
    axes[2].set_xticks(positions, metric_labels)
    axes[2].set_ylim(0, 0.86)
    axes[2].set_ylabel("Score on the forward test segment")
    axes[2].set_title("c  Forward-test discrimination", loc="left", fontweight="bold")
    axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), frameon=False)
    axes[2].text(
        0.98,
        0.92,
        "5% risk qualification:\nABSTAIN",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#B33A3A",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#B33A3A"},
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)

    stem = FIGURES / "figure_freeinference_7d"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    source = pd.concat(
        [
            fleet.assign(panel="a_fleet"),
            risk.assign(panel="b_risk"),
            test.assign(panel="c_metrics"),
        ],
        ignore_index=True,
        sort=False,
    )
    source.to_csv(stem.with_name(stem.name + "_source.csv"), index=False)
    caption = (
        "FreeInference seven-day external shadow probe. (a) Hourly coverage and "
        "serving activity for seven deployments. (b) Descriptive six-hour revival "
        "risk by consecutive idle age; landmarks within an episode are dependent. "
        "(c) Forward-test ranking performance. The hierarchical hazard improves "
        "ROC-AUC and PR-AUC over cold age, while the calibrated 5% risk contract "
        "correctly abstains under the short, drifting window."
    )
    stem.with_name(stem.name + "_caption.md").write_text(caption, encoding="utf-8")
    manifest = {
        "inputs": [
            str(ARTIFACTS / "fleet_audit.csv"),
            str(ARTIFACTS / "cold_landmarks.parquet"),
            str(ARTIFACTS / "predictive_metrics.csv"),
        ],
        "outputs": [str(stem.with_suffix(ext)) for ext in [".png", ".pdf", ".svg"]],
        "script": str(Path(__file__).resolve()),
    }
    stem.with_name(stem.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
