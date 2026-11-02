from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/missingness_diagnostic"
OUTPUT = ROOT / "figures/figure_missingness_diagnostic"
COLORS = {"primary": "#0072B2", "strict": "#D55E00"}
LABELS = {
    "log1p_cold_age": "Cold age (log1p days)",
    "log1p_days_since_last_active": "Time since activity (log1p days)",
    "log1p_prior_active_days_30": "Prior active days (log1p count)",
    "log1p_mem_gib_hours": "Daily allocation (log1p GiB-hours)",
    "log1p_avg_mem_gib_calendar": "Average allocated GiB (log1p)",
    "log1p_params_b": "Parameters (log1p billions)",
    "sm_util": "SM utilization",
    "zero_fraction": "Zero-utilization fraction",
    "predicted_risk_3d": "Predicted 3-day revival risk",
}
ORDER = [
    "predicted_risk_3d",
    "log1p_prior_active_days_30",
    "sm_util",
    "zero_fraction",
    "log1p_mem_gib_hours",
    "log1p_avg_mem_gib_calendar",
    "log1p_days_since_last_active",
    "log1p_cold_age",
    "log1p_params_b",
]


def main() -> None:
    effects = pd.read_csv(ARTIFACT / "standardized_differences.csv")
    counts = pd.read_csv(ARTIFACT / "end_reason_counts.csv")
    models = pd.read_csv(ARTIFACT / "missingness_model_metrics.csv")
    predictions = pd.read_parquet(ARTIFACT / "missingness_oof_predictions.parquet")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    fig = plt.figure(figsize=(190 / 25.4, 132 / 25.4))
    fig.subplots_adjust(left=0.30, right=0.985, bottom=0.12, top=0.86, wspace=0.40, hspace=0.50)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.40, 1.0])
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    primary = effects.loc[
        effects["comparison"].eq("deployment_disappearance_vs_reactivated")
    ].copy()
    y = np.arange(len(ORDER))
    for definition, offset, marker, key, label in [
        ("sm_le_1.0_pct", -0.11, "o", "primary", "Primary"),
        ("sm_le_0.1_pct", 0.11, "s", "strict", "Strict"),
    ]:
        group = primary.loc[primary["cold_definition"].eq(definition)].set_index("feature").loc[ORDER]
        ax_a.errorbar(
            group["smd"],
            y + offset,
            xerr=np.vstack(
                [
                    group["smd"] - group["smd_ci_low"],
                    group["smd_ci_high"] - group["smd"],
                ]
            ),
            fmt=marker,
            color=COLORS[key],
            markerfacecolor=(COLORS[key] if key == "primary" else "white"),
            markeredgecolor=COLORS[key],
            markersize=4.3,
            elinewidth=1.0,
            capsize=2.2,
            label=label,
        )
    ax_a.axvline(0, color="#666666", linewidth=0.8)
    ax_a.set(
        title="Observable separation before endpoint loss",
        xlabel="Standardized mean difference\n(disappearance − reactivation; 95% deployment-bootstrap CI)",
        yticks=y,
        yticklabels=[LABELS[value] for value in ORDER],
    )
    ax_a.invert_yaxis()
    ax_a.legend(frameon=False, loc="lower right")

    reason_order = ["reactivated", "telemetry_gap_then_reappears", "deployment_disappearance"]
    reason_labels = ["Reactivated", "Gap + return", "Disappeared"]
    x = np.arange(len(reason_order))
    width = 0.34
    for definition, shift, key, label, hatch in [
        ("sm_le_1.0_pct", -width / 2, "primary", "Primary", None),
        ("sm_le_0.1_pct", width / 2, "strict", "Strict", "//"),
    ]:
        group = counts.loc[counts["cold_definition"].eq(definition)].set_index("end_reason").loc[reason_order]
        bars = ax_b.bar(
            x + shift,
            100 * group["share"],
            width,
            color=COLORS[key],
            edgecolor="white" if hatch is None else COLORS[key],
            hatch=hatch,
            alpha=0.9,
            label=label,
        )
        for bar, episodes in zip(bars, group["episodes"]):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 1.6,
                f"{int(episodes):,}",
                ha="center",
                va="top",
                fontsize=6.4,
                color="white",
                fontweight="bold",
                zorder=5,
                rotation=90,
            )
    ax_b.set(
        title="Test-period terminal episodes",
        ylabel="Share of analyzed episodes (%)",
        xticks=x,
        xticklabels=reason_labels,
        ylim=(0, 56),
    )
    ax_b.tick_params(axis="x", rotation=12)
    ax_b.legend(frameon=False, ncol=2, loc="upper right")

    for definition, key, label, linestyle in [
        ("sm_le_1.0_pct", "primary", "Primary", "-"),
        ("sm_le_0.1_pct", "strict", "Strict", "--"),
    ]:
        group = predictions.loc[predictions["cold_definition"].eq(definition)]
        fpr, tpr, _ = roc_curve(
            group["target_disappearance"], group["predicted_disappearance"]
        )
        metric = models.loc[models["cold_definition"].eq(definition)].iloc[0]
        ax_c.plot(
            fpr,
            tpr,
            color=COLORS[key],
            linestyle=linestyle,
            linewidth=1.5,
            label=(
                f"{label}: {metric['roc_auc']:.3f} "
                f"[{metric['roc_auc_ci_low']:.3f}, {metric['roc_auc_ci_high']:.3f}]"
            ),
        )
    ax_c.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=0.8)
    ax_c.set(
        title="Grouped-CV predictability",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax_c.legend(
        title="ROC-AUC [95% CI]",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        framealpha=0.96,
        loc="lower right",
    )

    for label, ax in zip(["a", "b", "c"], [ax_a, ax_b, ax_c]):
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Endpoint disappearance is observably selective",
        y=0.965,
        fontsize=10.2,
        fontweight="bold",
    )

    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(
            OUTPUT.with_suffix(f".{suffix}"),
            dpi=600,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)
    png = OUTPUT.with_suffix(".png")
    with Image.open(png) as image:
        image.convert("RGB").save(png, dpi=(600, 600))
    with Image.open(png) as image:
        raster = {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "dpi": image.info.get("dpi"),
        }

    manifest = {
        "figure": "observable endpoint-disappearance diagnostic",
        "cohort": "one terminal cold landmark per episode on decision days 121-177",
        "primary_contrast": "deployment disappearance minus observed next-day reactivation",
        "uncertainty": "95% percentile intervals from 500 logical-deployment cluster bootstrap resamples",
        "model_validation": "five-fold GroupKFold with logical deployments isolated between train and test",
        "source_files": [
            str(ARTIFACT / "end_reason_counts.csv"),
            str(ARTIFACT / "standardized_differences.csv"),
            str(ARTIFACT / "missingness_model_metrics.csv"),
            str(ARTIFACT / "missingness_oof_predictions.parquet"),
        ],
        "raster_metadata": raster,
    }
    OUTPUT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT.with_suffix(".caption.md").write_text(
        "**Observable endpoint-disappearance diagnostic.** (a) Standardized mean "
        "differences at each episode's last visible cold day; negative values indicate "
        "lower values before disappearance than before observed reactivation. Intervals "
        "are 95% percentiles from 500 logical-deployment cluster bootstrap resamples. "
        "(b) Composition of test-period terminal episodes; labels are episode counts. "
        "Administrative trace-end episodes are outside the day-121–177 cohort. (c) "
        "Five-fold out-of-fold ROC curves for a prespecified logistic diagnostic using "
        "the displayed observables, with logical deployments isolated between folds. "
        "The diagnostic targets observable endpoint selection; ColdGuard's extreme "
        "endpoints govern unresolved lifecycle outcomes.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
