#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


BLUE = "#0072B2"
VERMILLION = "#D55E00"
DARK = "#262626"
MID = "#666666"
LIGHT = "#D9D9D9"
PASS = "#B8E0C2"
ABSTAIN = "#E5E5E5"
HORIZONS = [1, 3, 7, 14, 30]
BLOCKS = [180, 210, 240, 270, 300, 330]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics_path = root / "artifacts/chutes_external/predictive_metrics.csv"
    novelty_path = root / "artifacts/chutes_external/novelty_metrics.csv"
    rolling_path = root / "artifacts/chutes_external/rolling_recalibration.csv"
    manifest_path = root / "artifacts/chutes_external/manifest.json"
    output_dir = root / "artifacts/chutes_external/figure"
    figure_dir = root / "paper/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    novelty = pd.read_csv(novelty_path)
    rolling = pd.read_csv(rolling_path)
    test = metrics.loc[
        metrics["window"].eq("test")
        & metrics["method"].isin(["age_hazard", "hierarchical_hazard"])
    ].copy()
    novelty_test = novelty.loc[
        novelty["window"].eq("test")
        & novelty["method"].eq("hierarchical_hazard")
    ].copy()
    hierarchy_roll = rolling.loc[rolling["policy"].eq("hierarchical_hazard")].copy()

    source = pd.concat(
        [
            test.assign(panel="A-B"),
            novelty_test.assign(panel="C"),
            hierarchy_roll.assign(panel="D"),
        ],
        ignore_index=True,
        sort=False,
    )
    source.to_csv(output_dir / "figure_chutes_external_source.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(190 / 25.4, 122 / 25.4),
        layout="constrained",
        gridspec_kw={"height_ratios": [0.92, 1.08]},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    style = {
        "age_hazard": (BLUE, "o", "-", "Cold-age hazard"),
        "hierarchical_hazard": (VERMILLION, "s", "--", "Hierarchical hazard"),
    }
    for method, (color, marker, linestyle, label) in style.items():
        frame = test.loc[test["method"].eq(method)].set_index("horizon_days").loc[HORIZONS]
        ax_a.plot(
            HORIZONS,
            frame["roc_auc"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4.2,
            label=label,
        )
        ax_b.plot(
            HORIZONS,
            frame["average_precision"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4.2,
            label=label,
        )
    prevalence = (
        test.loc[test["method"].eq("age_hazard")]
        .set_index("horizon_days")
        .loc[HORIZONS, "prevalence"]
    )
    ax_a.axhline(0.5, color=MID, linestyle=":", linewidth=1.0, label="Chance")
    ax_a.set(
        xscale="log",
        xticks=HORIZONS,
        xticklabels=[str(value) for value in HORIZONS],
        xlabel="Revival horizon (days)",
        ylabel="ROC-AUC",
        ylim=(0.48, 0.92),
    )
    ax_a.set_title("a  Forward-test discrimination", loc="left", fontweight="bold", pad=7)
    ax_a.legend(frameon=False, loc="lower left")

    ax_b.plot(
        HORIZONS,
        prevalence,
        color=MID,
        marker="x",
        linestyle=":",
        linewidth=1.0,
        markersize=4.0,
        label="Event prevalence",
    )
    ax_b.set(
        xscale="log",
        xticks=HORIZONS,
        xticklabels=[str(value) for value in HORIZONS],
        xlabel="Revival horizon (days)",
        ylabel="PR-AUC",
        ylim=(0.0, 0.34),
    )
    ax_b.set_title("b  Precision under rare revival", loc="left", fontweight="bold", pad=7)
    ax_b.legend(frameon=False, loc="upper left")

    novelty_style = {
        "seen": (BLUE, "o", "-", "Seen in training"),
        "unseen": (VERMILLION, "s", "--", "Unseen in training"),
    }
    for novelty_label, (color, marker, linestyle, label) in novelty_style.items():
        frame = (
            novelty_test.loc[novelty_test["novelty"].eq(novelty_label)]
            .set_index("horizon_days")
            .loc[HORIZONS]
        )
        ax_c.plot(
            HORIZONS,
            frame["average_precision"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4.2,
            label=label,
        )
    ax_c.set(
        xscale="log",
        xticks=HORIZONS,
        xticklabels=[str(value) for value in HORIZONS],
        xlabel="Revival horizon (days)",
        ylabel="Hierarchical PR-AUC",
        ylim=(0.0, 0.75),
    )
    ax_c.set_title("c  Novel-model stress test", loc="left", fontweight="bold", pad=7)
    ax_c.legend(
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.70, 0.27),
    )
    ax_c.text(
        0.02,
        0.95,
        "Test: 94 seen / 1,313 unseen models",
        transform=ax_c.transAxes,
        fontsize=6.7,
        color=MID,
        va="top",
    )

    matrix = np.zeros((len(HORIZONS), len(BLOCKS)), dtype=int)
    labels = np.full(matrix.shape, "ABSTAIN", dtype=object)
    for row_index, horizon in enumerate(HORIZONS):
        for column_index, block in enumerate(BLOCKS):
            row = hierarchy_roll.loc[
                hierarchy_roll["horizon_days"].eq(horizon)
                & hierarchy_roll["test_start_day"].eq(block)
            ].iloc[0]
            if bool(row["selected"]):
                matrix[row_index, column_index] = 1
                labels[row_index, column_index] = f"{100 * row['test_action_risk_ucb']:.1f}%"
    ax_d.imshow(
        matrix,
        cmap=ListedColormap([ABSTAIN, PASS]),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax_d.text(
                column_index,
                row_index,
                labels[row_index, column_index],
                ha="center",
                va="center",
                fontsize=6.5 if matrix[row_index, column_index] else 5.8,
                color=DARK if matrix[row_index, column_index] else MID,
                fontweight="bold" if matrix[row_index, column_index] else "normal",
            )
    ax_d.set(
        xticks=np.arange(len(BLOCKS)),
        xticklabels=[f"{value}–{value + 29}" for value in BLOCKS],
        yticks=np.arange(len(HORIZONS)),
        yticklabels=[str(value) for value in HORIZONS],
        xlabel="Forward decision block (trace day)",
        ylabel="Horizon (days)",
    )
    ax_d.set_title(
        "d  Rolling certificate frontier (hierarchical hazard)",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    ax_d.tick_params(length=0)
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.text(
        0.0,
        -0.19,
        "Green cells: calibration-qualified; text is next-block one-sided 95% risk UCB.",
        transform=ax_d.transAxes,
        fontsize=6.4,
        color=MID,
    )

    for axis in [ax_a, ax_b, ax_c]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=LIGHT, linewidth=0.5)
        axis.set_axisbelow(True)

    png_path = figure_dir / "chutes_external_validation.png"
    pdf_path = figure_dir / "chutes_external_validation.pdf"
    fig.savefig(png_path, dpi=600, metadata={"Title": "Chutes external validation"})
    fig.savefig(pdf_path, metadata={"Title": "Chutes external validation"})
    plt.close(fig)

    caption = (
        "Chutes provides a year-scale independent stress test for revival prediction and "
        "risk-certificate refresh. (a) Forward-test ROC-AUC for the cold-age and hierarchical "
        "discrete hazards across five coherent horizons. (b) PR-AUC together with event "
        "prevalence; the gap indicates enrichment above the rare-event base rate. (c) The "
        "hierarchical model retains useful ranking on unseen models but loses substantial "
        "precision relative to 94 models observed during training. (d) The frozen hierarchical "
        "model is recalibrated from the preceding 60 days before each decision block. Green "
        "cells passed the multiplicity-corrected 5% calibration contract with at least 200 "
        "episode-level actions; cell values are one-sided 95% action-risk upper bounds measured "
        "on the following block. Gray cells are abstentions, not zero risk. Outcomes may mature "
        "after the decision block; calibration uses only labels mature before block entry."
    )
    alt_text = (
        "Four-panel Chutes validation figure. Cold-age hazard ROC-AUC declines from about "
        "0.87 at one day to 0.78 at 30 days; hierarchical ROC-AUC is lower but its short- and "
        "medium-horizon PR-AUC is higher. Hierarchical PR-AUC is much stronger on 94 seen "
        "models than on 1,313 unseen models. A rolling heat map shows the hierarchical policy "
        "qualified first at one and three days, then at seven days, and by trace days 330–359 "
        "at all five horizons; every qualified cell has a next-block risk upper bound below 5%."
    )
    (output_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    (output_dir / "alt_text.txt").write_text(alt_text + "\n", encoding="utf-8")
    manifest = {
        "audience": "manuscript; provisional general two-column figure",
        "publisher_requirements": "pending final live target-journal verification",
        "physical_size_mm": [190, 122],
        "png_dpi": 600,
        "outputs": [str(png_path.relative_to(root)), str(pdf_path.relative_to(root))],
        "source_data": str((output_dir / "figure_chutes_external_source.csv").relative_to(root)),
        "inputs": {
            str(path.relative_to(root)): sha256(path)
            for path in [metrics_path, novelty_path, rolling_path, manifest_path]
        },
        "transformations": [
            "strict chronological test metrics",
            "seen/unseen strata defined only from the training chute set",
            "rolling 60-day calibration with horizon-maturity embargo",
            "gray heat-map cells encode abstention, not zero",
        ],
        "uncertainty": (
            "panels A-C show descriptive point estimates; panel D text shows one-sided "
            "95% Wilson UCB for the single frozen test policy"
        ),
        "missing_data": "right-censored labels excluded horizon-wise; abstentions shown explicitly",
        "software_method_reference": (
            "Kassis et al. (2026), Scientific Agent Skills, arXiv:2609.00065, "
            "https://doi.org/10.48550/arXiv.2609.00065"
        ),
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
