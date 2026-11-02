from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/selection_aware_coverage"
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
DARK = "#222222"
GRID = "#DDDDDD"
SCENARIOS = ["iid", "day_shock", "repeated_deployment", "two_axis_heavy_weight", "two_axis_test_shift"]
SCENARIO_LABELS = ["IID", "Day", "Deploy", "Heavy", "Shift"]


def _draw_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    target: float | None = None,
) -> None:
    x = np.arange(len(SCENARIOS), dtype=float)
    styles = [
        ("fixed_sequence_pointwise", "Pointwise fixed sequence", BLUE, "o", -0.12),
        ("threshold_bonferroni", "Threshold Bonferroni", ORANGE, "s", 0.12),
    ]
    for protocol, label, color, marker, offset in styles:
        part = summary.loc[summary.protocol.eq(protocol)].set_index("scenario").loc[SCENARIOS]
        center = 100 * part[metric].to_numpy(float)
        low = 100 * part[f"{metric}_ci_low"].to_numpy(float)
        high = 100 * part[f"{metric}_ci_high"].to_numpy(float)
        ax.errorbar(
            x + offset,
            center,
            yerr=np.vstack([
                np.maximum(0.0, center - low),
                np.maximum(0.0, high - center),
            ]),
            color=color,
            marker=marker,
            linestyle="none",
            markersize=4.5,
            capsize=2.2,
            linewidth=1.1,
            label=label,
        )
    if target is not None:
        ax.axhline(target, color=DARK, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set(
        title=title,
        ylabel=ylabel,
        xticks=x,
        xticklabels=SCENARIO_LABELS,
        ylim=ylim,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.5)


def draw(summary: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 82 / 25.4), layout="constrained")
    _draw_metric(
        axes[0], summary, "selection_rate", title="Calibration selection", ylabel="Replications selected (%)", ylim=(0, 105)
    )
    _draw_metric(
        axes[1], summary, "forward_true_safe", title="True safety after selection", ylabel="Selected policies within both budgets (%)", ylim=(65, 103)
    )
    _draw_metric(
        axes[2], summary, "forward_joint_coverage", title="Forward bound coverage", ylabel="Joint oracle coverage (%)", ylim=(90, 101.5), target=95
    )
    axes[2].text(0.45, 95.25, "95% target", ha="left", va="bottom", fontsize=6.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    for label, axis in zip("abc", axes):
        axis.text(-0.17, 1.08, label, transform=axis.transAxes, fontsize=10.5, fontweight="bold", va="top", clip_on=False)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ARTIFACT / "selection_summary.csv")
    summary.to_csv(OUT / "figure_selection_aware_coverage_source.csv", index=False)
    fig = draw(summary)
    for suffix in ["pdf", "svg", "png"]:
        path = OUT / f"figure_selection_aware_coverage.{suffix}"
        fig.savefig(path, dpi=600, facecolor="white")
        if suffix == "png":
            with Image.open(path) as raster:
                raster.convert("RGB").save(path, dpi=(600, 600))
    plt.close(fig)
    caption = (
        "**Figure. Full-procedure simulation separates selection, forward safety, and bound coverage.** "
        "Each point summarizes 300 Monte Carlo replications; error bars are 95% Wilson intervals. Every replication generates three landmarks per episode, scans a predeclared 12-threshold conservative-to-permissive sequence, applies the dual-risk bound, freezes the last passing threshold or abstains, and audits that policy on an independently generated forward panel. "
        "Day, Deploy, Heavy, and Shift denote day shock, repeated deployment, two-axis heavy weight, and the same heavy-weight process with a forward logit shift. "
        "Pointwise fixed sequence uses a 95% family level for the two risks at each candidate; threshold Bonferroni additionally allocates error across all 12 candidates. The forward-shift scenario adds 0.90 to test-period outcome logits but not scores. True-safety and coverage denominators include selected replications only; Bonferroni selects 46 heavy-weight and 45 forward-shift replications. This is an empirical stress test, not a finite-sample theorem."
    )
    (OUT / "figure_selection_aware_coverage_caption.md").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "size_mm": [174, 82],
        "formats": ["pdf", "svg", "png"],
        "raster_dpi": 600,
        "font_min_pt": 6.7,
        "input": "artifacts/revision_round2/selection_aware_coverage/selection_summary.csv",
        "source_data": "figures/figure_selection_aware_coverage_source.csv",
        "uncertainty": "95% Wilson intervals over Monte Carlo replications; selected-policy metrics condition on selection",
        "missing_data": "undefined selected-policy rates when no policy is selected; no summary cell has zero selected replications",
        "transformations": "rates multiplied by 100; no smoothing",
        "axis_policy": "panel-specific declared percent limits; point and interval encoding, no bars or area",
        "accessibility": "protocol encoded by color and marker shape; source data and caption supplied",
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_selection_aware_coverage_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
