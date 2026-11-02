from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/loading_event_sensitivity"
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
DARK = "#222222"
GRID = "#D9D9D9"


def draw(events: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(174 / 25.4, 82 / 25.4),
        gridspec_kw={"width_ratios": [0.92, 1.15]},
        layout="constrained",
    )

    strict = events.loc[events["cold_definition"] == "sm_le_0.1_pct"].copy()
    styles = {
        "hazard_hierarchical": (BLUE, "o", "ColdGuard hazard"),
        "direct_monotone": (ORANGE, "s", "Projected direct"),
    }
    ax = axes[0]
    for method, (color, marker, label) in styles.items():
        values = np.sort(
            strict.loc[strict["method"] == method, "allocation_proxy_added_s"].to_numpy(float)
        )
        rank = np.arange(1, len(values) + 1)
        ax.vlines(rank, 0, values, color=color, alpha=0.32, linewidth=1.0)
        ax.scatter(rank, values, s=27, color=color, marker=marker, label=f"{label} (n={len(values)})", zorder=3)
    ax.axhline(30, color=DARK, linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(8.9, 30.8, "30-s descriptive threshold", ha="right", va="bottom", fontsize=7.2)
    ax.set(
        title="Strict three-day restoration events",
        xlabel="Within-policy event rank",
        ylabel="Scenario-implied added time (s)",
        xlim=(0.5, 9.5),
        ylim=(0, 40),
        xticks=np.arange(1, 10),
        yticks=np.arange(0, 41, 10),
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    strict_summary = summary.loc[summary["cold_definition"] == "sm_le_0.1_pct"].copy()
    scenarios = [
        "allocation_proxy",
        "fp16_checkpoint",
        "int8_checkpoint",
        "fp16_per_rank_parallel",
    ]
    labels = ["Allocation\nproxy", "FP16\nweights", "INT8\nweights", "FP16 /\nparallel ranks"]
    x = np.arange(len(scenarios), dtype=float)
    offsets = {"hazard_hierarchical": -0.13, "direct_monotone": 0.13}
    for method, (color, marker, label) in styles.items():
        rows = strict_summary.set_index(["method", "scenario"])
        centers = np.array([rows.loc[(method, s), "p95_added_s"] for s in scenarios], dtype=float)
        lows = np.array([rows.loc[(method, s), "loo_p95_min_s"] for s in scenarios], dtype=float)
        highs = np.array([rows.loc[(method, s), "loo_p95_max_s"] for s in scenarios], dtype=float)
        xpos = x + offsets[method]
        ax.vlines(xpos, lows, highs, color=color, linewidth=2.0, alpha=0.55, zorder=2)
        ax.scatter(xpos, centers, color=color, marker=marker, s=31, label=label, zorder=3)
    ax.axhline(30, color=DARK, linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(3.42, 30.8, "30 s", ha="right", va="bottom", fontsize=7.2)
    ax.set(
        title="p95 under alternative transfer volumes",
        xlabel="Transfer-volume assumption",
        ylabel="Scenario-implied p95 added time (s)",
        xlim=(-0.45, 3.45),
        ylim=(0, 40),
        xticks=x,
        xticklabels=labels,
        yticks=np.arange(0, 41, 10),
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.97,
        "Points: full finite-event p95\nBars: leave-one-event-out range",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": GRID},
    )
    ax.text(
        0.50,
        -0.30,
        "Parameter-derived scenarios: hazard 8/9 events; direct 8/8 events",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
    )

    for label, ax in zip("ab", axes):
        ax.text(
            -0.11,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            clip_on=False,
        )
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(ARTIFACT / "event_level_transfer_scenarios.csv")
    summary = pd.read_csv(ARTIFACT / "scenario_summary.csv")
    strict_events = events.loc[events["cold_definition"] == "sm_le_0.1_pct"]
    strict_summary = summary.loc[summary["cold_definition"] == "sm_le_0.1_pct"]
    strict_events.to_csv(OUT / "figure_loading_event_sensitivity_events.csv", index=False)
    strict_summary.to_csv(OUT / "figure_loading_event_sensitivity_summary.csv", index=False)

    fig = draw(events, summary)
    for suffix in ["pdf", "svg", "png"]:
        path = OUT / f"figure_loading_event_sensitivity.{suffix}"
        fig.savefig(path, dpi=600, facecolor="white")
        if suffix == "png":
            with Image.open(path) as raster:
                raster.convert("RGB").save(path, dpi=(600, 600))
    plt.close(fig)

    caption = (
        "**Figure. Event-level loading sensitivity separates selected allocation from weight-transfer assumptions.** "
        "(a) Every observed strict three-day restoration event under the canonical 16-GiB/s plus 2-s service equation; "
        "the deployment-allocation proxy produces one projected-direct event above the descriptive 30-s threshold. "
        "(b) Finite-event p95 under four transfer-volume assumptions. Vertical bars are leave-one-event-out p95 ranges, "
        "not confidence intervals. Parameter-derived scenarios exclude one hazard event with missing parameter count; "
        "the FP16-per-rank scenario divides weight volume by rounded trace-observed average GPU count. All values are "
        "counterfactual service-equation outputs rather than measured production latency."
    )
    (OUT / "figure_loading_event_sensitivity_caption.md").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "size_mm": [174, 82],
        "formats": ["pdf", "svg", "png"],
        "raster_dpi": 600,
        "font_min_pt": 7.1,
        "palette": "Okabe-Ito blue/vermillion with circle/square markers",
        "inputs": [
            "artifacts/revision_round2/loading_event_sensitivity/event_level_transfer_scenarios.csv",
            "artifacts/revision_round2/loading_event_sensitivity/scenario_summary.csv",
        ],
        "source_data": [
            "figures/figure_loading_event_sensitivity_events.csv",
            "figures/figure_loading_event_sensitivity_summary.csv",
        ],
        "transformation": "strict-definition filtering, within-method sort, and plotted stored summary values; no smoothing",
        "missing_data": "parameter-derived points omit one of nine hazard events with missing params_b; coverage printed below panel b",
        "uncertainty": "vertical ranges are deterministic leave-one-event-out sensitivity, not confidence intervals",
        "accessibility": "method uses color and marker-shape redundancy; caption and source tables supplied",
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_loading_event_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
