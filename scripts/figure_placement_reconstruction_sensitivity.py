from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/placement_reconstruction_sensitivity"
OUT = ROOT / "figures"
TEMPLATES = ["7B", "13B", "32B", "70B"]
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
DARK = "#222222"
GRID = "#DDDDDD"


def _thousands(value: float, _: float) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def draw(structural: pd.DataFrame, daily: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.5,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 83 / 25.4), layout="constrained")
    x = np.arange(4)

    canonical = structural.loc[np.isclose(structural["runtime_headroom"], 0.20)]
    for variant, label, color, marker, linestyle in [
        ("round_uniform", "Uniform virtual GPUs", BLUE, "o", "-"),
        ("single_instance", "Single / concentrated", ORANGE, "s", "--"),
    ]:
        part = canonical.loc[canonical["variant"].eq(variant)].set_index("template").loc[TEMPLATES]
        axes[0].plot(
            x,
            100 * part["placement_retention"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.3,
        )
    axes[0].set(
        title="Block reconstruction",
        xlabel="Weight template",
        ylabel="Block-packed / global pool (%)",
        xticks=x,
        xticklabels=TEMPLATES,
        ylim=(30, 90),
        yticks=np.arange(30, 91, 10),
    )
    axes[0].grid(axis="y", color=GRID, linewidth=0.5)
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].text(
        0.03,
        0.97,
        "Floor = round = ceil",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
    )

    canonical_daily = daily.loc[np.isclose(daily["runtime_headroom"], 0.20)]
    pivot = canonical_daily.set_index(["source", "template"])
    for column, label, color, marker, linestyle in [
        ("global_pool_replica_days", "Global pool", DARK, "o", "-"),
        ("block_packed_replica_days", "Block-packed", GREEN, "D", "--"),
    ]:
        ratio = [
            100
            * pivot.loc[("trace_daily_reconstructed", template), column]
            / pivot.loc[("repeated_action_day", template), column]
            for template in TEMPLATES
        ]
        axes[1].plot(
            x,
            ratio,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.3,
        )
    axes[1].axhline(100, color="#777777", linestyle=":", linewidth=1.0)
    axes[1].set(
        title="Per-day trace reconstruction",
        xlabel="Weight template",
        ylabel="Daily / repeated-action-day (%)",
        xticks=x,
        xticklabels=TEMPLATES,
        ylim=(97.5, 101.0),
        yticks=[98, 99, 100, 101],
    )
    axes[1].grid(axis="y", color=GRID, linewidth=0.5)
    axes[1].legend(frameon=False, loc="lower right")

    uniform = structural.loc[structural["variant"].eq("round_uniform")]
    for headroom, label, color, marker, linestyle in [
        (0.15, "15% headroom", GREEN, "^", "-."),
        (0.20, "20% headroom", BLUE, "o", "-"),
        (0.30, "30% headroom", ORANGE, "s", "--"),
    ]:
        part = uniform.loc[np.isclose(uniform["runtime_headroom"], headroom)].set_index("template").loc[TEMPLATES]
        axes[2].plot(
            x,
            part["block_packed_replica_days"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.3,
        )
    axes[2].set_yscale("log")
    axes[2].set(
        title="Runtime-headroom sensitivity",
        xlabel="Weight template",
        ylabel="Block-packed replica-days (log)",
        xticks=x,
        xticklabels=TEMPLATES,
    )
    axes[2].yaxis.set_major_formatter(FuncFormatter(_thousands))
    axes[2].grid(axis="y", color=GRID, linewidth=0.5, which="major")
    axes[2].legend(frameon=False, loc="lower left")

    for label, axis in zip("abc", axes):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="top",
            clip_on=False,
        )
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    structural = pd.read_csv(ARTIFACT / "structural_sensitivity.csv")
    daily = pd.read_csv(ARTIFACT / "daily_reconstruction_sensitivity.csv")
    structural.to_csv(OUT / "figure_placement_reconstruction_structural.csv", index=False)
    daily.to_csv(OUT / "figure_placement_reconstruction_daily.csv", index=False)
    fig = draw(structural, daily)
    for suffix in ["pdf", "svg", "png"]:
        path = OUT / f"figure_placement_reconstruction_sensitivity.{suffix}"
        fig.savefig(path, dpi=600, facecolor="white")
        if suffix == "png":
            with Image.open(path) as raster:
                raster.convert("RGB").save(path, dpi=(600, 600))
    plt.close(fig)
    caption = (
        "**Figure. The daily-average weight-residency envelope is stable to time reconstruction but sensitive to block shape for larger tensor-parallel templates.** "
        "All panels use the strict three-day ColdGuard shadow set and FP16/BF16 weights. (a) At 20% headroom, uniform virtual-GPU splitting is compared with an indivisible or fully concentrated instance block; floor, round, and ceiling GPU-count rules coincide because the reconstructed action-day counts are integral. "
        "(b) Reconstructing all 540 active action-days from their own trace records changes global and block-packed replica-days by at most about 1.2% relative to repeating the action-day block. "
        "(c) Block-packed replica-days under 15%, 20%, and 30% runtime headroom. These deterministic daily-average scenarios conserve trace allocation but do not establish instantaneous placement, topology, or communication feasibility."
    )
    (OUT / "figure_placement_reconstruction_sensitivity_caption.md").write_text(
        caption + "\n", encoding="utf-8"
    )
    manifest = {
        "size_mm": [174, 83],
        "formats": ["pdf", "svg", "png"],
        "raster_dpi": 600,
        "font_min_pt": 6.7,
        "inputs": [
            "artifacts/revision_round2/placement_reconstruction_sensitivity/structural_sensitivity.csv",
            "artifacts/revision_round2/placement_reconstruction_sensitivity/daily_reconstruction_sensitivity.csv",
        ],
        "source_data": [
            "figures/figure_placement_reconstruction_structural.csv",
            "figures/figure_placement_reconstruction_daily.csv",
        ],
        "uncertainty": "deterministic structural sensitivity; no confidence intervals",
        "missing_data": "all 540 expected active action-days matched trace records",
        "axis_policy": "panels a and b use declared linear percent axes; panel c uses declared log count axis",
        "accessibility": "color, marker, and line-style redundancy",
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_placement_reconstruction_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
