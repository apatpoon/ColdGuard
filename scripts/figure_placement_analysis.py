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
INPUT = ROOT / "artifacts/placement_analysis/scenario_summary.csv"
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
DARK = "#222222"
GREY = "#666666"
GRID = "#E2E2E2"
TEMPLATES = ["7B", "13B", "32B", "70B"]


def source_data(summary: pd.DataFrame) -> pd.DataFrame:
    common = (
        (summary["horizon"] == 3)
        & np.isclose(summary["runtime_headroom"], 0.20)
        & summary["method"].isin(["hazard_hierarchical", "direct_monotone"])
    )
    panel_a = summary.loc[
        common
        & summary["cold_definition"].eq("sm_le_0.1_pct")
        & summary["method"].eq("hazard_hierarchical")
        & summary["precision"].eq("FP16/BF16")
    ].copy()
    panel_a["panel"] = "a_nested_bounds"
    panel_b = summary.loc[common & summary["precision"].eq("FP16/BF16")].copy()
    panel_b["panel"] = "b_retention"
    panel_c = summary.loc[
        common
        & summary["cold_definition"].eq("sm_le_0.1_pct")
        & summary["method"].eq("hazard_hierarchical")
    ].copy()
    panel_c["panel"] = "c_precision"
    return pd.concat([panel_a, panel_b, panel_c], ignore_index=True)


def _thousands(value: float, _: float) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def draw(source: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(174 / 25.4, 82 / 25.4),
        layout="constrained",
    )
    x = np.arange(len(TEMPLATES))

    panel_a = source[source["panel"] == "a_nested_bounds"].set_index("template").loc[TEMPLATES]
    stages = [
        ("global_pool_replica_days", "Global pool", DARK, "o", "-"),
        ("site_gpu_pool_replica_days", "Site + GPU type", BLUE, "s", "--"),
        ("cluster_gpu_pool_replica_days", "Cluster + GPU type", GREEN, "D", "-."),
        ("block_packed_replica_days", "Block-packed", ORANGE, "^", ":"),
    ]
    for column, label, color, marker, linestyle in stages:
        axes[0].plot(
            x,
            panel_a[column],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=4.2,
        )
    axes[0].set_yscale("log")
    axes[0].set(
        xticks=x,
        xticklabels=TEMPLATES,
        xlabel="Target weight template",
        ylabel="Deployable replica-days (log scale)",
        title="Nested placement bounds",
    )
    axes[0].yaxis.set_major_formatter(FuncFormatter(_thousands))
    axes[0].grid(color=GRID, linewidth=0.5, which="major")
    axes[0].legend(frameon=False, loc="lower left")

    panel_b = source[source["panel"] == "b_retention"].copy()
    series = [
        ("sm_le_1.0_pct", "hazard_hierarchical", "Primary · ColdGuard", BLUE, "o", "-"),
        ("sm_le_1.0_pct", "direct_monotone", "Primary · Direct", ORANGE, "^", "-"),
        ("sm_le_0.1_pct", "hazard_hierarchical", "Strict · ColdGuard", BLUE, "s", "--"),
        ("sm_le_0.1_pct", "direct_monotone", "Strict · Direct", ORANGE, "D", "--"),
    ]
    for definition, method, label, color, marker, linestyle in series:
        part = panel_b.loc[
            panel_b["cold_definition"].eq(definition) & panel_b["method"].eq(method)
        ].set_index("template").reindex(TEMPLATES)
        axes[1].plot(
            x,
            100 * part["placement_retention"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=4.2,
        )
    axes[1].set(
        xticks=x,
        xticklabels=TEMPLATES,
        xlabel="Target weight template",
        ylabel="Block-packed / global pool (%)",
        ylim=(30, 95),
        yticks=[40, 50, 60, 70, 80, 90],
        title="Capacity surviving fragmentation",
    )
    axes[1].grid(color=GRID, linewidth=0.5)
    axes[1].legend(frameon=False, loc="lower left")

    panel_c = source[source["panel"] == "c_precision"].copy()
    precision_styles = [
        ("FP16/BF16", "FP16/BF16", DARK, "o", "-"),
        ("INT8", "INT8", BLUE, "s", "--"),
        ("INT4", "INT4", GREEN, "^", "-."),
    ]
    for precision, label, color, marker, linestyle in precision_styles:
        part = panel_c.loc[panel_c["precision"].eq(precision)].set_index("template").loc[TEMPLATES]
        axes[2].plot(
            x,
            part["block_packed_replica_days"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=4.2,
        )
    axes[2].set_yscale("log")
    axes[2].set(
        xticks=x,
        xticklabels=TEMPLATES,
        xlabel="Target weight template",
        ylabel="Block-packed replica-days (log scale)",
        title="Precision sensitivity",
    )
    axes[2].yaxis.set_major_formatter(FuncFormatter(_thousands))
    axes[2].grid(color=GRID, linewidth=0.5, which="major")
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
    summary = pd.read_csv(INPUT)
    source = source_data(summary)
    source_path = OUT / "figure_placement_analysis_source.csv"
    source.to_csv(source_path, index=False)
    figure = draw(source)
    for suffix in ["png", "pdf", "svg"]:
        figure.savefig(
            OUT / f"figure_placement_analysis.{suffix}",
            dpi=600,
            facecolor="white",
            transparent=False,
        )
    plt.close(figure)
    png_path = OUT / "figure_placement_analysis.png"
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))
    caption = (
        "**Figure. Placement constraints reduce but do not erase trace-implied deployable memory.** "
        "All panels use the three-day complete-case shadow action sets and 20% runtime headroom. "
        "(a) Strict ColdGuard FP16/BF16 replica-days under progressively stronger constraints: a global fungible pool, same-site/GPU-type pooling, same-cluster/GPU-type pooling, and instance-block placement with distinct virtual GPU blocks for tensor-parallel ranks. "
        "(b) Fraction of the global-pool upper bound retained after block placement for both policies and cold definitions. "
        "(c) Precision sensitivity for strict ColdGuard block placement. Templates use 7B/TP1, 13B/TP1, 32B/TP2, and 70B/TP4. "
        "The deterministic daily-average replay reports structural scenario contrasts, not confidence intervals or measured production throughput."
    )
    (OUT / "figure_placement_analysis_caption.md").write_text(caption + "\n", encoding="utf-8")
    alt = (
        "Three-panel placement analysis. For strict three-day ColdGuard at FP16/BF16, global-pool capacity declines from 4,606 replica-days for 7B to 447 for 70B; block packing retains 3,919, 1,682, 546, and 186 replica-days for 7B, 13B, 32B, and 70B. Retention falls as models grow, from roughly 83 to 88 percent for 7B across policy-definition combinations to roughly 38 to 50 percent for 70B. INT8 and INT4 increase block-packed replica-days at every model size."
    )
    (OUT / "figure_placement_analysis_alt.txt").write_text(alt + "\n", encoding="utf-8")
    manifest = {
        "status": "JNSM-oriented full-width manuscript figure; pending final visual inspection in compiled manuscript",
        "size_mm": [174, 82],
        "formats": ["png", "pdf", "svg"],
        "raster_dpi": 600,
        "font_min_pt": 6.5,
        "source_data": str(source_path.relative_to(ROOT)),
        "inputs": [str(INPUT.relative_to(ROOT))],
        "transformations": [
            "three-day complete-case shadow action sets only",
            "20% runtime headroom",
            "panels a and b use FP16/BF16 two-byte weight footprints",
            "replica-day counts summed over the deterministic daily-average placement replay",
            "retention equals block-packed replica-days divided by global-pool replica-days",
        ],
        "uncertainty": "none displayed; deterministic structural scenario grid rather than independent statistical replications",
        "missing_data": "unsupported policy-definition-horizon routes are absent by design",
        "palette": "Okabe-Ito dark subset with marker and line-style redundancy",
        "axis_policy": "log axes are declared for positive replica-day counts; retention uses an explicit 30-95% position scale",
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_placement_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
