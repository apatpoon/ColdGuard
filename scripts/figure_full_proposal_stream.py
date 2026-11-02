from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/full_proposal_stream"
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
DARK = "#222222"
GRID = "#D9D9D9"
EXCEED = "#F7E4DC"


def _panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    unresolved_actions: int,
    first_crossing: float,
) -> None:
    ordered = frame.sort_values("unresolved_revival_fraction")
    x = 100.0 * ordered["unresolved_revival_fraction"].to_numpy(float)
    action = 100.0 * ordered["action_upper"].to_numpy(float)
    resource = 100.0 * ordered["resource_upper"].to_numpy(float)
    crossing_pct = 100.0 * first_crossing
    equivalent = int(np.ceil(first_crossing * unresolved_actions))

    ax.axvspan(crossing_pct, 100.0, color=EXCEED, alpha=0.72, zorder=0)
    ax.axhline(10.0, color=DARK, linestyle=(0, (3, 2)), linewidth=1.0, zorder=1)
    ax.plot(
        x,
        action,
        color=BLUE,
        linewidth=1.8,
        linestyle="-",
        marker="o",
        markevery=10,
        markersize=3.2,
        label="Action-risk upper bound",
        zorder=3,
    )
    ax.plot(
        x,
        resource,
        color=ORANGE,
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markevery=10,
        markersize=3.0,
        label="Exposure-risk upper bound",
        zorder=3,
    )
    ax.axvline(crossing_pct, color=DARK, linestyle=":", linewidth=1.0, zorder=2)
    ax.text(
        0.03,
        0.94,
        f"First dual-budget crossing: {crossing_pct:.0f}%\n"
        f"≈ {equivalent}/{unresolved_actions} unresolved actions",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": GRID},
    )
    ax.text(
        98,
        10.45,
        "10% budget",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=DARK,
    )
    ax.set(
        title=title,
        xlabel="Assumed revival share among unresolved actions (%)",
        xlim=(0, 100),
        ylim=(0, 25.5),
        xticks=np.arange(0, 101, 20),
        yticks=np.arange(0, 26, 5),
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)


def draw(curves: pd.DataFrame, policies: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(174 / 25.4, 78 / 25.4),
        sharey=True,
        layout="constrained",
    )
    selections = [
        ("sm_le_1.0_pct", "Primary cold definition (SM ≤ 1%)"),
        ("sm_le_0.1_pct", "Strict cold definition (SM ≤ 0.1%)"),
    ]
    source_rows = []
    for panel, (definition, title) in zip(axes, selections):
        curve = curves.loc[
            (curves["cold_definition"] == definition)
            & (curves["horizon"] == 3)
            & (curves["method"] == "hazard_hierarchical")
        ].copy()
        policy = policies.loc[
            (policies["cold_definition"] == definition)
            & (policies["horizon"] == 3)
            & (policies["method"] == "hazard_hierarchical")
        ].iloc[0]
        _panel(
            panel,
            curve,
            title=title,
            unresolved_actions=int(policy["unresolved_actions"]),
            first_crossing=float(policy["dual_bound_first_crossing_fraction"]),
        )
        source_rows.append(curve)
    axes[0].set_ylabel("One-sided risk upper bound (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    for label, ax in zip("ab", axes):
        ax.text(
            -0.10,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            clip_on=False,
        )
    pd.concat(source_rows, ignore_index=True).to_csv(
        OUT / "figure_full_proposal_stream_source.csv", index=False
    )
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(ARTIFACT / "tipping_curves.csv")
    policies = pd.read_csv(ARTIFACT / "policy_bounds.csv")
    fig = draw(curves, policies)
    for suffix in ["pdf", "svg", "png"]:
        output_path = OUT / f"figure_full_proposal_stream.{suffix}"
        fig.savefig(
            output_path,
            dpi=600,
            facecolor="white",
        )
        if suffix == "png":
            with Image.open(output_path) as raster:
                raster.convert("RGB").save(output_path, dpi=(600, 600))
    plt.close(fig)

    caption = (
        "**Figure. Full-proposal-stream sensitivity reroutes both three-day hazard policies to shadow-only operation.** "
        "The frozen proposal stream contains 811 actions with 74 unresolved outcomes under the primary cold definition (a), "
        "and 514 actions with 53 unresolved outcomes under the strict definition (b). Curves recompute the one-sided "
        "two-axis cluster stress bounds while the assumed unresolved revival share increases from zero to one; missing "
        "exposure is coupled from its trace-observed portion to the declared full-horizon persistence envelope. The horizontal "
        "line marks each 10% budget, vertical dotted lines mark the first grid value at which either bound crosses its budget, "
        "and shaded regions are therefore shadow-only under this sensitivity path. Intermediate values are sensitivity "
        "assumptions rather than identified revival probabilities."
    )
    (OUT / "figure_full_proposal_stream_caption.md").write_text(
        caption + "\n", encoding="utf-8"
    )
    manifest = {
        "size_mm": [174, 78],
        "formats": ["pdf", "svg", "png"],
        "raster_dpi": 600,
        "font_min_pt": 7.5,
        "palette": "Okabe-Ito blue/vermillion with solid/dashed lines and circle/square markers",
        "input": "artifacts/revision_round2/full_proposal_stream/tipping_curves.csv",
        "source_data": "figures/figure_full_proposal_stream_source.csv",
        "transformation": (
            "risk upper bounds multiplied by 100; 0.01 unresolved-revival grid; "
            "no smoothing or interpolation"
        ),
        "missing_data": (
            "unresolved labels and exposure shown through the declared coupled sensitivity path"
        ),
        "axis_policy": "common linear 0-25.5% y-axis and 0-100% x-axis across panels",
        "accessibility": (
            "risk dimensions use color plus line style plus marker; caption and source table supplied"
        ),
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_full_proposal_stream_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

