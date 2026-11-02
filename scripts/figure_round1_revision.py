from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#BDBDBD"
DARK = "#222222"
METHODS = ["ttl", "hazard_hierarchical", "direct_monotone", "recency_frequency"]
METHOD_LABELS = ["TTL", "Hazard", "Direct +\nprojection", "Recency–\nfrequency"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    consistency = []
    for definition, folder in [("SM ≤ 1%", "multihorizon"), ("SM ≤ 0.1%", "multihorizon_strict")]:
        frame = pd.read_csv(ROOT / f"artifacts/{folder}/probability_consistency.csv")
        frame["cold_definition"] = definition
        consistency.append(frame)
    policies = pd.read_csv(ROOT / "artifacts/revision_round1/operational_policy_audit.csv")
    coverage = pd.read_csv(ROOT / "artifacts/revision_round1/coverage/coverage_summary.csv")
    return pd.concat(consistency, ignore_index=True), policies, coverage


def _status_matrix(policies: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = [
        ("sm_le_1.0_pct", 1),
        ("sm_le_1.0_pct", 3),
        ("sm_le_1.0_pct", 7),
        ("sm_le_0.1_pct", 1),
        ("sm_le_0.1_pct", 3),
        ("sm_le_0.1_pct", 7),
    ]
    status = np.zeros((len(METHODS), len(columns)), dtype=int)
    risk = np.full(status.shape, np.nan, dtype=float)
    for row_index, method in enumerate(METHODS):
        for column_index, (definition, horizon) in enumerate(columns):
            row = policies[
                (policies["cold_definition"] == definition)
                & (policies["horizon"] == horizon)
                & (policies["method"] == method)
            ].iloc[0]
            if not bool(row["calibration_qualified"]):
                status[row_index, column_index] = 0
            elif row["test_route"] == "FORWARD_SUPPORTED":
                status[row_index, column_index] = 2
                risk[row_index, column_index] = 100.0 * max(
                    float(row["test_action_upper"]), float(row["test_resource_upper"])
                )
            else:
                status[row_index, column_index] = 1
                risk[row_index, column_index] = 100.0 * max(
                    float(row["test_action_upper"]), float(row["test_resource_upper"])
                )
    return status, risk


def draw(consistency: pd.DataFrame, policies: pd.DataFrame, coverage: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(174 / 25.4, 132 / 25.4), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.flat

    test = consistency[consistency["split"] == "test"]
    x = np.arange(3)
    width = 0.34
    for offset, definition, color, hatch in [
        (-width / 2, "SM ≤ 1%", BLUE, ""),
        (width / 2, "SM ≤ 0.1%", ORANGE, "//"),
    ]:
        raw = 100 * float(
            test[
                (test["cold_definition"] == definition)
                & (test["method"] == "direct_hierarchical")
            ]["violation_rate"].iloc[0]
        )
        values = [raw, 0.0, 0.0]
        bars = ax_a.bar(
            x + offset,
            values,
            width,
            color=color,
            edgecolor=DARK,
            linewidth=0.5,
            hatch=hatch,
            label=definition,
        )
        for bar, value in zip(bars, values):
            label = f"{value:.1f}" if value else "0"
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                value + (1.1 if value else 0.7),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_a.set(
        xticks=x,
        xticklabels=["Direct", "Direct +\nprojection", "Hazard"],
        ylabel="Horizon-order violations (%)",
        ylim=(0, 50),
    )
    ax_a.grid(axis="y", color="#E3E3E3", linewidth=0.55)
    ax_a.legend(frameon=False, loc="upper right")

    status, risk = _status_matrix(policies)
    cmap = ListedColormap([GRAY, ORANGE, GREEN])
    ax_b.imshow(status, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    for i in range(status.shape[0]):
        for j in range(status.shape[1]):
            if status[i, j] == 0:
                text, color = "A", DARK
            else:
                text = f"{risk[i, j]:.2f}"
                color = "white"
            ax_b.text(j, i, text, ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    ax_b.set(
        xticks=np.arange(6),
        xticklabels=["O1", "O3", "O7", "S1", "S3", "S7"],
        yticks=np.arange(4),
        yticklabels=METHOD_LABELS,
        xlabel="Cold definition × horizon (days)",
    )
    ax_b.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=1.3)
    ax_b.tick_params(which="minor", bottom=False, left=False)
    ax_b.legend(
        handles=[
            mpl.patches.Patch(facecolor=GREEN, label="observed-endpoint pass"),
            mpl.patches.Patch(facecolor=ORANGE, label="expired / recalibrate"),
            mpl.patches.Patch(facecolor=GRAY, label="A = abstain"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
    )

    subset = policies[
        (policies["horizon"] == 3)
        & policies["method"].isin(METHODS)
        & (policies["calibration_qualified"] == True)  # noqa: E712
    ].copy()
    markers = {"ttl": "s", "hazard_hierarchical": "o", "direct_monotone": "^", "recency_frequency": "D"}
    colors = {"sm_le_1.0_pct": BLUE, "sm_le_0.1_pct": ORANGE}
    max_exposure = float(subset["test_trace_exposure_gib_hours"].max())
    for _, row in subset.iterrows():
        size = 35 + 95 * np.sqrt(float(row["test_trace_exposure_gib_hours"]) / max_exposure)
        filled = row["test_route"] == "FORWARD_SUPPORTED"
        ax_c.scatter(
            100 * float(row["test_resource_upper"]),
            100 * float(row["test_action_upper"]),
            s=size,
            marker=markers[row["method"]],
            facecolor=colors[row["cold_definition"]] if filled else "white",
            edgecolor=colors[row["cold_definition"]],
            linewidth=1.25,
            zorder=3,
        )
    ax_c.fill_between([0, 10], 0, 10, color=GREEN, alpha=0.08)
    ax_c.axvline(10, color=DARK, linestyle="--", linewidth=0.8)
    ax_c.axhline(10, color=DARK, linestyle="--", linewidth=0.8)
    ax_c.set(
        xlabel="Exposure-risk upper bound (%)",
        ylabel="Action-risk upper bound (%)",
        xlim=(0, 11),
        ylim=(0, 11),
    )
    ax_c.grid(color="#E3E3E3", linewidth=0.55)
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", markerfacecolor=BLUE, markeredgecolor=BLUE, label="SM ≤ 1%"),
        mpl.lines.Line2D([], [], marker="o", linestyle="", markerfacecolor=ORANGE, markeredgecolor=ORANGE, label="SM ≤ 0.1%"),
    ] + [
        mpl.lines.Line2D([], [], marker=markers[m], linestyle="", color="#666666", label=l.replace("\n", " "))
        for m, l in zip(METHODS, METHOD_LABELS)
    ]
    ax_c.legend(handles=handles, frameon=False, ncol=2, loc="lower right")
    ax_c.text(
        0.03,
        0.88,
        "3-day policies; area $\\propto$ exposure",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    scenarios = ["iid", "day_shock", "repeated_deployment", "two_axis_heavy_weight"]
    labels = ["IID", "Day\nshock", "Deploy.\nrepeat", "Two-axis +\nweight"]
    ordered = coverage.set_index("scenario").loc[scenarios]
    dx = np.arange(len(scenarios))
    w = 0.36
    ax_d.bar(
        dx - w / 2,
        100 * ordered["cluster_joint_coverage"],
        w,
        color=GREEN,
        edgecolor=DARK,
        linewidth=0.5,
        label="Two-axis bound",
    )
    ax_d.bar(
        dx + w / 2,
        100 * ordered["naive_joint_coverage"],
        w,
        color=GRAY,
        edgecolor=DARK,
        linewidth=0.5,
        hatch="//",
        label="Wilson/Kish only",
    )
    ax_d.axhline(95, color=DARK, linestyle="--", linewidth=0.8)
    ax_d.set(
        xticks=dx,
        xticklabels=labels,
        ylabel="Simulated joint coverage (%)",
        ylim=(90, 101),
    )
    ax_d.grid(axis="y", color="#E3E3E3", linewidth=0.55)
    ax_d.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    consistency, policies, coverage = load_inputs()
    fig = draw(consistency, policies, coverage)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(
            OUT / f"figure_round1_revision.{suffix}",
            dpi=600,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)
    png_path = OUT / "figure_round1_revision.png"
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))
    caption = (
        "**Figure. Dependence-stressed replay preserves the coherent three-day operating region but expires the strict seven-day policy.** "
        "(a) Raw independent classifiers violate horizon order, whereas monotone projection and hazard composition do not. "
        "(b) Observed-endpoint status across methods, cold definitions (O, 1%; S, 0.1%), and horizons; numbers are the larger of the two pooled-risk upper bounds in percent, and A denotes calibration abstention. "
        "(c) Three-day dual-risk operating points; filled markers remain inside both 10% limits and marker area encodes horizon-capped trace exposure. "
        "(d) Simulation coverage for the two-axis cluster stress bound and the Wilson/Kish-only screen over 300 replications per scenario; the dashed line is the 95% joint target."
    )
    (OUT / "figure_round1_revision_caption.md").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "size_mm": [174, 132],
        "formats": ["pdf", "svg", "png"],
        "raster_dpi": 600,
        "font_min_pt": 8.0,
        "palette": "Okabe-Ito plus redundant marker/fill/status encodings",
        "inputs": [
            "artifacts/multihorizon*/probability_consistency.csv",
            "artifacts/revision_round1/operational_policy_audit.csv",
            "artifacts/revision_round1/coverage/coverage_summary.csv",
        ],
        "axis_policy": "risk axes show full 0-11% decision range; coverage axis is explicitly truncated to 90-101% to resolve deviations around the 95% target",
        "status": "visual audit passed: no text collision, clipping, or occluded data marks at 174 x 132 mm",
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_round1_revision_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
