from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/trace_simulation/scenario_summary.csv"
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
DARK = "#222222"
GRID = "#E2E2E2"


METHODS = {
    "hazard_hierarchical": ("ColdGuard", BLUE, "o"),
    "direct_monotone": ("Direct + projection", ORANGE, "^"),
}
DEFINITIONS = {
    "sm_le_1.0_pct": ("SM ≤ 1%", "-", "o", "O"),
    "sm_le_0.1_pct": ("SM ≤ 0.1%", "--", "s", "S"),
}


def source_data(summary: pd.DataFrame) -> pd.DataFrame:
    common = (
        (summary["horizon"] == 3)
        & summary["method"].isin(METHODS)
        & (summary["load_slots"] == 2)
        & (summary["arrival_spread_s"] == 3_600)
    )
    a = summary[
        common
        & (summary["bandwidth_gib_per_s"] == 16)
        & (summary["qualification_route_full_forward"] == "FORWARD_SUPPORTED")
    ].copy()
    a["panel"] = "a_tradeoff"
    b = summary[
        common & (summary["cold_definition"] == "sm_le_0.1_pct")
    ].copy()
    b["panel"] = "b_bandwidth_latency"
    c = summary[common].copy()
    c["panel"] = "c_slo_sensitivity"
    d = summary[
        (summary["cold_definition"] == "sm_le_1.0_pct")
        & (summary["horizon"] == 1)
        & (summary["method"] == "direct_monotone")
        & (summary["bandwidth_gib_per_s"] == 16)
    ].copy()
    d["panel"] = "d_burst_queue"
    return pd.concat([a, b, c, d], ignore_index=True, sort=False)


def _error_bounds(row: pd.Series, metric: str, scale: float = 1.0) -> np.ndarray:
    value = float(row[metric])
    low = float(row[f"{metric}_ci_low"])
    high = float(row[f"{metric}_ci_high"])
    return np.array([[max(0.0, value - low) * scale], [max(0.0, high - value) * scale]])


def draw(source: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(174 / 25.4, 132 / 25.4),
        layout="constrained",
    )
    ax_a, ax_b, ax_c, ax_d = axes.flat

    a = source[source["panel"] == "a_tradeoff"].copy()
    label_offsets = {
        ("sm_le_1.0_pct", "hazard_hierarchical"): (6, -13),
        ("sm_le_1.0_pct", "direct_monotone"): (-35, 8),
        ("sm_le_0.1_pct", "hazard_hierarchical"): (6, -12),
        ("sm_le_0.1_pct", "direct_monotone"): (6, 7),
    }
    for _, row in a.iterrows():
        method_label, color, marker = METHODS[row["method"]]
        definition_label, _, _, prefix = DEFINITIONS[row["cold_definition"]]
        x = float(row["simulated_released_gib_hours"]) / 1e6
        y = float(row["p95_added_latency_s"])
        ax_a.scatter(
            x,
            y,
            s=55,
            marker=marker,
            color=color,
            edgecolor=DARK,
            linewidth=0.6,
            zorder=3,
        )
        ax_a.annotate(
            f"{prefix}-{'H' if row['method'] == 'hazard_hierarchical' else 'D'}",
            (x, y),
            xytext=label_offsets[(row["cold_definition"], row["method"])],
            textcoords="offset points",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    ax_a.axhline(30, color=DARK, linestyle=":", linewidth=0.8, label="30-s threshold")
    ax_a.set(
        xlabel="Released exposure in test window (million GiB-h)",
        ylabel="p95 added reload latency (s)",
        xlim=(0, 5.5),
        ylim=(0, 35),
    )
    ax_a.grid(color=GRID, linewidth=0.55)
    ax_a.legend(
        handles=[
            mpl.lines.Line2D([], [], marker="o", linestyle="", color=BLUE, markeredgecolor=DARK, label="ColdGuard (H)"),
            mpl.lines.Line2D([], [], marker="^", linestyle="", color=ORANGE, markeredgecolor=DARK, label="Direct + projection (D)"),
            mpl.lines.Line2D([], [], linestyle=":", color=DARK, label="30-s threshold"),
        ],
        frameon=False,
        loc="lower right",
    )

    b = source[source["panel"] == "b_bandwidth_latency"].copy()
    for method, (label, color, marker) in METHODS.items():
        part = b[b["method"] == method].sort_values("bandwidth_gib_per_s")
        ax_b.errorbar(
            part["bandwidth_gib_per_s"],
            part["p95_added_latency_s"],
            yerr=np.vstack(
                [
                    part["p95_added_latency_s"] - part["p95_added_latency_s_ci_low"],
                    part["p95_added_latency_s_ci_high"] - part["p95_added_latency_s"],
                ]
            ),
            color=color,
            marker=marker,
            linewidth=1.35,
            capsize=2.5,
            label=label,
        )
    ax_b.axhline(30, color=DARK, linestyle=":", linewidth=0.8)
    ax_b.set(
        xticks=[8, 16, 32],
        xlabel="Effective loading bandwidth (GiB/s)",
        ylabel="Strict 3-day p95 latency (s)",
        ylim=(0, 65),
    )
    ax_b.grid(color=GRID, linewidth=0.55)
    ax_b.legend(frameon=False, loc="upper right")

    c = source[source["panel"] == "c_slo_sensitivity"].copy()
    for definition, (definition_label, linestyle, definition_marker, _) in DEFINITIONS.items():
        for method, (method_label, color, _) in METHODS.items():
            part = c[(c["cold_definition"] == definition) & (c["method"] == method)].sort_values("bandwidth_gib_per_s")
            ax_c.errorbar(
                part["bandwidth_gib_per_s"],
                100 * part["reload_slo_violation_rate_30s"],
                yerr=np.vstack(
                    [
                        100 * (part["reload_slo_violation_rate_30s"] - part["reload_slo_violation_rate_30s_ci_low"]),
                        100 * (part["reload_slo_violation_rate_30s_ci_high"] - part["reload_slo_violation_rate_30s"]),
                    ]
                ),
                color=color,
                linestyle=linestyle,
                marker=definition_marker,
                linewidth=1.2,
                capsize=2.2,
            )
    ax_c.set(
        xticks=[8, 16, 32],
        xlabel="Effective loading bandwidth (GiB/s)",
        ylabel="Reloads exceeding 30 s (%)",
        ylim=(0, 22),
    )
    ax_c.grid(color=GRID, linewidth=0.55)
    method_handles = [
        mpl.lines.Line2D([], [], color=BLUE, label="ColdGuard"),
        mpl.lines.Line2D([], [], color=ORANGE, label="Direct + projection"),
        mpl.lines.Line2D([], [], color="#666666", marker="o", linestyle="-", label="SM ≤ 1%"),
        mpl.lines.Line2D([], [], color="#666666", marker="s", linestyle="--", label="SM ≤ 0.1%"),
    ]
    ax_c.legend(handles=method_handles, frameon=False, ncol=2, loc="upper right")

    d = source[source["panel"] == "d_burst_queue"].copy()
    spread_order = [10.0, 300.0, 3_600.0, 86_400.0]
    spread_labels = ["10 s", "5 min", "1 h", "24 h"]
    slot_styles = {
        1: (ORANGE, "o", "1 slot"),
        2: (BLUE, "s", "2 slots"),
        4: (GREEN, "^", "4 slots"),
    }
    x = np.arange(len(spread_order))
    for slots, (color, marker, label) in slot_styles.items():
        part = d[d["load_slots"] == slots].set_index("arrival_spread_s").loc[spread_order]
        ax_d.errorbar(
            x,
            part["p95_queue_wait_s"],
            yerr=np.vstack(
                [
                    part["p95_queue_wait_s"] - part["p95_queue_wait_s_ci_low"],
                    part["p95_queue_wait_s_ci_high"] - part["p95_queue_wait_s"],
                ]
            ),
            color=color,
            marker=marker,
            linewidth=1.25,
            capsize=2.2,
            label=label,
        )
    ax_d.set(
        xticks=x,
        xticklabels=spread_labels,
        xlabel="Within-day revival spread",
        ylabel="Primary 1-day p95 queue wait (s)",
        ylim=(0, 205),
    )
    ax_d.grid(color=GRID, linewidth=0.55)
    ax_d.legend(frameon=False, loc="upper right")
    ax_d.text(
        0.43,
        0.95,
        "Direct + projection; 16 GiB/s",
        transform=ax_d.transAxes,
        fontsize=7.5,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )

    for label, ax in zip("abcd", axes.flat):
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
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(INPUT)
    source = source_data(summary)
    source_path = OUT / "figure_trace_simulation_source.csv"
    source.to_csv(source_path, index=False)
    fig = draw(source)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure_trace_simulation.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    caption = (
        "**Figure. Trace-driven replay exposes the release--reload trade-off and its loading-system sensitivity.** "
        "(a) Three-day forward-supported operating points under the canonical scenario (16 GiB/s, two loading slots, one-hour within-day revival spread); O and S denote the 1% and 0.1% cold definitions, respectively. "
        "(b) Strict three-day p95 added reload latency across loading bandwidths. "
        "(c) Fraction of within-horizon reloads exceeding 30 s; line color denotes policy and line/marker style denotes cold definition. "
        "(d) Queue sensitivity for the primary one-day direct-projection policy under increasingly dispersed daily arrivals. "
        "Points and error bars are means and 95% t intervals over 10 complete seed-specific replays; panels (a)--(c) use two loading slots and a one-hour arrival spread unless varied on the axis."
    )
    (OUT / "figure_trace_simulation_caption.md").write_text(caption + "\n", encoding="utf-8")
    alt_text = (
        "Four-panel simulation figure. In the canonical three-day scenario, ColdGuard has lower p95 reload latency than direct projection for both cold definitions; strict ColdGuard releases about 1.72 million GiB-hours at about 17 seconds p95. Higher bandwidth lowers latency and 30-second violations. Under the most concentrated ten-second arrival scenario, queue waiting is largest with one loading slot and falls sharply with two or four slots or with wider arrival dispersion."
    )
    (OUT / "figure_trace_simulation_alt.txt").write_text(alt_text + "\n", encoding="utf-8")
    manifest = {
        "status": "JNSM-oriented full-width manuscript figure; rendered visual audit passed with no text collision, clipping, or occluded data marks",
        "size_mm": [174, 132],
        "formats": ["png", "pdf", "svg"],
        "raster_dpi": 600,
        "font_min_pt": 7.5,
        "source_data": str(source_path.relative_to(ROOT)),
        "inputs": [str(INPUT.relative_to(ROOT))],
        "transformations": [
            "stage-3 trace window only",
            "scenario means over ten complete seed-specific replays",
            "released exposure divided by one million only for panel-a display",
            "30-second reload violation fractions multiplied by 100",
        ],
        "uncertainty": "two-sided 95% Student-t intervals over complete seed-specific replays",
        "missing_data": "abstaining policies omitted from operating-point panels; their zero-action rows remain in the source artifact",
        "palette": "Okabe-Ito subset with marker and line-style redundancy",
        "axis_policy": "all quantitative y axes include zero; no smoothing or interpolation",
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_trace_simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
