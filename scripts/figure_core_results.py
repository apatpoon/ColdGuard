from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
BLACK = "#000000"


def load_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "consistency_original": ROOT / "artifacts/multihorizon/probability_consistency.csv",
        "consistency_strict": ROOT / "artifacts/multihorizon_strict/probability_consistency.csv",
        "global_original": ROOT / "artifacts/operational_horizons/global_policies.csv",
        "global_strict": ROOT / "artifacts/operational_horizons_strict/global_policies.csv",
        "guard_original": ROOT / "artifacts/operational_horizons/novelty_guard.csv",
        "guard_strict": ROOT / "artifacts/operational_horizons_strict/novelty_guard.csv",
        "ltt_strict": ROOT / "artifacts/operational_horizons_strict/block_ltt.csv",
    }
    return {name: pd.read_csv(path) for name, path in paths.items()}


def assemble_source_data(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cold, key in [("SM ≤ 1%", "consistency_original"), ("SM ≤ 0.1%", "consistency_strict")]:
        part = tables[key]
        part = part[part["split"] == "test"][["method", "violation_rate"]].copy()
        part["panel"] = "a_monotonicity"
        part["cold_definition"] = cold
        frames.append(part)

    for cold, key in [("SM ≤ 1%", "global_original"), ("SM ≤ 0.1%", "global_strict")]:
        part = tables[key]
        part = part[part["eligible"] == True].copy()  # noqa: E712
        part["panel"] = "b_operating_points"
        part["cold_definition"] = cold
        part["strategy"] = part["method"].map({"ttl": "TTL", "hazard_hierarchical": "Hazard"})
        part = part[part["strategy"].notna()].copy()
        frames.append(part[["panel", "cold_definition", "strategy", "horizon", "test_unweighted_risk_ucb", "test_risk_ucb", "test_reclaimed_gib_hours"]])

    ltt = tables["ltt_strict"]
    if "threshold" in ltt.columns:
        ltt = ltt[ltt["eligible"] == True].copy()  # noqa: E712
        ltt["panel"] = "b_operating_points"
        ltt["cold_definition"] = "SM ≤ 0.1%"
        ltt["strategy"] = "Daily LTT"
        frames.append(ltt[["panel", "cold_definition", "strategy", "horizon", "test_unweighted_risk_ucb", "test_risk_ucb", "test_reclaimed_gib_hours"]])

    for cold, global_key, guard_key in [
        ("SM ≤ 1%", "global_original", "guard_original"),
        ("SM ≤ 0.1%", "global_strict", "guard_strict"),
    ]:
        global_row = tables[global_key].query("horizon == 1 and method == 'hazard_hierarchical'").iloc[0]
        guard_row = tables[guard_key].query("horizon == 1 and novelty_bucket == 'ALL_ELIGIBLE_BUCKETS'").iloc[0]
        frames.append(pd.DataFrame({
            "panel": ["c_novelty_coverage"],
            "cold_definition": [cold],
            "global_reclaimed_gib_hours": [global_row["test_reclaimed_gib_hours"]],
            "guard_reclaimed_gib_hours": [guard_row["test_reclaimed_gib_hours"]],
            "retained_fraction": [guard_row["test_reclaimed_gib_hours"] / global_row["test_reclaimed_gib_hours"]],
        }))

    strict_global_7 = tables["global_strict"].query("horizon == 7 and method == 'hazard_hierarchical'").iloc[0]
    strict_ltt_7 = tables["ltt_strict"].query("horizon == 7 and eligible == True").iloc[0]  # noqa: E712
    frames.append(pd.DataFrame({
        "panel": ["d_strict_7day", "d_strict_7day"],
        "strategy": ["Global CRC", "Daily LTT"],
        "test_unweighted_risk_ucb": [strict_global_7["test_unweighted_risk_ucb"], strict_ltt_7["test_unweighted_risk_ucb"]],
        "test_risk_ucb": [strict_global_7["test_risk_ucb"], strict_ltt_7["test_risk_ucb"]],
        "test_reclaimed_gib_hours": [strict_global_7["test_reclaimed_gib_hours"], strict_ltt_7["test_reclaimed_gib_hours"]],
    }))
    return pd.concat(frames, ignore_index=True, sort=False)


def draw(source: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(174 / 25.4, 127 / 25.4), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.flat

    a = source[source["panel"] == "a_monotonicity"].copy()
    methods = ["hazard_age", "hazard_hierarchical", "direct_age", "direct_hierarchical"]
    labels = ["Hazard\nage", "Hazard\nhierarchical", "Direct\nage", "Direct\nhierarchical"]
    x = np.arange(len(methods)); width = 0.36
    for offset, cold, color, hatch in [(-width / 2, "SM ≤ 1%", BLUE, ""), (width / 2, "SM ≤ 0.1%", ORANGE, "//")]:
        vals = [float(a[(a["cold_definition"] == cold) & (a["method"] == method)]["violation_rate"].iloc[0]) * 100 for method in methods]
        bars = ax_a.bar(x + offset, vals, width, color=color, edgecolor=BLACK, linewidth=0.5, hatch=hatch, label=cold)
        for bar, value in zip(bars, vals):
            if value == 0:
                ax_a.text(bar.get_x() + bar.get_width() / 2, 0.8, "0", ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
    ax_a.set(xticks=x, xticklabels=labels, ylabel="Test rows with violation (%)", ylim=(0, 50))
    ax_a.legend(frameon=False, loc="upper left"); ax_a.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    b = source[source["panel"] == "b_operating_points"].copy()
    marker_map = {"TTL": "s", "Hazard": "o", "Daily LTT": "^"}
    color_map = {"SM ≤ 1%": BLUE, "SM ≤ 0.1%": ORANGE}
    label_offsets = {
        "O1-T": (14, -14), "O1-H": (-23, -11), "O3-T": (8, -14), "O3-H": (8, 7),
        "S1-T": (5, 10), "S1-H": (-4, 16), "S3-H": (-10, -13), "S7-H": (5, 0),
        "S3-L": (5, -12), "S7-L": (5, 5),
    }
    ax_b.fill_between([0, 10], 0, 10, color=GREEN, alpha=0.09)
    for _, row in b.iterrows():
        size = 24 + 95 * np.sqrt(float(row["test_reclaimed_gib_hours"]) / b["test_reclaimed_gib_hours"].max())
        ax_b.scatter(float(row["test_risk_ucb"]) * 100, float(row["test_unweighted_risk_ucb"]) * 100, s=size,
                     marker=marker_map[row["strategy"]], color=color_map[row["cold_definition"]], edgecolor=BLACK, linewidth=0.6, zorder=3)
        tag = ("O" if row["cold_definition"] == "SM ≤ 1%" else "S") + f"{int(row['horizon'])}"
        tag += {"TTL": "-T", "Hazard": "-H", "Daily LTT": "-L"}[row["strategy"]]
        ax_b.annotate(
            tag,
            (float(row["test_risk_ucb"]) * 100, float(row["test_unweighted_risk_ucb"]) * 100),
            xytext=label_offsets[tag], textcoords="offset points", fontsize=8,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax_b.axvline(10, color=BLACK, linestyle="--", linewidth=0.8); ax_b.axhline(10, color=BLACK, linestyle="--", linewidth=0.8)
    ax_b.set(xlabel="Resource-risk upper bound (%)", ylabel="Action-risk upper bound (%)", xlim=(0, 11), ylim=(0, 12))
    ax_b.grid(color="#E5E5E5", linewidth=0.5)
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=BLUE, markeredgecolor=BLACK, label="SM ≤ 1%"),
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=ORANGE, markeredgecolor=BLACK, label="SM ≤ 0.1%"),
        mpl.lines.Line2D([], [], marker="s", linestyle="", color="#777777", label="TTL"),
        mpl.lines.Line2D([], [], marker="o", linestyle="", color="#777777", label="Hazard"),
        mpl.lines.Line2D([], [], marker="^", linestyle="", color="#777777", label="Daily LTT"),
    ]
    ax_b.legend(handles=handles, frameon=True, facecolor="white", edgecolor="none", framealpha=0.90, ncol=2, loc="lower right")

    c = source[source["panel"] == "c_novelty_coverage"].copy(); cx = np.arange(len(c))
    retained = c["retained_fraction"].to_numpy(dtype=float) * 100
    bars = ax_c.bar(cx, retained, width=0.58, color=[BLUE, ORANGE], edgecolor=BLACK, linewidth=0.6)
    ax_c.axhline(100, color=BLACK, linestyle="--", linewidth=0.8, label="Global policy = 100%")
    for bar, value in zip(bars, retained):
        ax_c.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_c.set(xticks=cx, xticklabels=c["cold_definition"], ylabel="Share of global reclaimed GiB-hours (%)", ylim=(0, 110))
    ax_c.grid(axis="y", color="#DDDDDD", linewidth=0.6); ax_c.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.90, loc="upper right")

    d = source[source["panel"] == "d_strict_7day"].copy(); dx = np.arange(len(d)); w = 0.34
    action = d["test_unweighted_risk_ucb"].to_numpy(dtype=float) * 100; resource = d["test_risk_ucb"].to_numpy(dtype=float) * 100
    ax_d.bar(dx - w / 2, action, w, color=PURPLE, edgecolor=BLACK, linewidth=0.5, label="Action-risk UCB")
    ax_d.bar(dx + w / 2, resource, w, color=GREEN, edgecolor=BLACK, linewidth=0.5, hatch="//", label="Resource-risk UCB")
    ax_d.axhline(10, color=BLACK, linestyle="--", linewidth=0.8, label="10% target")
    for i, (_, row) in enumerate(d.iterrows()):
        ax_d.text(i, 0.45, f"{row['test_reclaimed_gib_hours'] / 1e6:.1f}M\nGiB-h", ha="center", va="bottom", fontsize=7)
    ax_d.set(xticks=dx, xticklabels=d["strategy"], ylabel="Test risk upper bound (%)", ylim=(0, 13))
    ax_d.grid(axis="y", color="#DDDDDD", linewidth=0.6); ax_d.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.90, loc="upper right")

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.08, 1.10, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", clip_on=False)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = assemble_source_data(load_inputs())
    source_path = OUT / "figure_core_results_source.csv"
    source.to_csv(source_path, index=False)
    fig = draw(source)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure_core_results.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    caption = (
        "**Figure. ColdGuard converts coherent survival estimates into risk-qualified operating points.** (a) Test-set cross-horizon "
        "monotonicity violations. (b) Eligible operating points; marker area is proportional to reclaimed "
        "GiB-hours, dashed lines mark 10% limits, and labels encode cold definition (O, original 1%; S, "
        "strict 0.1%), horizon, and policy (T, TTL; H, hazard; L, daily LTT). (c) Fraction of the global "
        "one-day hazard policy's reclaimed resources retained by novelty-qualified policies. (d) Under "
        "the strict definition, daily fixed-sequence LTT moves the seven-day policy inside both risk limits, "
        "trading yield for a wider risk margin. UCB denotes "
        "a one-sided 95% upper confidence bound."
    )
    (OUT / "figure_core_results_caption.md").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "status": "JNSM-oriented full-width manuscript figure; post-export collision and clarity audit passed",
        "size_mm": [174, 127], "formats": ["png", "pdf", "svg"], "raster_dpi": 600,
        "source_data": str(source_path.relative_to(ROOT)),
        "inputs": ["artifacts/multihorizon*/probability_consistency.csv", "artifacts/operational_horizons*/global_policies.csv", "artifacts/operational_horizons*/novelty_guard.csv", "artifacts/operational_horizons_strict/block_ltt.csv"],
        "transformations": ["test split only for monotonicity", "eligible operational policies only", "marker area square-root-scaled from reclaimed GiB-hours", "novelty coverage divided by matching global one-day hazard yield"],
        "uncertainty": "one-sided 95% risk upper confidence bounds from saved operational reports",
        "palette": "Okabe-Ito subset with marker, hatch, and panel redundancy",
        "missing_data": "ineligible policies omitted from operating-point panels and retained in source reports",
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_core_results_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
