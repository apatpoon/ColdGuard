from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/resource_aware_baseline"
FIGURES = ROOT / "figures"

BLUE = "#3B73B9"
ORANGE = "#E69F00"
GREEN = "#2A9D8F"
RED = "#C84C4C"
DARK = "#263746"
GRAY = "#8795A1"


def key_labels(frame: pd.DataFrame) -> pd.Series:
    definition = frame["cold_definition"].map(
        {"sm_le_1.0_pct": "O", "sm_le_0.1_pct": "S"}
    )
    return definition + frame["horizon"].astype(int).astype(str)


def main() -> None:
    operational = pd.read_csv(ARTIFACT / "operational_policy_audit.csv")
    candidates = pd.read_csv(ARTIFACT / "candidate_sequence_audit.csv")
    bounds = pd.read_csv(ARTIFACT / "policy_bounds.csv")
    order = ["O1", "O3", "O7", "S1", "S3", "S7"]

    cost = operational.loc[operational.method.eq("reload_cost_recency")].copy()
    cost["key"] = key_labels(cost)
    cost["calibration_max_ucb"] = cost[
        ["calibration_action_upper", "calibration_resource_upper"]
    ].max(axis=1)
    cost["complete_case_max_ucb"] = cost[
        ["test_action_upper", "test_resource_upper"]
    ].max(axis=1)
    cost = cost.set_index("key").loc[order].reset_index()

    size_first = (
        candidates.loc[candidates.method.eq("size_benefit_recency")]
        .sort_values("candidate_index")
        .groupby(["cold_definition", "horizon"], as_index=False)
        .first()
    )
    size_first["key"] = key_labels(size_first)
    size_first["first_max_ucb"] = size_first[
        ["calibration_action_upper", "calibration_resource_upper"]
    ].max(axis=1)
    size_first = size_first.set_index("key").loc[order].reset_index()

    cost_bounds = bounds.loc[bounds.method.eq("reload_cost_recency")].copy()
    cost_bounds["key"] = key_labels(cost_bounds)
    cost_bounds["lower_max_ucb"] = cost_bounds[
        ["lower_action_upper", "lower_resource_upper"]
    ].max(axis=1)
    cost_bounds["upper_max_ucb"] = cost_bounds[
        ["upper_action_upper", "upper_resource_upper"]
    ].max(axis=1)
    cost_bounds = cost_bounds.set_index("key").loc[order].reset_index()

    source = cost.merge(
        size_first[["key", "first_max_ucb"]], on="key", validate="one_to_one"
    ).merge(
        cost_bounds[
            [
                "key",
                "lower_max_ucb",
                "upper_max_ucb",
                "proposed_actions",
                "observed_actions",
                "unresolved_actions",
                "dual_bound_first_crossing_fraction",
            ]
        ],
        on="key",
        validate="one_to_one",
    )
    source.to_csv(FIGURES / "figure_resource_aware_baseline_source.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.5), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36

    ax = axes[0, 0]
    ax.bar(
        x - width / 2,
        100 * source["first_max_ucb"],
        width,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.5,
        label="Size-benefit first test",
    )
    ax.bar(
        x + width / 2,
        100 * source["calibration_max_ucb"],
        width,
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
        label="Reload-cost selected",
    )
    ax.axhline(10, color=DARK, linestyle="--", linewidth=1.0, label="10% study budget")
    ax.set_title("(a) Same calibration gate separates the heuristics", loc="left")
    ax.set_ylabel("Larger calibration UCB (%)")
    ax.set_xticks(x, order)
    ax.set_ylim(0, max(25, 100 * source["first_max_ucb"].max() * 1.12))
    ax.legend(frameon=False, ncol=1, loc="upper right")

    ax = axes[0, 1]
    for prefix, color, marker in [("O", BLUE, "o"), ("S", GREEN, "s")]:
        part = source.loc[source.key.str.startswith(prefix)].copy()
        ax.plot(
            part["horizon"],
            100 * part["complete_case_max_ucb"],
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=4.8,
            label="Primary (O)" if prefix == "O" else "Strict (S)",
        )
    ax.axhline(10, color=DARK, linestyle="--", linewidth=1.0)
    ax.set_title("(b) Complete-case support ends after one day", loc="left")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Larger complete-case UCB (%)")
    ax.set_xticks([1, 3, 7])
    ax.set_ylim(0, max(14, 100 * source["complete_case_max_ucb"].max() * 1.18))
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    for prefix, color, marker in [("O", BLUE, "o"), ("S", GREEN, "s")]:
        part = source.loc[source.key.str.startswith(prefix)].copy()
        ax.plot(
            part["horizon"],
            100 * part["lower_max_ucb"],
            color=color,
            marker=marker,
            linestyle="--",
            linewidth=1.4,
            markersize=4.4,
            label=f"{prefix} lower",
        )
        ax.plot(
            part["horizon"],
            100 * part["upper_max_ucb"],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.8,
            markersize=4.8,
            label=f"{prefix} upper",
        )
    ax.axhline(10, color=DARK, linestyle=":", linewidth=1.1, label="10% study budget")
    ax.fill_between([0.7, 7.3], 10, 50, color=RED, alpha=0.055, zorder=0)
    ax.set_title("(c) Full-stream upper endpoints force shadow-only", loc="left")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Larger endpoint UCB (%)")
    ax.set_xticks([1, 3, 7])
    ax.set_xlim(0.7, 7.3)
    ax.set_ylim(0, 50)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax = axes[1, 1]
    observed = source["observed_actions"].to_numpy(float)
    unresolved = source["unresolved_actions"].to_numpy(float)
    ax.bar(x, observed, width=0.66, color=GRAY, label="Observed")
    ax.bar(x, unresolved, width=0.66, bottom=observed, color=ORANGE, label="Unresolved")
    for index, row in source.iterrows():
        ax.text(
            index,
            row["proposed_actions"] + 85,
            f"q*={row['dual_bound_first_crossing_fraction']:.0%}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=DARK,
        )
    ax.set_title("(d) Unresolved outcomes set the promotion boundary", loc="left")
    ax.set_ylabel("Full-stream proposals")
    ax.set_xticks(x, order)
    ax.set_ylim(0, source["proposed_actions"].max() * 1.22)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D7DEE5", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Allocation-aware proposal heuristics remain subject to ColdGuard's identical risk contract",
        fontsize=10.2,
        fontweight="bold",
    )

    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if suffix == "png" else {}
        output_path = FIGURES / f"figure_resource_aware_baseline.{suffix}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            **kwargs,
        )
        if suffix == "png":
            with Image.open(output_path) as raster:
                raster.convert("RGB").save(output_path, dpi=(600, 600))
    plt.close(fig)

    caption = (
        "**Supplementary Fig. S3. Allocation-aware proposal heuristics pass through the identical "
        "ColdGuard contract.** O and S denote the primary (SM ≤ 1%) and strict (SM ≤ 0.1%) cold "
        "definitions. (a) A benefit-oriented score that prioritizes large allocations fails at its "
        "first candidate with at least 200 actions, whereas the reload-cost score selects a passing "
        "calibration candidate in all six definition–horizon combinations. (b) The reload-cost "
        "heuristic passes the complete-case dual-risk endpoint only at one day. (c) Its lower "
        "full-stream endpoints remain below the study budget, but every conservative upper endpoint "
        "crosses it and routes to shadow-only. (d) Bars separate observed and unresolved proposals; "
        "q* is the first coupled unresolved-revival fraction at which either bound crosses 10%. "
        "Allocated GiB is a trace-available size/cost proxy, not identified weight-transfer bytes; "
        "these fixed heuristics are not reproductions of a named caching system."
    )
    (FIGURES / "figure_resource_aware_baseline_caption.md").write_text(
        caption + "\n", encoding="utf-8"
    )
    manifest = {
        "figure": "figure_resource_aware_baseline",
        "panels": 4,
        "source_rows": len(source),
        "formats": ["png", "pdf", "svg"],
        "dpi": 600,
        "notes": "O/S are cold definitions; q* is a declared sensitivity crossing, not an identified probability.",
    }
    (FIGURES / "figure_resource_aware_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
