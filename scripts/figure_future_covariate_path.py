from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/future_covariate_path"
OUTPUT = ROOT / "figures/figure_future_covariate_path"
COLORS = {
    "frozen": "#000000",
    "rolling_exact": "#0072B2",
    "fast_decay": "#D55E00",
    "robust_max": "#009E73",
}
LABELS = {
    "frozen": "Frozen",
    "rolling_exact": "Exact rolling",
    "fast_decay": "Fast decay",
    "robust_max": "Robust max",
}
PATHS = ["frozen", "rolling_exact", "fast_decay", "robust_max"]
MARKERS = {"frozen": "o", "rolling_exact": "s", "fast_decay": "^", "robust_max": "D"}
LINESTYLES = {"frozen": ":", "rolling_exact": "-", "fast_decay": "--", "robust_max": "-."}


def main() -> None:
    metrics = pd.read_csv(ARTIFACT / "predictive_metrics.csv")
    coefficients = pd.read_csv(ARTIFACT / "calibration_coefficients.csv")
    deltas = pd.read_csv(ARTIFACT / "score_path_deltas.csv")
    policies = pd.read_csv(ARTIFACT / "policy_path_audit.csv")
    source = metrics.merge(
        coefficients,
        on=["cold_definition", "horizon", "split", "path"],
        how="left",
        validate="many_to_one",
    )
    source.to_csv(OUTPUT.with_suffix(".source.csv"), index=False)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    fig = plt.figure(figsize=(190 / 25.4, 165 / 25.4))
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.115, top=0.825,
        wspace=0.34, hspace=0.42,
    )
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    test_delta = deltas.loc[deltas["split"].eq("test")]
    for definition, alpha, suffix in [
        ("sm_le_1.0_pct", 1.0, "primary"),
        ("sm_le_0.1_pct", 0.55, "strict"),
    ]:
        for path in ["rolling_exact", "fast_decay", "robust_max"]:
            group = test_delta.loc[
                test_delta["cold_definition"].eq(definition)
                & test_delta["path"].eq(path)
            ].sort_values("horizon")
            ax_a.plot(
                group["horizon"],
                100 * group["mean_absolute_delta"],
                color=COLORS[path],
                marker=MARKERS[path],
                linestyle=LINESTYLES[path],
                alpha=alpha,
                linewidth=1.35,
                markersize=4.2,
                markerfacecolor=(COLORS[path] if definition == "sm_le_1.0_pct" else "white"),
                markeredgecolor=COLORS[path],
            )
    ax_a.set(
        title="Risk-score path sensitivity",
        xlabel="Decision horizon (days)",
        ylabel="Mean absolute change (percentage points)",
        xticks=[1, 3, 7],
    )

    test_metrics = metrics.loc[metrics["split"].eq("test")].copy()
    frozen = test_metrics.loc[test_metrics["path"].eq("frozen"), [
        "cold_definition", "horizon", "brier"
    ]].rename(columns={"brier": "frozen_brier"})
    brier = test_metrics.merge(
        frozen, on=["cold_definition", "horizon"], validate="many_to_one"
    )
    brier["delta_milli"] = 1000 * (brier["brier"] - brier["frozen_brier"])
    for definition, alpha, suffix in [
        ("sm_le_1.0_pct", 1.0, "primary"),
        ("sm_le_0.1_pct", 0.55, "strict"),
    ]:
        for path in ["rolling_exact", "fast_decay", "robust_max"]:
            group = brier.loc[
                brier["cold_definition"].eq(definition) & brier["path"].eq(path)
            ].sort_values("horizon")
            ax_b.plot(
                group["horizon"], group["delta_milli"],
                color=COLORS[path], marker=MARKERS[path],
                linestyle=LINESTYLES[path], alpha=alpha,
                linewidth=1.35, markersize=4.2,
                markerfacecolor=(COLORS[path] if definition == "sm_le_1.0_pct" else "white"),
                markeredgecolor=COLORS[path],
            )
    ax_b.axhline(0, color="#666666", linewidth=0.75)
    ax_b.set(
        title="Predictive loss relative to frozen path",
        xlabel="Decision horizon (days)",
        ylabel="Brier-score change (×10⁻³; lower is better)",
        xticks=[1, 3, 7],
    )

    offsets = {"frozen": -0.18, "rolling_exact": -0.06, "fast_decay": 0.06, "robust_max": 0.18}
    for definition, base_x, label in [
        ("sm_le_1.0_pct", 0.0, "Primary"),
        ("sm_le_0.1_pct", 4.0, "Strict"),
    ]:
        for horizon_index, horizon in enumerate([1, 3, 7]):
            center = base_x + horizon_index
            for path in PATHS:
                row = coefficients.loc[
                    coefficients["cold_definition"].eq(definition)
                    & coefficients["horizon"].eq(horizon)
                    & coefficients["path"].eq(path)
                ].iloc[0]
                x = center + offsets[path]
                ax_c.errorbar(
                    x,
                    row["calibration_slope"],
                    yerr=[
                        [row["calibration_slope"] - row["calibration_slope_ci_low"]],
                        [row["calibration_slope_ci_high"] - row["calibration_slope"]],
                    ],
                    fmt=MARKERS[path],
                    color=COLORS[path],
                    markersize=3.8,
                    markerfacecolor=(COLORS[path] if definition == "sm_le_1.0_pct" else "white"),
                    markeredgecolor=COLORS[path],
                    capsize=2,
                    elinewidth=0.9,
                )
    ax_c.axhline(1.0, color="#666666", linewidth=0.75, linestyle=":")
    ax_c.axvline(3.0, color="#BBBBBB", linewidth=0.7)
    ax_c.set(
        title="Deployment-cluster bootstrap calibration",
        ylabel="Calibration slope (95% CI)",
        xticks=[0, 1, 2, 4, 5, 6],
        xticklabels=["P1", "P3", "P7", "S1", "S3", "S7"],
        xlabel="Cold definition and horizon (P = primary; S = strict)",
    )
    ax_c.set_ylim(0.49, 1.02)

    order = [
        ("sm_le_1.0_pct", 1, "Primary · 1d"),
        ("sm_le_1.0_pct", 3, "Primary · 3d"),
        ("sm_le_1.0_pct", 7, "Primary · 7d"),
        ("sm_le_0.1_pct", 1, "Strict · 1d"),
        ("sm_le_0.1_pct", 3, "Strict · 3d"),
        ("sm_le_0.1_pct", 7, "Strict · 7d"),
    ]
    route_color = {
        "SHADOW_ONLY_UNRESOLVED_BOUND": "#D9ECF5",
        "EXPIRED_RECALIBRATE": "#F4D6CF",
    }
    route_code = {
        "SHADOW_ONLY_UNRESOLVED_BOUND": "S",
        "EXPIRED_RECALIBRATE": "R",
    }
    matrix = np.zeros((len(order), len(PATHS)))
    annotations: list[list[str]] = []
    for row_index, (definition, horizon, _) in enumerate(order):
        group = policies.loc[
            policies["cold_definition"].eq(definition)
            & policies["horizon"].eq(horizon)
        ].set_index("path")
        frozen_actions = int(group.loc["frozen", "test_actions"])
        annotation_row = []
        for column_index, path in enumerate(PATHS):
            route = str(group.loc[path, "full_stream_route"])
            matrix[row_index, column_index] = 0 if route.startswith("SHADOW") else 1
            delta = int(group.loc[path, "test_actions"]) - frozen_actions
            annotation_row.append(f"{delta:+d}\n{route_code[route]}")
        annotations.append(annotation_row)
    cmap = mpl.colors.ListedColormap([
        route_color["SHADOW_ONLY_UNRESOLVED_BOUND"],
        route_color["EXPIRED_RECALIBRATE"],
    ])
    ax_d.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=1.5, aspect="auto", interpolation="nearest")
    for i in range(len(order)):
        for j in range(len(PATHS)):
            ax_d.text(j, i, annotations[i][j], ha="center", va="center", fontsize=7.0)
    ax_d.set(
        title="Operational stability",
        xticks=np.arange(len(PATHS)),
        xticklabels=[LABELS[path] for path in PATHS],
        yticks=np.arange(len(order)),
        yticklabels=[item[2] for item in order],
    )
    ax_d.tick_params(axis="x", rotation=15)
    ax_d.set_xticks(np.arange(-0.5, len(PATHS), 1), minor=True)
    ax_d.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax_d.grid(which="minor", color="white", linewidth=1.0)
    ax_d.tick_params(which="minor", bottom=False, left=False)
    ax_d.set_xlabel("Cell: Δactions / route (S = shadow; R = recalibrate)")

    for ax in [ax_a, ax_b, ax_c]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        ax.set_axisbelow(True)

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(
            0.01, 0.98, label, transform=ax.transAxes,
            fontsize=10.0, fontweight="bold", va="top", ha="left",
        )

    legend_handles = [
        Line2D(
            [0], [0], color=COLORS[path], marker=MARKERS[path],
            linestyle=LINESTYLES[path], linewidth=1.35, markersize=4.2,
            label=LABELS[path],
        )
        for path in ["rolling_exact", "fast_decay", "robust_max"]
    ] + [
        Line2D([0], [0], color="#444444", marker="o", linestyle="none",
               markerfacecolor="#444444", label="Primary"),
        Line2D([0], [0], color="#777777", marker="o", linestyle="none",
               markerfacecolor="white", label="Strict"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.895),
        ncol=5, frameon=False, columnspacing=1.4, handlelength=2.3,
    )
    fig.suptitle(
        "Future-covariate paths change scores, not ColdGuard decisions",
        y=0.975, fontsize=10.2,
        fontweight="bold",
    )
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(
            OUTPUT.with_suffix(f".{suffix}"),
            dpi=600,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)

    png_path = OUTPUT.with_suffix(".png")
    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
        rgb.save(png_path, dpi=(600, 600))

    with Image.open(png_path) as image:
        metadata = {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "dpi": image.info.get("dpi"),
        }
    manifest = {
        "figure": "future covariate path sensitivity",
        "source_files": [
            str(ARTIFACT / "predictive_metrics.csv"),
            str(ARTIFACT / "calibration_coefficients.csv"),
            str(ARTIFACT / "score_path_deltas.csv"),
            str(ARTIFACT / "policy_path_audit.csv"),
        ],
        "uncertainty": "95% percentile interval from 200 logical-deployment cluster bootstrap resamples",
        "transformations": [
            "Brier changes are relative to the frozen path within definition and horizon",
            "action changes are relative to frozen-path complete-case actions",
        ],
        "palette": "Okabe-Ito-derived colors with marker and line-style redundancy",
        "raster_metadata": metadata,
    }
    OUTPUT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT.with_suffix(".caption.md").write_text(
        "**Future-covariate path sensitivity.** (a) Mean absolute risk-score change "
        "relative to freezing the 30-day activity count. (b) Test Brier-score change; "
        "negative values improve predictive loss. Primary and strict cold definitions "
        "are distinguished by opacity. (c) Calibration slopes and 95% percentile "
        "intervals from 200 logical-deployment cluster-bootstrap resamples. (d) Change "
        "in complete-case action count relative to the frozen path and the resulting "
        "full-proposal-stream route. Exact rolling updates use observed pre-landmark "
        "activity under continued inactivity; robust max takes the row-wise maximum "
        "risk across all declared paths.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
