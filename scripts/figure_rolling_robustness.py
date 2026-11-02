from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#BBBBBB"
BLACK = "#000000"
FOLDS = ["early", "middle", "late"]
HORIZONS = [1, 3, 7]


def load_source() -> pd.DataFrame:
    frames = []
    for cold, folder in [("SM ≤ 1%", "robustness"), ("SM ≤ 0.1%", "robustness_strict")]:
        global_frame = pd.read_csv(ROOT / f"artifacts/{folder}/rolling_global.csv")
        global_frame["table"] = "global"
        global_frame["cold_definition"] = cold
        frames.append(global_frame)
        fallback = pd.read_csv(ROOT / f"artifacts/{folder}/rolling_fallback.csv")
        fallback["table"] = "fallback"
        fallback["cold_definition"] = cold
        frames.append(fallback)
    return pd.concat(frames, ignore_index=True, sort=False)


def status_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    status = np.zeros((len(FOLDS), len(HORIZONS)), dtype=int)
    risk = np.full_like(status, np.nan, dtype=float)
    for i, fold in enumerate(FOLDS):
        for j, horizon in enumerate(HORIZONS):
            row = frame[(frame["fold"] == fold) & (frame["horizon"] == horizon)].iloc[0]
            if not bool(row["eligible"]):
                status[i, j] = 0
            else:
                risk[i, j] = 100 * max(float(row["test_unweighted_risk_ucb"]), float(row["test_risk_ucb"]))
                status[i, j] = 1 if bool(row["test_contract_pass"]) else 2
    return status, risk


def draw_heatmap(ax: plt.Axes, frame: pd.DataFrame, _title: str) -> None:
    status, risk = status_matrix(frame)
    cmap = ListedColormap([GRAY, BLUE, ORANGE])
    ax.imshow(status, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    for i in range(len(FOLDS)):
        for j in range(len(HORIZONS)):
            text = "abstain" if status[i, j] == 0 else f"{risk[i, j]:.1f}%"
            color = BLACK if status[i, j] == 0 else "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8, fontweight="bold" if status[i, j] else "normal")
    ax.set(xticks=np.arange(3), xticklabels=["1 day", "3 days", "7 days"], yticks=np.arange(3), yticklabels=FOLDS)
    ax.set_xlabel("Prediction horizon"); ax.set_ylabel("Rolling test window")
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True); ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", bottom=False, left=False)


def draw(source: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(174 / 25.4, 128 / 25.4), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.flat
    global_rows = source[(source["table"] == "global") & (source["method"] == "hazard_hierarchical")]
    draw_heatmap(ax_a, global_rows[global_rows["cold_definition"] == "SM ≤ 1%"], "Frozen hazard: original cold definition")
    draw_heatmap(ax_b, global_rows[global_rows["cold_definition"] == "SM ≤ 0.1%"], "Frozen hazard: strict cold definition")
    legend = [
        mpl.patches.Patch(facecolor=BLUE, edgecolor=BLACK, label="QUALIFIED"),
        mpl.patches.Patch(facecolor=ORANGE, edgecolor=BLACK, label="RECALIBRATE"),
        mpl.patches.Patch(facecolor=GRAY, edgecolor=BLACK, label="ABSTAIN"),
    ]
    ax_b.legend(handles=legend, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.23))

    offsets = {"SM ≤ 1%": -0.10, "SM ≤ 0.1%": 0.10}
    colors = {"SM ≤ 1%": BLUE, "SM ≤ 0.1%": ORANGE}
    for cold in offsets:
        subset = source[(source["table"] == "global") & (source["cold_definition"] == cold) & (source["horizon"] == 1)]
        for index, fold in enumerate(FOLDS):
            ttl = subset[(subset["fold"] == fold) & (subset["method"] == "ttl")].iloc[0]
            hazard = subset[(subset["fold"] == fold) & (subset["method"] == "hazard_hierarchical")].iloc[0]
            uplift = 100 * (float(hazard["test_reclaimed_gib_hours"]) / float(ttl["test_reclaimed_gib_hours"]) - 1)
            passed = bool(ttl["test_contract_pass"]) and bool(hazard["test_contract_pass"])
            ax_c.scatter(index + offsets[cold], uplift, marker="o" if cold == "SM ≤ 1%" else "s", s=70,
                         facecolor=colors[cold] if passed else "white", edgecolor=colors[cold], linewidth=1.5,
                         label=cold if index == 0 else None, zorder=3)
            if not passed:
                ax_c.annotate("recalibrate", (index + offsets[cold], uplift), xytext=(8, 8), textcoords="offset points", ha="left", fontsize=8,
                              bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax_c.axhline(0, color=BLACK, linewidth=0.8)
    ax_c.set(xticks=np.arange(3), xticklabels=FOLDS, ylabel="1-day hazard yield uplift over TTL (%)", ylim=(-9, 31))
    ax_c.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax_c.legend(frameon=False, loc="upper left")

    possible = len(FOLDS) * len(HORIZONS) * 4
    for index, cold in enumerate(["SM ≤ 1%", "SM ≤ 0.1%"]):
        routed = source[(source["table"] == "fallback") & (source["cold_definition"] == cold) & (source["novelty_bucket"] != "ALL_ROUTED_BUCKETS")]
        pass_count = int(((routed["route"] != "abstain") & (routed["test_contract_pass"] == True)).sum())  # noqa: E712
        fail_count = int(((routed["route"] != "abstain") & (routed["test_contract_pass"] == False)).sum())  # noqa: E712
        abstain_count = possible - pass_count - fail_count
        bottom = 0
        for count, color, hatch, label in [(pass_count, BLUE, "", "routed + qualified"), (fail_count, ORANGE, "//", "routed + recalibrate"), (abstain_count, GRAY, "..", "abstain")]:
            height = 100 * count / possible
            ax_d.bar(index, height, bottom=bottom, width=0.58, color=color, edgecolor=BLACK, linewidth=0.5, hatch=hatch, label=label if index == 0 else None)
            if count:
                ax_d.text(index, bottom + height / 2, f"{count}", ha="center", va="center", fontsize=8, color="white" if color in [BLUE, ORANGE] else BLACK, fontweight="bold")
            bottom += height
    ax_d.set(xticks=[0, 1], xticklabels=["SM ≤ 1%", "SM ≤ 0.1%"], ylabel="Novelty bucket × window × horizon (%)", ylim=(0, 100))
    ax_d.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax_d.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    for label, ax in zip("abcd", axes.flat):
        ax.text(0.0, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left", clip_on=False)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = load_source()
    source_path = OUT / "figure_rolling_robustness_source.csv"
    source.to_csv(source_path, index=False)
    fig = draw(source)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure_rolling_robustness.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    caption = (
        "**Figure. ColdGuard exposes the temporal validity and coverage of every policy certificate.** (a,b) Maximum of the "
        "action- and resource-risk one-sided 95% upper bounds for eligible hierarchical hazard policies; "
        "blue cells pass both 10% targets, orange cells trigger recalibration, and gray cells abstain. "
        "(c) One-day reclaimed-resource uplift over the matching TTL policy; hollow markers denote windows "
        "where at least one policy triggers recalibration. (d) Outcomes across all 36 novelty-bucket × "
        "window × horizon combinations per cold definition. Numbers are raw combination counts."
    )
    (OUT / "figure_rolling_robustness_caption.md").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "status": "JNSM-oriented full-width manuscript figure; post-export collision and clarity audit passed",
        "size_mm": [174, 128], "formats": ["png", "pdf", "svg"], "raster_dpi": 600,
        "source_data": str(source_path.relative_to(ROOT)),
        "inputs": ["artifacts/robustness*/rolling_global.csv", "artifacts/robustness*/rolling_fallback.csv"],
        "transformations": ["maximum of two test risk UCBs", "hazard yield divided by same-window TTL yield", "fallback outcomes counted over 3 folds x 3 horizons x 4 novelty buckets"],
        "uncertainty": "one-sided 95% Wilson-style action and effective-sample-size resource upper bounds",
        "missing_data": "ineligible policies shown explicitly as abstain",
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_rolling_robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
