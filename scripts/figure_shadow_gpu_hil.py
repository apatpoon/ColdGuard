from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts/shadow_gpu_analysis"
RUN_ROOT = (
    ROOT
    / "artifacts/shadow_gpu_remote/20261030T082536Z/extracted/artifacts/shadow_gpu"
    / "full_run_20261030T082536Z"
)
OUT = ROOT / "figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
DARK = "#222222"
MID = "#666666"
GRID = "#E3E3E3"
MODEL_LABELS = {
    "qwen_0_5b": "Qwen2.5-0.5B",
    "qwen_1_5b": "Qwen2.5-1.5B",
    "qwen_3b": "Qwen2.5-3B",
    "qwen_7b": "Qwen2.5-7B",
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_source() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cycles = pd.read_csv(ANALYSIS / "cycle_metrics.csv")
    measured = cycles[cycles["role"] == "measured"].copy()
    measured["model_label"] = measured["model_id"].map(MODEL_LABELS)
    measured["peak_memory_above_baseline_gib"] = (
        measured["peak_memory_above_baseline_mib"] / 1024.0
    )

    representative_id = "022_measured_b05_qwen_7b"
    run_dir = RUN_ROOT / "runs" / representative_id
    raw = _read_jsonl(run_dir / "raw_metrics.jsonl")[0]
    telemetry = pd.DataFrame(_read_jsonl(run_dir / "telemetry.jsonl"))
    telemetry["relative_time_s"] = telemetry["monotonic_s"] - telemetry["monotonic_s"].iloc[0]
    telemetry["memory_above_baseline_gib"] = (
        telemetry["memory_used_mib"] - float(raw["baseline_memory_mib"])
    ).clip(lower=0) / 1024.0
    events = {
        "representative_run_id": representative_id,
        "stop_requested_s": float(raw["stop_requested_monotonic_s"])
        - float(telemetry["monotonic_s"].iloc[0]),
        "process_exit_s": float(raw["process_exit_monotonic_s"])
        - float(telemetry["monotonic_s"].iloc[0]),
        "baseline_confirmed_s": float(raw["baseline_confirmed_monotonic_s"])
        - float(telemetry["monotonic_s"].iloc[0]),
        "baseline_memory_mib": float(raw["baseline_memory_mib"]),
    }
    return measured, telemetry, events


def _interval(summary: pd.DataFrame, model_id: str, metric: str) -> tuple[float, float, float]:
    row = summary.loc[summary["model_id"] == model_id].iloc[0]
    return (
        float(row[f"{metric}_mean"]),
        float(row[f"{metric}_ci95_low"]),
        float(row[f"{metric}_ci95_high"]),
    )


def draw(
    measured: pd.DataFrame,
    summary: pd.DataFrame,
    telemetry: pd.DataFrame,
    events: dict[str, object],
    curve: dict[str, object],
) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
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
        figsize=(174 / 25.4, 126 / 25.4),
        layout="constrained",
    )
    ax_a, ax_b, ax_c, ax_d = axes.flat
    model_order = ["qwen_0_5b", "qwen_1_5b", "qwen_3b", "qwen_7b"]

    # (a) Every complete cycle remains visible; the line is a descriptive calibration.
    for block, marker in zip(range(1, 6), ["o", "s", "^", "D", "v"]):
        part = measured[measured["block"] == block]
        ax_a.scatter(
            part["model_artifact_gib"],
            part["load_total_s"],
            s=22,
            marker=marker,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.8,
            zorder=3,
            label=f"Block {block}",
        )
    x_line = np.linspace(0, 15, 200)
    y_line = float(curve["intercept_s"]) + float(curve["slope_s_per_artifact_gib"]) * x_line
    ax_a.plot(x_line, y_line, color=DARK, linewidth=1.15, zorder=2, label="Descriptive fit")
    for model_id in model_order:
        row = summary[summary["model_id"] == model_id].iloc[0]
        x = float(row["model_artifact_gib"])
        y = float(row["load_total_s_mean"])
        ax_a.annotate(
            MODEL_LABELS[model_id].replace("Qwen2.5-", ""),
            (x, y),
            xytext=(0, 7 if model_id != "qwen_7b" else -13),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax_a.text(
        0.04,
        0.94,
        f"{float(curve['intercept_s']):.3f} + "
        f"{float(curve['slope_s_per_artifact_gib']):.3f} × GiB\n"
        f"$R^2$ = {float(curve['r_squared']):.4f}",
        transform=ax_a.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": GRID},
    )
    ax_a.set(
        xlabel="Local model artifact (GiB)",
        ylabel="End-to-end load time (s)",
        xlim=(0, 15.4),
        ylim=(0, 5.05),
    )
    ax_a.grid(color=GRID, linewidth=0.55)
    ax_a.legend(frameon=False, ncol=2, loc="lower right", handletextpad=0.3, columnspacing=0.7)

    # (b) Peak memory is shown from a true zero with the physical device capacity.
    positions = np.arange(len(model_order))
    peaks = [
        float(summary.loc[summary["model_id"] == model_id, "peak_memory_above_baseline_mib_mean"].iloc[0])
        / 1024.0
        for model_id in model_order
    ]
    bars = ax_b.bar(
        positions,
        peaks,
        width=0.62,
        color=SKY,
        edgecolor=DARK,
        linewidth=0.65,
    )
    for bar, value in zip(bars, peaks):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.55,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    baseline_gib = float(measured["baseline_memory_mib"].median()) / 1024.0
    available_above_baseline_gib = 24.0 - baseline_gib
    ax_b.axhline(
        available_above_baseline_gib,
        color=ORANGE,
        linestyle="--",
        linewidth=1.0,
        label=f"Available above baseline: {available_above_baseline_gib:.2f} GiB",
    )
    ax_b.set(
        xticks=positions,
        xticklabels=["0.5B", "1.5B", "3B", "7B"],
        xlabel="Qwen2.5 model size",
        ylabel="Peak GPU memory above baseline (GiB)",
        ylim=(0, 26.5),
    )
    ax_b.grid(axis="y", color=GRID, linewidth=0.55)
    ax_b.legend(frameon=False, loc="upper left")

    # (c) Two distinct lifecycle primitives with 95% t intervals over five cycles.
    offsets = {"first_request_s": -0.10, "unload_to_baseline_s": 0.10}
    styles = {
        "first_request_s": (BLUE, "o", "First 32-token request"),
        "unload_to_baseline_s": (ORANGE, "s", "Confirmed unload"),
    }
    for metric, offset in offsets.items():
        means, lows, highs = [], [], []
        for model_id in model_order:
            mean, low, high = _interval(summary, model_id, metric)
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        color, marker, label = styles[metric]
        ax_c.errorbar(
            positions + offset,
            means,
            yerr=np.vstack([lows, highs]),
            color=color,
            marker=marker,
            linestyle="none",
            markeredgecolor=DARK,
            markeredgewidth=0.45,
            markersize=5.2,
            capsize=2.6,
            elinewidth=1.0,
            label=label,
        )
    ax_c.set(
        xticks=positions,
        xticklabels=["0.5B", "1.5B", "3B", "7B"],
        xlabel="Qwen2.5 model size",
        ylabel="Elapsed time (s)",
        ylim=(0, 1.08),
    )
    ax_c.grid(axis="y", color=GRID, linewidth=0.55)
    ax_c.legend(frameon=False, loc="upper left")
    ax_c.text(
        0.98,
        0.04,
        "Mean ± 95% t interval; n=5",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=MID,
    )

    # (d) A complete raw lifecycle, including explicit stop and three-sample confirmation.
    ax_d.plot(
        telemetry["relative_time_s"],
        telemetry["memory_above_baseline_gib"],
        color=GREEN,
        linewidth=1.25,
    )
    stop = float(events["stop_requested_s"])
    confirmed = float(events["baseline_confirmed_s"])
    ax_d.axvline(stop, color=ORANGE, linestyle="--", linewidth=1.0, label="Stop requested")
    ax_d.axvline(confirmed, color=DARK, linestyle=":", linewidth=1.0, label="Baseline confirmed")
    ax_d.annotate(
        "14.53 GiB peak",
        xy=(telemetry.loc[telemetry["memory_above_baseline_gib"].idxmax(), "relative_time_s"], telemetry["memory_above_baseline_gib"].max()),
        xytext=(-48, -19),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": MID, "linewidth": 0.7},
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    bracket_y = 1.05
    ax_d.hlines(bracket_y, stop, confirmed, color=MID, linewidth=0.8)
    ax_d.vlines([stop, confirmed], bracket_y - 0.22, bracket_y + 0.22, color=MID, linewidth=0.8)
    ax_d.text(
        stop - 0.12,
        bracket_y + 0.42,
        "0.733 s\nconfirmed unload",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax_d.set(
        xlabel="Time from first telemetry sample (s)",
        ylabel="GPU memory above baseline (GiB)",
        xlim=(0, float(telemetry["relative_time_s"].max()) + 0.12),
        ylim=(0, 16.2),
    )
    ax_d.grid(color=GRID, linewidth=0.55)
    ax_d.legend(frameon=False, loc="upper left")

    for label, ax in zip("abcd", axes.flat):
        ax.text(
            -0.105,
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
    measured, telemetry, events = build_source()
    summary = pd.read_csv(ANALYSIS / "model_summary.csv")
    curve = json.loads((ANALYSIS / "service_curve.json").read_text(encoding="utf-8"))
    measured.to_csv(OUT / "figure_shadow_gpu_hil_cycle_source.csv", index=False)
    telemetry.to_csv(OUT / "figure_shadow_gpu_hil_telemetry_source.csv", index=False)

    fig = draw(measured, summary, telemetry, events, curve)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"figure_shadow_gpu_hil.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)

    caption = (
        "**Figure. Real RTX 3090 lifecycle measurements turn the loading layer into a hardware-calibrated service model.** "
        "(a) Twenty independent warm-host-cache cycles across pinned Qwen2.5 FP16 checkpoints; shapes denote randomized complete blocks, and the line is a descriptive fit rather than a cross-platform performance law. "
        "(b) Peak NVML memory above each run's baseline; the dashed line marks the 24-GiB device capacity after subtracting the measured NVML baseline. "
        "(c) First 32-token request and confirmed unload times; points and error bars are means and 95% Student-t intervals over five complete cycles per model. "
        "(d) Raw 50-ms NVML telemetry for a representative 7B cycle. Confirmed unload is measured from the explicit stop request until three consecutive samples return to baseline; it therefore includes the confirmation window."
    )
    (OUT / "figure_shadow_gpu_hil_caption.md").write_text(caption + "\n", encoding="utf-8")
    alt = (
        "Four-panel RTX 3090 hardware-in-the-loop figure. Total load time rises almost linearly from 1.18 seconds for the 0.5B model to 4.42 seconds for 7B. Peak memory rises from 1.25 to 14.53 GiB and remains below 24 GiB. First-request time increases with model size while confirmed unload remains near 0.74 seconds. A representative 7B memory trace rises to 14.53 GiB and returns to baseline after an explicit stop."
    )
    (OUT / "figure_shadow_gpu_hil_alt.txt").write_text(alt + "\n", encoding="utf-8")
    manifest = {
        "status": "generated; visual inspection required before manuscript insertion",
        "figure_type": "experimental-results",
        "paradigm": "four-panel lifecycle evidence: scaling, capacity, latency, raw trace",
        "size_mm": [174, 126],
        "formats": ["pdf", "svg", "png"],
        "font_min_pt": 8.0,
        "source_data": [
            "figures/figure_shadow_gpu_hil_cycle_source.csv",
            "figures/figure_shadow_gpu_hil_telemetry_source.csv",
            "artifacts/shadow_gpu_analysis/model_summary.csv",
            "artifacts/shadow_gpu_analysis/service_curve.json",
        ],
        "unit_of_analysis": "complete independent process load-infer-unload cycle",
        "uncertainty": "95% Student-t intervals over five measured cycles per model",
        "palette": "Okabe-Ito subset with marker and line-style redundancy",
        "axis_policy": "all quantitative axes include zero; no smoothing",
        "representative_run": events["representative_run_id"],
        "matplotlib": mpl.__version__,
    }
    (OUT / "figure_shadow_gpu_hil_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
