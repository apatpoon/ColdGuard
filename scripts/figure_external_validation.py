#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
VERMILLION = "#D55E00"
DARK = "#262626"
MID = "#666666"
LIGHT = "#D9D9D9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    agent_metrics_path = root / "artifacts/agentic_external/predictive_metrics.csv"
    agent_policy_path = root / "artifacts/agentic_external/policy_replay.csv"
    gentd_metrics_path = root / "artifacts/gentd_external/predictive_metrics.csv"
    gentd_policy_path = root / "artifacts/gentd_external/policy_replay.csv"
    loading_path = root / "artifacts/gentd_external/loading_consequence_summary.csv"
    inputs = [
        agent_metrics_path,
        agent_policy_path,
        gentd_metrics_path,
        gentd_policy_path,
        loading_path,
    ]
    output_dir = root / "artifacts/external_validation_summary"
    figure_dir = root / "paper/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    agent_metrics = pd.read_csv(agent_metrics_path).query("split == 'test'")
    gentd_metrics = pd.read_csv(gentd_metrics_path).query("split == 'test'")
    methods = ["cold_age", "hierarchical"]
    metric_rows = []
    for dataset, frame in [("Agentic 6 h", agent_metrics), ("GenTD26 1 h", gentd_metrics)]:
        for method in methods:
            row = frame.loc[frame["method"].eq(method)].iloc[0]
            metric_rows.append(
                {
                    "panel": "A",
                    "dataset": dataset,
                    "method": method,
                    "roc_auc": row["roc_auc"],
                    "pr_auc": row["average_precision"],
                }
            )
    metric_source = pd.DataFrame(metric_rows)

    loading = pd.read_csv(loading_path)
    loading_source = loading.assign(panel="B")

    policy_rows = []
    for dataset, path in [
        ("Agentic 6 h", agent_policy_path),
        ("GenTD26 1 h", gentd_policy_path),
    ]:
        frame = pd.read_csv(path)
        if "cal_resource_risk_ucb" in frame:
            frame["governing_ucb"] = frame[
                ["cal_action_risk_ucb", "cal_resource_risk_ucb"]
            ].max(axis=1)
            contract = "max(action, resource)"
        else:
            frame["governing_ucb"] = frame["cal_action_risk_ucb"]
            contract = "action"
        for method, group in frame.groupby("policy", sort=False):
            policy_rows.append(
                {
                    "panel": "C",
                    "dataset": dataset,
                    "method": method,
                    "minimum_governing_ucb": group["governing_ucb"].min(),
                    "contract": contract,
                    "target": 0.05,
                }
            )
    policy_source = pd.DataFrame(policy_rows)

    source = pd.concat(
        [
            metric_source.melt(
                id_vars=["panel", "dataset", "method"],
                value_vars=["roc_auc", "pr_auc"],
                var_name="quantity",
                value_name="value",
            ),
            loading_source.melt(
                id_vars=["panel", "event_type", "revived_episodes"],
                value_vars=["actual_window_rate", "placebo_window_rate"],
                var_name="quantity",
                value_name="value",
            ),
            policy_source.rename(
                columns={"minimum_governing_ucb": "value", "contract": "quantity"}
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    source.to_csv(output_dir / "figure_external_validation_source.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(190 / 25.4, 82 / 25.4), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 1.35])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    # A: common-scale matrix. Values are printed so color is redundant.
    row_order = [
        ("Agentic 6 h", "cold_age", "Agentic · cold age"),
        ("Agentic 6 h", "hierarchical", "Agentic · hierarchical"),
        ("GenTD26 1 h", "cold_age", "GenTD26 · cold age"),
        ("GenTD26 1 h", "hierarchical", "GenTD26 · hierarchical"),
    ]
    matrix = np.array(
        [
            [
                metric_source.query("dataset == @dataset and method == @method").iloc[0][
                    "roc_auc"
                ],
                metric_source.query("dataset == @dataset and method == @method").iloc[0][
                    "pr_auc"
                ],
            ]
            for dataset, method, _ in row_order
        ]
    )
    image = ax_a.imshow(matrix, vmin=0.45, vmax=0.90, cmap="Blues", aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax_a.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value >= 0.68 else DARK,
                fontweight="bold",
                fontsize=7.1,
            )
    ax_a.set(
        xticks=[0, 1],
        xticklabels=["ROC-AUC", "PR-AUC"],
        yticks=np.arange(4),
        yticklabels=[label for _, _, label in row_order],
    )
    ax_a.set_title("a  Forward-test ranking", loc="left", fontweight="bold", pad=8)
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    # B: bars start at zero because their length encodes an event rate.
    x = np.arange(2)
    width = 0.34
    actual = loading.set_index("event_type").loc[
        ["pipeline", "base_model"], "actual_window_rate"
    ]
    placebo = loading.set_index("event_type").loc[
        ["pipeline", "base_model"], "placebo_window_rate"
    ]
    bars_actual = ax_b.bar(
        x - width / 2,
        actual,
        width,
        color=BLUE,
        edgecolor=DARK,
        linewidth=0.5,
        label="Revival window",
    )
    bars_placebo = ax_b.bar(
        x + width / 2,
        placebo,
        width,
        color="white",
        edgecolor=VERMILLION,
        linewidth=1.0,
        hatch="////",
        label="−30 min placebo",
    )
    for bar in bars_actual:
        value = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=DARK,
        )
    for bar in bars_placebo:
        value = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.022,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=DARK,
        )
    ax_b.set(
        xticks=x,
        xticklabels=["Pipeline", "Base model"],
        ylabel="Episodes with loading event",
        ylim=(0, 0.63),
    )
    ax_b.set_title("b  GenTD26 revival consequence", loc="left", fontweight="bold", pad=8)
    ax_b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_b.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=1,
        handlelength=1.5,
    )
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.grid(axis="y", color=LIGHT, linewidth=0.5, zorder=0)
    ax_b.set_axisbelow(True)

    # C: minimum calibration upper bound across each predeclared policy grid.
    display_names = {
        ("Agentic 6 h", "ttl"): "Agentic · TTL",
        ("Agentic 6 h", "cold_age"): "Agentic · cold age",
        ("Agentic 6 h", "hierarchical"): "Agentic · hierarchical",
        ("GenTD26 1 h", "ttl"): "GenTD26 · TTL",
        ("GenTD26 1 h", "cold_age"): "GenTD26 · cold age",
        ("GenTD26 1 h", "hierarchical"): "GenTD26 · hierarchical",
    }
    policy_source["label"] = [
        display_names[(row.dataset, row.method)] for row in policy_source.itertuples()
    ]
    policy_source.sort_values(["dataset", "method"], inplace=True, kind="stable")
    y = np.arange(len(policy_source))[::-1]
    colors = [BLUE if value.startswith("Agentic") else VERMILLION for value in policy_source["label"]]
    markers = ["o" if "Agentic" in value else "s" for value in policy_source["label"]]
    for yi, value, color, marker in zip(
        y,
        policy_source["minimum_governing_ucb"],
        colors,
        markers,
        strict=True,
    ):
        ax_c.hlines(yi, 0, value, color=color, linewidth=1.2)
        ax_c.scatter(value, yi, s=25, color=color, marker=marker, edgecolor="white", linewidth=0.5, zorder=3)
        ax_c.text(value + 0.025, yi, f"{value:.3f}", va="center", fontsize=6.5)
    ax_c.axvline(0.05, color=DARK, linestyle="--", linewidth=1.0)
    ax_c.set(
        yticks=y,
        yticklabels=policy_source["label"],
        xlabel="Minimum governing calibration UCB",
        xlim=(0, 1.02),
    )
    ax_c.set_title("c  No policy clears 5% risk", loc="left", fontweight="bold", pad=8)
    ax_c.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_c.spines[["top", "right", "left"]].set_visible(False)
    ax_c.tick_params(axis="y", length=0)
    ax_c.grid(axis="x", color=LIGHT, linewidth=0.5, zorder=0)
    ax_c.set_axisbelow(True)

    png_path = figure_dir / "external_validation.png"
    pdf_path = figure_dir / "external_validation.pdf"
    fig.savefig(png_path, dpi=600, metadata={"Title": "ColdGuard external validation"})
    fig.savefig(pdf_path, metadata={"Title": "ColdGuard external validation"})
    plt.close(fig)

    caption = (
        "External traces separate predictive transport from operational qualification. "
        "(a) Forward-test ROC-AUC and PR-AUC for the cold-age and hierarchy-informed "
        "risk adapters on the 16-week FreeInference Agentic request trace and the 22.8-h "
        "GenTD26 GPU trace. (b) Among 545 GenTD26 cold episodes with an observed revival, "
        "loading events are concentrated in the [−2, +5] min revival window compared with "
        "a paired equal-width window centered 30 min earlier. (c) The smallest calibration "
        "upper bound found in each predeclared policy family remains above the 5% contract. "
        "Agentic uses action-risk UCB; GenTD26 uses the maximum of action- and memory-weighted "
        "risk UCBs. No external policy is authorized. Point estimates are descriptive; no "
        "confidence intervals are shown in panels a–b."
    )
    alt_text = (
        "Three-panel figure. Agentic ranking is strongest for the hierarchical adapter "
        "(ROC 0.738, PR 0.863), whereas GenTD26 favors the cold-age adapter (ROC 0.636, "
        "PR 0.545). GenTD26 pipeline and base-model loading-event rates are 52.7% and "
        "13.6% near revival versus 0.18% and 0% in paired placebo windows. Every minimum "
        "calibration upper risk bound lies far to the right of the 5% qualification line, "
        "so both traces remain shadow-only."
    )
    (output_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    (output_dir / "alt_text.txt").write_text(alt_text + "\n", encoding="utf-8")
    manifest = {
        "audience": "manuscript; provisional general two-column figure",
        "publisher_requirements": "pending final live target-journal verification",
        "physical_size_mm": [190, 82],
        "png_dpi": 600,
        "outputs": [str(png_path.relative_to(root)), str(pdf_path.relative_to(root))],
        "source_data": str(
            (output_dir / "figure_external_validation_source.csv").relative_to(root)
        ),
        "inputs": {str(path.relative_to(root)): sha256(path) for path in inputs},
        "transformations": [
            "test-split metrics only",
            "loading-event rates copied from paired-window audit",
            "policy-family minimum of governing calibration upper risk bound",
        ],
        "uncertainty": "no interval displayed in panels A-B; panel C displays already-computed one-sided UCBs",
        "missing_data": "none in plotted quantities",
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    agent_hierarchy = metric_source.query(
        "dataset == 'Agentic 6 h' and method == 'hierarchical'"
    ).iloc[0]
    gentd_age = metric_source.query(
        "dataset == 'GenTD26 1 h' and method == 'cold_age'"
    ).iloc[0]
    gentd_hierarchy = metric_source.query(
        "dataset == 'GenTD26 1 h' and method == 'hierarchical'"
    ).iloc[0]
    min_agent = policy_source.loc[
        policy_source["dataset"].eq("Agentic 6 h"), "minimum_governing_ucb"
    ].min()
    min_gentd = policy_source.loc[
        policy_source["dataset"].eq("GenTD26 1 h"), "minimum_governing_ucb"
    ].min()
    chutes_metrics = pd.read_csv(
        root / "artifacts/chutes_external/predictive_metrics.csv"
    )
    chutes_rolling = pd.read_csv(
        root / "artifacts/chutes_external/rolling_recalibration.csv"
    )
    chutes_test = chutes_metrics.loc[
        chutes_metrics["window"].eq("test")
        & chutes_metrics["horizon_days"].eq(7)
    ].set_index("method")
    chutes_age = chutes_test.loc["age_hazard"]
    chutes_hierarchy = chutes_test.loc["hierarchical_hazard"]
    rolling_selected = int(chutes_rolling["selected"].sum())
    rolling_passes = int(chutes_rolling["test_contract_pass"].sum())
    report = f"""# ColdGuard 独立外部证据汇总

## 综合结论

三份独立数据集形成互补证据链。FreeInference Agentic 的16周请求轨迹验证长期模型复活排序：层级适配器在严格后移测试段达到 ROC-AUC {agent_hierarchy['roc_auc']:.3f}、PR-AUC {agent_hierarchy['pr_auc']:.3f}。Chutes进一步把同领域外部验证扩展到61.22亿次调用、9174个模型和全年尺度；其7日冷龄/层级hazard达到 ROC-AUC {chutes_age['roc_auc']:.3f}/{chutes_hierarchy['roc_auc']:.3f}、PR-AUC {chutes_age['average_precision']:.3f}/{chutes_hierarchy['average_precision']:.3f}。GenTD26则提供独立的真实GPU duty-cycle、显存驻留和模型加载事件。

外部证据同时揭示边界。GenTD26 的层级适配器仅为 {gentd_hierarchy['roc_auc']:.3f}/{gentd_hierarchy['pr_auc']:.3f}，说明短时、跨工作负载场景不能依靠身份层级自动迁移。风险层保持正确行为：Agentic 和 GenTD26 的最小治理风险上界分别为 {min_agent:.3f} 与 {min_gentd:.3f}，初始固定校准均不授权动作；Chutes的60日滚动再校准随后产生{rolling_selected}个合格方法—时域组合，并有{rolling_passes}个在下一时间块继续通过5%合同。

## 四类贡献证据

1. **长期复活可预测性**：Agentic 跨16周、17个模型和11个Provider，支持层级特征对未来请求复活的排序价值。
2. **全年跨平台证书刷新**：Chutes在11,595个冷episode上显示，固定早期校准应当拒绝执行，而滚动校准可随风险分布变化逐步扩展合格时域。
3. **真实GPU机制链**：GenTD26 在143个容器、157,415条精确对齐记录上识别“低duty但高显存驻留”的冷状态，并连接到随后GPU复活。
4. **复活代价的事件证据**：545个可观测复活episode中，Pipeline/Base Model加载事件在复活窗口的出现率为52.7%/13.6%，而配对的提前30分钟安慰剂窗口仅为0.2%/0.0%。

## 论文使用边界

- Agentic 支撑请求级复活排序与长期漂移，不用于计算GPU显存回收量。
- Chutes 支撑同领域跨平台、未见模型和全年证书刷新；它没有GPU驻留字段，因此不报告GiB-hours。
- GenTD26 支撑真实显存冷驻留和加载事件关联；22.8小时长度不足以承担长期稳定性结论。
- Chutes滚动策略通过的是请求复活动作风险合同，不等价于真实GPU显存回收安全证明；真实资源结论仍由Alibaba MaaS和GenTD26承担。

## 可复现产物

- `figure_external_validation_source.csv`：总图底层数据。
- `figure_manifest.json`：输入哈希、变换和导出参数。
- `caption.txt` 与 `alt_text.txt`：论文图注和无障碍描述。

## 可视化方法引用

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
"""
    (output_dir / "report_CN.md").write_text(report, encoding="utf-8")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
