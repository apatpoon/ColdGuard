from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/selection_aware_coverage"


def one(frame: pd.DataFrame, scenario: str, protocol: str) -> pd.Series:
    return frame.loc[frame.scenario.eq(scenario) & frame.protocol.eq(protocol)].iloc[0]


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    frame = pd.read_csv(ARTIFACT / "selection_summary.csv")
    point = frame.loc[frame.protocol.eq("fixed_sequence_pointwise")]
    stable = point.loc[~point.scenario.eq("two_axis_test_shift")]
    heavy = one(frame, "two_axis_heavy_weight", "fixed_sequence_pointwise")
    drift = one(frame, "two_axis_test_shift", "fixed_sequence_pointwise")
    heavy_b = one(frame, "two_axis_heavy_weight", "threshold_bonferroni")
    drift_b = one(frame, "two_axis_test_shift", "threshold_bonferroni")
    text = f"""# 完整选择流程覆盖仿真

## 结论

本轮仿真不再把风险上界应用于一个预先固定的面板，而是完整执行：生成特征和风险分数 → 逐 episode 产生三个候选 landmark → 扫描预声明的 12 个阈值 → 双风险上界 → 固定最后一个通过阈值或弃权 → 在独立前向面板上重新审计。每个场景 300 次复制，所有比例均附 95% Wilson 区间。

在 IID、day shock、repeated deployment 和 two-axis heavy-weight 四个无分布漂移场景中，pointwise fixed-sequence 的选择后 calibration joint coverage 为 {pct(stable.calibration_joint_coverage.min())}–{pct(stable.calibration_joint_coverage.max())}，独立前向 joint coverage 为 {pct(stable.forward_joint_coverage.min())}–{pct(stable.forward_joint_coverage.max())}。四个场景中前向真实双预算安全率最低为 {pct(stable.forward_true_safe.min())}。这支持“经验压力测试下表现保守”，但不是任意序列相关过程的覆盖定理。

## 重尾与弃权

two-axis heavy-weight 场景下，pointwise fixed-sequence 选择 246/300 个策略，选择率 {pct(heavy.selection_rate)}；其中 245/246 的前向 oracle 双风险仍在预算内。阈值 Bonferroni 仅选择 46/300，选择率 {pct(heavy_b.selection_rate)}，且全部安全。额外阈值校正提高选择期覆盖，但显著增加弃权，说明该方案是保守敏感性而非免费改进。

## 漂移与过期

前向漂移场景只在测试期 outcome logit 上增加 0.90，分数模型不获知该变化。Pointwise 序列在 calibration 选择 258/300 个策略，其中 62 个在前向 oracle 下不再满足双预算，真实安全率为 {pct(drift.forward_true_safe)}（95% CI {pct(drift.forward_true_safe_ci_low)}–{pct(drift.forward_true_safe_ci_high)}）。前向双风险审计捕获 61/62 个不安全策略；仅 1/258 个已选策略同时真实不安全且再次通过上界，false-promotion rate 为 {pct(drift.false_promotion)}。因此，时间外审计和 expiry 是控制漂移的必要机制，不能由 calibration coverage 取代。

阈值 Bonferroni 在漂移场景只选择 45/300 个策略，全部保持 oracle-safe，但这一结果来自强烈弃权，不能外推为理论保证。

## 保证类型

1. 论文中的 95% family level 在每个 method–definition–horizon 候选上分配给 action 与 exposure 两个风险；
2. 预声明 fixed sequence 限制了阈值搜索路径，但当前 two-axis clustered bound 仍是经验 dependence-stress construction；
3. 阈值 Bonferroni 是额外敏感性，覆盖全部 12 个候选，代价是 heavy/shift 场景约 85% 的弃权；
4. 仿真支持经验校准和 fail-closed expiry，不构成任意非交换时间过程的 finite-sample theorem；
5. Monte Carlo 的独立单位是完整生成的 panel replication，不是 panel 内的行、天或 deployment。

## 可复现产物

- `selection_replications.csv`：3,000 个场景×协议复制记录；
- `selection_summary.csv`：选择率、安全率、覆盖率、错误资格与 Wilson 区间；
- `design.json`：DGP、阈值、种子、预算和保证边界；
- `claim_verification.json`：32 个机器断言；
- `figure_selection_aware_coverage.*`：完整流程图及源数据。

## 程序性知识来源

本轮统计审计与科学可视化流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
"""
    (ARTIFACT / "SELECTION_AWARE_COVERAGE_CN.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
