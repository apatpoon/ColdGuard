from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/full_proposal_stream"


def _one(frame: pd.DataFrame, definition: str) -> pd.Series:
    return frame.loc[
        (frame["cold_definition"] == definition)
        & (frame["horizon"] == 3)
        & (frame["method"] == "hazard_hierarchical")
    ].iloc[0]


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    frame = pd.read_csv(ARTIFACT / "policy_bounds.csv")
    primary = _one(frame, "sm_le_1.0_pct")
    strict = _one(frame, "sm_le_0.1_pct")
    originally_forward = frame.loc[
        frame["original_complete_case_route"] == "FORWARD_SUPPORTED"
    ]
    mismatch = frame.loc[~frame["observed_count_matches_complete_case"].astype(bool)]
    text = f"""# 完整 Proposal Stream 风险审计

## 结论

本轮分析将策略资格判断从 complete-case 子集改为冻结策略实际产生的完整 proposal stream。原先 {len(originally_forward)} 个标记为 `FORWARD_SUPPORTED` 的 operating points，在未决结局极端上界下均不能维持正式通过，全部改为 `SHADOW_ONLY_UNRESOLVED_BOUND`。这不是预测模型失效，而是生产决策证据尚未成熟：只要未决动作的复活比例没有被额外假设或后续观测约束，最保守上界就会跨越至少一个 10% 风险预算。

## 三天 Hazard 主结果

| 冷定义 | 提案数 | 已观察 | 未决 | 未决比例 | 全不复活：动作/暴露上界 | 全复活：动作/暴露上界 | 首次越界比例 | 新路由 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SM ≤ 1% | {int(primary.proposed_actions)} | {int(primary.observed_actions)} | {int(primary.unresolved_actions)} | {pct(primary.unresolved_share)} | {pct(primary.lower_action_upper)} / {pct(primary.lower_resource_upper)} | {pct(primary.upper_action_upper)} / {pct(primary.upper_resource_upper)} | {pct(primary.dual_bound_first_crossing_fraction)} | `SHADOW_ONLY_UNRESOLVED_BOUND` |
| SM ≤ 0.1% | {int(strict.proposed_actions)} | {int(strict.observed_actions)} | {int(strict.unresolved_actions)} | {pct(strict.unresolved_share)} | {pct(strict.lower_action_upper)} / {pct(strict.lower_resource_upper)} | {pct(strict.upper_action_upper)} / {pct(strict.upper_resource_upper)} | {pct(strict.dual_bound_first_crossing_fraction)} | `SHADOW_ONLY_UNRESOLVED_BOUND` |

严格三天策略的完整流包含 514 个动作，其中 53 个未决。若未决动作全部不复活，两类上界仍低于预算；若全部复活，动作和暴露上界分别升至 {pct(strict.upper_action_upper)} 和 {pct(strict.upper_resource_upper)}。在预设 1% 网格上，当未决复活比例达到 {pct(strict.dual_bound_first_crossing_fraction)}，即约 {math.ceil(strict.dual_bound_first_crossing_fraction * strict.unresolved_actions)} 个未决动作的等效量时，至少一个上界首次超过 10%。

主要冷定义的三天策略包含 811 个动作，其中 74 个未决；首次越界发生在 {pct(primary.dual_bound_first_crossing_fraction)}，约 {math.ceil(primary.dual_bound_first_crossing_fraction * primary.unresolved_actions)} 个未决动作的等效量。

## 方法口径

动作候选期仍严格截止于 `test_end - horizon`，没有使用冻结测试窗之后的结局。对每个 calibration-qualified 策略，阈值保持不变，并在删除任何未决行之前生成每个 episode 的首次动作。端点定义为：

- `q=0`：所有未决动作记为不复活，只使用轨迹内已观察暴露；
- `q=1`：所有未决动作记为复活，暴露扩展到“动作日平均分配持续完整时间窗”的上界；
- `0<q<1`：未决标签取分数值 `q`，暴露在两个端点之间同步线性扩展。

中间曲线是 tipping sensitivity，不是 IPCW、插补模型或已识别概率。其作用是回答“未决结局需要恶化到什么程度，当前资格结论才会失效”。

## 新发现：complete-case 会改变动作身份

31 个校准通过策略中有 {len(mismatch)} 个无法将完整流中的已观察动作数与原 complete-case 动作数逐项对齐。原因是 complete-case 先删除未决行，随后可能在同一 episode 中选择一个更晚的可观察动作；完整策略则会在更早的未决时点触发。这说明 complete-case 不仅改变分母，还改变 intervention time 和 action identity。完整 proposal stream 因而应成为主 estimand，complete-case 只保留为描述性辅助结果。

## 对论文主张的影响

1. 摘要中的“strict three-day hazard rule remains forward-supported”必须撤回，改为“passes the observed endpoint but routes to shadow-only under unresolved-outcome bounds”。
2. `FORWARD_SUPPORTED` 应保留给上下两个端点都通过的策略；本轮没有策略满足这一标准。
3. `SHADOW_ONLY_UNRESOLVED_BOUND` 允许继续收集结局和评估漂移，但不授权真实回收动作。
4. 结果强化了 ColdGuard 的资格审查层定位：它能够拒绝证据不足的自家模型，而不是只为产生更多回收动作服务。

## 可复现产物

- `policy_bounds.csv`：31 个策略的完整流端点、tipping fraction 与新路由；
- `tipping_curves.csv`：每个策略 101 个敏感性点；
- `selected_actions.parquet`：固定阈值产生的动作级审计表；
- `claim_verification.json`：关键数字和时间边界的机器校验；
- `figure_full_proposal_stream.*`：主三天策略的投稿级敏感性图。

## 程序性知识来源

本轮统计审计与科学可视化流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
"""
    (ARTIFACT / "FULL_PROPOSAL_STREAM_AUDIT_CN.md").write_text(
        text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()

