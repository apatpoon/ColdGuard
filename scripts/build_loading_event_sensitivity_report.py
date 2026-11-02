from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/loading_event_sensitivity"


def row(frame: pd.DataFrame, method: str, scenario: str) -> pd.Series:
    return frame.loc[
        (frame["cold_definition"] == "sm_le_0.1_pct")
        & (frame["method"] == method)
        & (frame["scenario"] == scenario)
    ].iloc[0]


def main() -> None:
    frame = pd.read_csv(ARTIFACT / "scenario_summary.csv")
    h_alloc = row(frame, "hazard_hierarchical", "allocation_proxy")
    d_alloc = row(frame, "direct_monotone", "allocation_proxy")
    h_fp16 = row(frame, "hazard_hierarchical", "fp16_checkpoint")
    d_fp16 = row(frame, "direct_monotone", "fp16_checkpoint")
    h_int8 = row(frame, "hazard_hierarchical", "int8_checkpoint")
    d_int8 = row(frame, "direct_monotone", "int8_checkpoint")
    h_rank = row(frame, "hazard_hierarchical", "fp16_per_rank_parallel")
    d_rank = row(frame, "direct_monotone", "fp16_per_rank_parallel")
    text = f"""# 加载事件口径敏感性审计

## 结论

严格三天动作集的加载结论对“传输的到底是什么”高度敏感。以 trace 中 deployment-level allocated memory 作为传输体积时，ColdGuard hazard 的有限事件 p95 为 {h_alloc.p95_added_s:.2f} s，projected direct 为 {d_alloc.p95_added_s:.2f} s；但若使用模型参数数推导的 FP16 权重体积，两者分别降为 {h_fp16.p95_added_s:.2f} s 和 {d_fp16.p95_added_s:.2f} s。因而原 17.34/29.00 s 只能称为 allocation-proxy scenario，而不能解释为真实模型权重加载时间。

## 严格三天事件结果（16 GiB/s + 2 s 固定开销）

| 体积假设 | ColdGuard hazard p95 | LOO 范围 | 覆盖事件 | Projected direct p95 | LOO 范围 | 覆盖事件 |
|---|---:|---:|---:|---:|---:|---:|
| Deployment allocation proxy | {h_alloc.p95_added_s:.2f} s | {h_alloc.loo_p95_min_s:.2f}–{h_alloc.loo_p95_max_s:.2f} s | {int(h_alloc.events_with_volume)}/{int(h_alloc.events_total)} | {d_alloc.p95_added_s:.2f} s | {d_alloc.loo_p95_min_s:.2f}–{d_alloc.loo_p95_max_s:.2f} s | {int(d_alloc.events_with_volume)}/{int(d_alloc.events_total)} |
| FP16 parameter weights | {h_fp16.p95_added_s:.2f} s | {h_fp16.loo_p95_min_s:.2f}–{h_fp16.loo_p95_max_s:.2f} s | {int(h_fp16.events_with_volume)}/{int(h_fp16.events_total)} | {d_fp16.p95_added_s:.2f} s | {d_fp16.loo_p95_min_s:.2f}–{d_fp16.loo_p95_max_s:.2f} s | {int(d_fp16.events_with_volume)}/{int(d_fp16.events_total)} |
| INT8 parameter weights | {h_int8.p95_added_s:.2f} s | {h_int8.loo_p95_min_s:.2f}–{h_int8.loo_p95_max_s:.2f} s | {int(h_int8.events_with_volume)}/{int(h_int8.events_total)} | {d_int8.p95_added_s:.2f} s | {d_int8.loo_p95_min_s:.2f}–{d_int8.loo_p95_max_s:.2f} s | {int(d_int8.events_with_volume)}/{int(d_int8.events_total)} |
| FP16 / trace GPU ranks | {h_rank.p95_added_s:.2f} s | {h_rank.loo_p95_min_s:.2f}–{h_rank.loo_p95_max_s:.2f} s | {int(h_rank.events_with_volume)}/{int(h_rank.events_total)} | {d_rank.p95_added_s:.2f} s | {d_rank.loo_p95_min_s:.2f}–{d_rank.loo_p95_max_s:.2f} s | {int(d_rank.events_with_volume)}/{int(d_rank.events_total)} |

## 为什么这个修订重要

Allocation proxy 包含实例数、运行时保留和部署形态等因素，并不等于 checkpoint bytes。严格 direct 集合中最大的事件分配为 561.22 GiB，但参数元数据对应 82B 参数、平均 4 张 GPU；其 allocation-proxy 时间为 37.08 s，FP16 总权重时间为 11.55 s，按 trace GPU 数均分后为 4.39 s。这个事件单独决定了原叙述中“存在一个超过 30 s 的事件”，也使 direct allocation-proxy p95 的留一范围达到 {d_alloc.loo_p95_min_s:.2f}–{d_alloc.loo_p95_max_s:.2f} s。

ColdGuard hazard 的 allocation-proxy p95 留一范围为 {h_alloc.loo_p95_min_s:.2f}–{h_alloc.loo_p95_max_s:.2f} s，说明九个事件下尾部估计仍具有有限样本敏感性。参数体积口径只能覆盖 8/9 hazard 事件，因为一个复活事件缺少 `params_b`；该缺失没有被插补，直接报告覆盖率。

## 论文采用的表述边界

1. 所有结果均称为 `scenario-implied added time`，不称生产延迟或实测冷启动。
2. 17.34/29.00 s 保留为 allocation-proxy 口径，并与参数权重和并行分片口径并列。
3. 留一范围仅描述这组有限恢复事件对单个事件的敏感程度，不称置信区间。
4. 系统后果分析位于资格路由之后；完整 proposal stream 的未决上界仍使策略保持 shadow-only。

## 可复现产物

- `event_level_transfer_scenarios.csv`：每个恢复事件的四种体积和时间；
- `scenario_summary.csv`：p50、p95、最大值、30-s 超阈值数和覆盖率；
- `leave_one_event_out.csv`：逐事件删除后的 p95；
- `claim_verification.json`：12 个关键断言的机器验证；
- `figure_loading_event_sensitivity.*`：投稿级图与源数据。

## 程序性知识来源

本轮统计审计与科学可视化流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
"""
    (ARTIFACT / "LOADING_EVENT_SENSITIVITY_CN.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
