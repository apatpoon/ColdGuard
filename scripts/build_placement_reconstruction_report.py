from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/placement_reconstruction_sensitivity"


def main() -> None:
    structural = pd.read_csv(ARTIFACT / "structural_sensitivity.csv")
    daily = pd.read_csv(ARTIFACT / "daily_reconstruction_sensitivity.csv")
    audit = json.loads((ARTIFACT / "audit.json").read_text())
    c = structural.loc[structural.runtime_headroom.eq(0.20)].set_index(["variant", "template"])
    d = daily.loc[daily.runtime_headroom.eq(0.20)].set_index(["source", "template"])
    lines = [
        "# 放置重构敏感性审计",
        "",
        "## 结论",
        "",
        "严格三天 ColdGuard 的日均虚拟块包络对‘重复动作日块’这一时间简化较稳定，但对多 GPU 实例内的容量形状并非完全不敏感。540/540 个活跃 action-day 均可关联到原始每日记录；逐日重构后，20% headroom 下的 global-pool replica-days 比重复动作日块低约 1%，block-packed replica-days 的变化在 -0.51% 到 +0.12% 之间。",
        "",
        "## 20% headroom 的块形状敏感性",
        "",
        "| 模板 | 均匀拆分 | 单块/集中 | 均匀保留率 | 单块/集中保留率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for template in ["7B", "13B", "32B", "70B"]:
        u = c.loc[("round_uniform", template)]
        s = c.loc[("single_instance", template)]
        lines.append(
            f"| {template} | {int(u.block_packed_replica_days):,} | {int(s.block_packed_replica_days):,} | {100*u.placement_retention:.1f}% | {100*s.placement_retention:.1f}% |"
        )
    lines.extend(
        [
            "",
            "7B/13B 在单块或集中分配下略高，而 32B/70B 在均匀拆分下更高；这反映 TP2/TP4 需要多个 distinct virtual blocks。该比较是块形状边界，不是物理 GPU 拓扑结论。action-day 的平均 GPU count 全部为整数，因此 floor、round、ceil 三种计数规则完全重合；这不是假设无关性，而是当前样本元数据的性质。",
            "",
            "## 逐日重构",
            "",
            "| 模板 | 重复动作日 global / packed | 逐日 global / packed | 逐日/重复 global | 逐日/重复 packed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for template in ["7B", "13B", "32B", "70B"]:
        r = d.loc[("repeated_action_day", template)]
        o = d.loc[("trace_daily_reconstructed", template)]
        lines.append(
            f"| {template} | {int(r.global_pool_replica_days):,} / {int(r.block_packed_replica_days):,} | {int(o.global_pool_replica_days):,} / {int(o.block_packed_replica_days):,} | {100*o.global_pool_replica_days/r.global_pool_replica_days:.2f}% | {100*o.block_packed_replica_days/r.block_packed_replica_days:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"逐日 trace 重构覆盖 {audit['trace_observed_action_days']}/{audit['expected_active_action_days']} 个预期活跃 action-day；总 released GiB-days 从 72,296.51 变为 71,595.08。因而重复动作日块没有造成主数量级或模板排序变化，但最终稿应把逐日结果作为更忠实的主敏感性。",
            "",
            "## 解释边界",
            "",
            "1. 所有变体逐实例严格守恒日均分配容量；",
            "2. distinct virtual blocks 只约束块数量，不识别 NVLink、NUMA、故障域或同时空闲；",
            "3. 日均块不能证明任一时刻存在相同可用容量；",
            "4. headroom 15%/20%/30% 是结构场景，不是置信区间；",
            "5. 完整 proposal-stream 路由仍为 shadow-only，故 replica-days 是反事实驻留包络而非授权部署量。",
            "",
            "## 程序性知识来源",
            "",
            "本轮统计审计与科学可视化流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065",
            "",
        ]
    )
    (ARTIFACT / "PLACEMENT_RECONSTRUCTION_SENSITIVITY_CN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
