from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/revision_round2/runtime_promotion_contract"


CONTRACT = {
    "contract_version": "1.0",
    "study_action_budget": 0.10,
    "study_exposure_budget": 0.10,
    "study_minimum_calibration_actions": 200,
    "study_parameters_are_production_slos": False,
    "current_study_authorizes_real_memory_action": False,
    "current_empirical_route": "SHADOW_ONLY_UNRESOLVED_BOUND",
    "operator_preregistered_fields": [
        "canary exposure cap",
        "minimum independent restoration events",
        "minimum affected requests",
        "restoration latency and tail limit",
        "request/SLO loss limit",
        "consecutive passing canary windows",
        "expansion schedule",
        "evidence maximum age",
    ],
    "states": [
        {
            "state": "RETAIN",
            "entry_condition": "Default state; any required gate, field, or route is unavailable.",
            "memory_action": "Keep the allocation; do not reclaim memory.",
            "minimum_evidence": "None; fail-closed default.",
            "execution_exposure_cap": "0",
            "monitoring_record": "Telemetry completeness, route reason, and time since last eligible evaluation.",
            "expiry_or_transition": "Re-evaluate only after the missing evidence and all declared fields are available.",
            "rollback_destination": "RETAIN",
            "evidence_status": "Executable default in the present study",
        },
        {
            "state": "ABSTAIN",
            "entry_condition": "Fewer than 200 calibration first-actions, unsupported diagnostic stratum, or an unscorable required feature.",
            "memory_action": "Issue no proposal and retain the allocation.",
            "minimum_evidence": "The study's 200-action calibration minimum is not met for the global policy; group diagnostics never independently authorize action.",
            "execution_exposure_cap": "0",
            "monitoring_record": "Eligible-action count, unsupported-stratum count, missing-feature count, and abstention reason.",
            "expiry_or_transition": "Return to qualification after support and feature completeness recover.",
            "rollback_destination": "RETAIN",
            "evidence_status": "Executable routing rule in the present study",
        },
        {
            "state": "RECALIBRATE",
            "entry_condition": "Calibration fails; the full-stream lower endpoint fails; evidence is stale; or model, data, cold-definition, or topology version changes.",
            "memory_action": "Freeze the policy and build a new mature evidence window; reclaim no memory.",
            "minimum_evidence": "A new temporally valid evidence window with complete proposal and outcome accounting.",
            "execution_exposure_cap": "0",
            "monitoring_record": "Version identifiers, evidence age, drift statistics, lower-endpoint bounds, and unresolved counts.",
            "expiry_or_transition": "Only a newly calibrated frozen policy may enter a forward audit.",
            "rollback_destination": "RETAIN",
            "evidence_status": "Executable expiry rule in the present study",
        },
        {
            "state": "SHADOW_ONLY_UNRESOLVED_BOUND",
            "entry_condition": "Calibration and the full-stream lower endpoint pass, but either conservative upper-endpoint risk bound exceeds its study budget.",
            "memory_action": "Log every proposal and lifecycle outcome without changing memory state.",
            "minimum_evidence": "At least 200 calibration first-actions plus complete proposed, observed, unresolved, action-risk, exposure-risk, and tipping-point records.",
            "execution_exposure_cap": "0",
            "monitoring_record": "Both endpoint UCB pairs, proposed/observed/unresolved counts, tipping point, drift, missingness, and telemetry completeness.",
            "expiry_or_transition": "Move to canary eligibility only after both full-stream endpoint pairs pass and online evidence fields are preregistered; move to recalibration on lower-endpoint failure or staleness.",
            "rollback_destination": "RETAIN or RECALIBRATE",
            "evidence_status": "Observed endpoint for all 19 present complete-case passes",
        },
        {
            "state": "CANARY_ELIGIBLE",
            "entry_condition": "Both full-stream endpoint risk pairs pass and lifecycle, restoration, affected-request, criticality, cache-interference, and request/SLO telemetry are complete.",
            "memory_action": "Permit a bounded canary only under an operator-preregistered nonzero cap; otherwise remain shadow-only.",
            "minimum_evidence": "Complete full-stream accounting plus operator-preregistered independent restoration-event, affected-request, latency-tail, and request/SLO requirements.",
            "execution_exposure_cap": "Operator-preregistered nonzero cap; not instantiated by this trace study",
            "monitoring_record": "Online action/exposure UCBs, restoration latency and tail, affected requests, SLO burn, cache interference, drift, missingness, and unresolved lifecycle count.",
            "expiry_or_transition": "Eligibility expires on evidence-age, version, definition, topology, drift, or telemetry-completeness triggers.",
            "rollback_destination": "EMERGENCY_DISABLE then RETAIN",
            "evidence_status": "Protocol state; not reached or validated in the present study",
        },
        {
            "state": "EXPAND",
            "entry_condition": "The operator-preregistered number of consecutive canary windows passes every online risk, restoration, and request/SLO gate without unresolved or missing telemetry.",
            "memory_action": "Increase exposure in operator-preregistered stages; never skip a stage.",
            "minimum_evidence": "Predeclared passing-window, independent-event, affected-request, restoration-tail, and SLO evidence requirements.",
            "execution_exposure_cap": "Operator-preregistered staged cap; not instantiated by this trace study",
            "monitoring_record": "All canary monitors plus stage identifier, exposure denominator, policy version, and rollback readiness.",
            "expiry_or_transition": "Any gate failure stops expansion; material version or topology changes require recalibration and a new canary.",
            "rollback_destination": "EMERGENCY_DISABLE then RETAIN",
            "evidence_status": "Protocol state; not reached or validated in the present study",
        },
        {
            "state": "EMERGENCY_DISABLE",
            "entry_condition": "Either online risk UCB crosses its budget; any preregistered request/SLO or restoration-tail limit fails; required telemetry is missing; or an operator emergency trigger fires.",
            "memory_action": "Immediately stop new actions, preserve or restore allocations, and disable expansion.",
            "minimum_evidence": "A single declared trigger is sufficient.",
            "execution_exposure_cap": "0 for all new actions",
            "monitoring_record": "Trigger, time, affected deployments and requests, restoration outcome, incident status, and policy/version identifiers.",
            "expiry_or_transition": "No automatic recovery; a fresh evidence window, recalibration, and new canary qualification are required.",
            "rollback_destination": "RETAIN",
            "evidence_status": "Executable rollback interface; online trigger values require operator preregistration",
        },
    ],
}


def write_csv() -> None:
    fields = [
        "state",
        "entry_condition",
        "memory_action",
        "minimum_evidence",
        "execution_exposure_cap",
        "monitoring_record",
        "expiry_or_transition",
        "rollback_destination",
        "evidence_status",
    ]
    with (OUT / "runtime_promotion_contract.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(CONTRACT["states"])


def write_report() -> None:
    rows = CONTRACT["states"]
    lines = [
        "# ColdGuard 运行时晋级契约",
        "",
        "## 结论",
        "",
        "当前研究授权的真实内存动作数为 **0**。19 个通过完整案例端点的策略，在纳入未决结局的完整提议流后全部进入 `SHADOW_ONLY_UNRESOLVED_BOUND`。这不是方法失效，而是 ColdGuard 的关键系统贡献：把不完整证据转换为可审计、可回滚、默认不执行的运行态。",
        "",
        "研究中的 10% 行动风险和暴露风险预算只是用于展示通过与失败边界的研究参数，不是生产 SLO。任何非零 canary 暴露上限、恢复时延尾部阈值、受影响请求阈值、连续通过窗口数和扩容节奏，都必须由运营方在执行前登记；本轨迹研究不虚构这些数值。",
        "",
        "## 状态机",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['state']}",
                "",
                f"- 进入条件：{row['entry_condition']}",
                f"- 内存动作：{row['memory_action']}",
                f"- 最低证据：{row['minimum_evidence']}",
                f"- 执行暴露上限：{row['execution_exposure_cap']}",
                f"- 在线记录：{row['monitoring_record']}",
                f"- 失效/迁移：{row['expiry_or_transition']}",
                f"- 回滚目标：{row['rollback_destination']}",
                f"- 当前证据状态：{row['evidence_status']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 论文表述边界",
            "",
            "- `CANARY_ELIGIBLE` 和 `EXPAND` 是 ColdGuard 提供的运行协议接口，不是本数据集已经达到的实证成果。",
            "- 新颖性、站点和显存分组只用于诊断支持度与集中度；任何分组诊断都不能独立授权动作。",
            "- `SHADOW_ONLY_UNRESOLVED_BOUND` 的执行暴露上限严格为 0，但持续记录完整提议、生命周期与服务证据，为后续 canary 资格提供材料。",
            "- 任一在线风险、请求/SLO、恢复尾部或遥测完整性门槛失败，均直接进入 `EMERGENCY_DISABLE`，停止新动作并回到 `RETAIN`。",
            "",
        ]
    )
    (OUT / "RUNTIME_PROMOTION_CONTRACT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runtime_promotion_contract.json").write_text(
        json.dumps(CONTRACT, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv()
    write_report()
    print(json.dumps({"status": "OK", "states": len(CONTRACT["states"]), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
