from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/revision_round2/runtime_promotion_contract"


def main() -> None:
    contract = json.loads((ARTIFACT / "runtime_promotion_contract.json").read_text(encoding="utf-8"))
    table = pd.read_csv(ARTIFACT / "runtime_promotion_contract.csv", dtype=str).fillna("")

    expected_states = {
        "RETAIN",
        "ABSTAIN",
        "RECALIBRATE",
        "SHADOW_ONLY_UNRESOLVED_BOUND",
        "CANARY_ELIGIBLE",
        "EXPAND",
        "EMERGENCY_DISABLE",
    }
    assert len(table) == 7
    assert set(table.state) == expected_states
    assert table.state.is_unique
    assert contract["study_action_budget"] == 0.10
    assert contract["study_exposure_budget"] == 0.10
    assert contract["study_minimum_calibration_actions"] == 200
    assert contract["study_parameters_are_production_slos"] is False
    assert contract["current_study_authorizes_real_memory_action"] is False
    assert contract["current_empirical_route"] == "SHADOW_ONLY_UNRESOLVED_BOUND"

    required = [
        "entry_condition",
        "memory_action",
        "minimum_evidence",
        "execution_exposure_cap",
        "monitoring_record",
        "expiry_or_transition",
        "rollback_destination",
        "evidence_status",
    ]
    assert table[required].apply(lambda column: column.str.len().gt(0).all()).all()

    zero_states = table.loc[
        table.state.isin(
            ["RETAIN", "ABSTAIN", "RECALIBRATE", "SHADOW_ONLY_UNRESOLVED_BOUND"]
        )
    ]
    assert zero_states.execution_exposure_cap.eq("0").all()
    shadow = table.loc[table.state.eq("SHADOW_ONLY_UNRESOLVED_BOUND")].iloc[0]
    assert "200" in shadow.minimum_evidence
    assert "proposed" in shadow.minimum_evidence
    assert "observed" in shadow.minimum_evidence
    assert "unresolved" in shadow.minimum_evidence
    assert "all 19" in shadow.evidence_status

    canary = table.loc[table.state.eq("CANARY_ELIGIBLE")].iloc[0]
    expand = table.loc[table.state.eq("EXPAND")].iloc[0]
    assert "not instantiated" in canary.execution_exposure_cap
    assert "not instantiated" in expand.execution_exposure_cap
    assert "not reached or validated" in canary.evidence_status
    assert "not reached or validated" in expand.evidence_status
    assert "lifecycle" in canary.entry_condition
    assert "restoration" in canary.minimum_evidence
    assert "request/SLO" in canary.minimum_evidence

    emergency = table.loc[table.state.eq("EMERGENCY_DISABLE")].iloc[0]
    assert "Immediately stop" in emergency.memory_action
    assert emergency.rollback_destination == "RETAIN"
    assert "No automatic recovery" in emergency.expiry_or_transition

    report = (ARTIFACT / "RUNTIME_PROMOTION_CONTRACT_CN.md").read_text(encoding="utf-8")
    assert "真实内存动作数为 **0**" in report
    assert "不是生产 SLO" in report
    assert "不能独立授权动作" in report
    assert "不虚构这些数值" in report

    result = {
        "status": "PASS",
        "claims_checked": 34,
        "states": len(table),
        "zero_exposure_states": int(len(zero_states)),
        "current_empirical_route": contract["current_empirical_route"],
        "current_study_authorizes_real_memory_action": False,
    }
    (ARTIFACT / "claim_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
