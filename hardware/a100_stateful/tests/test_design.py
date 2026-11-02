from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "workload" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def config() -> dict:
    return json.loads((ROOT / "config" / "a100_stateful.json").read_text())


def test_config_capacity_and_safety() -> None:
    cfg = config()
    assert cfg["capacity"]["maximum_resident_models"] == 4
    assert cfg["capacity"]["maximum_resident_models"] * cfg["model"]["planned_reservation_gib"] <= cfg["capacity"]["planned_gpu_budget_gib"]
    assert cfg["guardrails"]["automatic_pod_stop"] is False
    assert cfg["physical_gpus"] == [0, 1, 2, 3]
    assert len(cfg["request_order_seeds"]) == 4
    assert len(cfg["logical_models"]) == 6
    assert len(cfg["policies"]) == 5
    assert cfg["guardrails"]["expected_run_count"] == 60


def test_frozen_workload_counts() -> None:
    requests = rows("request_sequence.csv")
    assert len(requests) == 340
    assert Counter(int(row["replicate"]) for row in requests) == {0: 85, 1: 85, 2: 85, 3: 85}
    for replicate in range(4):
        subset = [row for row in requests if int(row["replicate"]) == replicate]
        assert len({int(row["event_index"]) for row in subset}) == 85
        assert Counter(int(row["block"]) for row in subset) == {1: 34, 2: 28, 3: 23}
        assert set(row["logical_model_id"] for row in subset) == {"m1", "m2", "m3", "m4", "m5", "m6"}


def test_logical_models_are_exact_7b_and_complete() -> None:
    models = rows("logical_models.csv")
    assert len(models) == 6
    assert all(float(row["params_b"]) == 7.0 for row in models)
    assert all(int(row["observed_days"]) == 30 for row in models)
    assert all(int(row["active_days"]) >= 5 and int(row["cold_days"]) >= 5 for row in models)


def test_latin_schedule_is_fully_balanced() -> None:
    cfg = config()
    schedule = rows("run_schedule.csv")
    assert len(schedule) == 60
    assert Counter(int(row["physical_gpu"]) for row in schedule) == {0: 15, 1: 15, 2: 15, 3: 15}
    assert set(Counter((int(row["block"]), int(row["physical_gpu"]), row["policy"]) for row in schedule).values()) == {1}
    assert set(Counter((int(row["block"]), row["policy"]) for row in schedule).values()) == {4}
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in schedule:
        grouped[(int(row["block"]), int(row["parallel_round"]))].append(row)
    assert len(grouped) == 15
    for group in grouped.values():
        assert {int(row["physical_gpu"]) for row in group} == {0, 1, 2, 3}
        assert len({row["policy"] for row in group}) == 4
    for block in (1, 2, 3):
        for policy in cfg["policies"]:
            subset = [row for row in schedule if int(row["block"]) == block and row["policy"] == policy]
            assert {int(row["request_seed"]) for row in subset} == set(cfg["request_order_seeds"])


def test_ports_are_unique_for_all_24_services() -> None:
    cfg = config()
    ports = {
        int(cfg["port_base"]) + gpu * int(cfg["port_stride_per_gpu"]) + logical_index + 1
        for gpu in cfg["physical_gpus"]
        for logical_index, _ in enumerate(cfg["logical_models"])
    }
    assert len(ports) == 24


def test_risks_are_bounded() -> None:
    risks = rows("risk_scores.csv")
    assert risks
    for row in risks:
        assert 0 <= float(row["coldguard_risk_3d"]) <= 1
        assert 0 <= float(row["direct_risk_3d"]) <= 1


def test_logical_audit_has_policy_separation() -> None:
    audit = rows("logical_policy_audit.csv")
    assert len(audit) == 60
    block_one = Counter()
    for row in audit:
        if int(row["block"]) == 1:
            block_one[row["policy"]] += int(row["logical_cache_misses"])
    assert block_one["oracle"] < block_one["coldguard_risk"] < block_one["lru"]
    assert block_one["direct_risk"] <= block_one["coldguard_risk"]


def test_parallel_runner_has_graceful_stop_and_no_pod_stop() -> None:
    shell_text = "\n".join(path.read_text() for path in (ROOT / "scripts").glob("*.sh"))
    python_text = (ROOT / "scripts" / "run_competition.py").read_text()
    assert "runpodctl stop" not in shell_text
    assert "automatic pod stop is disabled" in shell_text.lower()
    assert "ThreadPoolExecutor" in python_text
    assert "STOP_EVENT" in python_text
    assert "signal.SIGINT" in python_text


def test_protocol_discloses_pseudoreplication_boundary() -> None:
    protocol = (ROOT / "PROTOCOL_CN.md").read_text()
    assert "60次" in protocol
    assert "不是四条独立生产轨迹" in protocol
    assert "拉丁" in protocol
