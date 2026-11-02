from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import atomic_json, read_json, utc_now


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json_if_present(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def mean(rows: list[float]) -> float | None:
    return statistics.mean(rows) if rows else None


def median(rows: list[float]) -> float | None:
    return statistics.median(rows) if rows else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary_dir = args.result_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "requested_workload" / "request_sequence.csv").open(newline="", encoding="utf-8") as handle:
        requested_events = list(csv.DictReader(handle))
    expected_event_counts = Counter((int(row["block"]), int(row["replicate"])) for row in requested_events)
    completions = [read_json(path) for path in sorted((args.result_dir / "policy_runs").glob("*/completion.json"))]
    event_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for completion in completions:
        run_dir = args.result_dir / "policy_runs" / completion["run_id"]
        events = load_jsonl(run_dir / "events.jsonl")
        evictions = load_jsonl(run_dir / "evictions.jsonl")
        schedule = completion["schedule"]
        policy = schedule["policy"]
        gpu = int(schedule["physical_gpu"])
        replicate = int(schedule["replicate"])
        active_metrics = [row for row in completion.get("gpu_metrics", []) if int(row["gpu_index"]) == gpu]
        for event in events:
            event_rows.append({
                "run_id": completion["run_id"],
                "block": int(schedule["block"]),
                "parallel_round": int(schedule["parallel_round"]),
                "replicate": replicate,
                "request_seed": int(schedule["request_seed"]),
                "policy": policy,
                "physical_gpu": gpu,
                "event_index": event["event_index"],
                "day": event["day"],
                "logical_model_id": event["logical_model_id"],
                "cache_hit": event["cache_hit"],
                "miss_type": event["miss_type"],
                "startup_s": event["startup_s"],
                "ttft_s": event["ttft_s"],
                "request_total_s": event["request_total_s"],
                "recovery_to_first_token_s": event["recovery_to_first_token_s"],
                "recovery_to_completion_s": event["recovery_to_completion_s"],
                "evictions_for_request": len(event["evicted_models"]),
            })
        first_token = [float(row["recovery_to_first_token_s"]) for row in events]
        hit_ttft = [float(row["ttft_s"]) for row in events if row["cache_hit"]]
        threshold = float(config["measurement"]["slo_recovery_to_first_token_s"])
        coldguard_in_contract = [
            bool(row.get("detail", {}).get("within_qualification_threshold"))
            for row in evictions
            if policy == "coldguard_risk"
        ]
        run_rows.append({
            "run_id": completion["run_id"],
            "block": int(schedule["block"]),
            "parallel_round": int(schedule["parallel_round"]),
            "replicate": replicate,
            "request_seed": int(schedule["request_seed"]),
            "policy": policy,
            "physical_gpu": gpu,
            "status": completion["status"],
            "guardrails_passed": completion.get("guardrails_passed"),
            "requests": len(events),
            "cache_hits": sum(bool(row["cache_hit"]) for row in events),
            "cache_misses": sum(not bool(row["cache_hit"]) for row in events),
            "compulsory_misses": sum(row["miss_type"] == "compulsory_miss" for row in events),
            "eviction_misses": sum(row["miss_type"] == "eviction_miss" for row in events),
            "evictions": len(evictions),
            "cache_hit_rate": sum(bool(row["cache_hit"]) for row in events) / len(events) if events else None,
            "median_hit_ttft_s": median(hit_ttft),
            "cumulative_recovery_to_first_token_s": sum(first_token),
            "median_recovery_to_first_token_s": median(first_token),
            "slo_violations": sum(value > threshold for value in first_token),
            "slo_violation_rate": sum(value > threshold for value in first_token) / len(first_token) if first_token else None,
            "absolute_gpu_energy_kj": sum(float(row.get("energy_j") or 0.0) for row in active_metrics) / 1000,
            "peak_gpu_memory_increment_gib": max((float(row["peak_increment_mib"]) for row in active_metrics), default=0.0) / 1024,
            "memory_returned": completion.get("gpu_memory_returned_to_baseline"),
            "wall_s": completion.get("wall_s"),
            "coldguard_evictions_within_threshold": sum(coldguard_in_contract) if coldguard_in_contract else None,
            "coldguard_evictions_total": len(coldguard_in_contract) if coldguard_in_contract else None,
        })
    write_csv(summary_dir / "event_results.csv", event_rows)
    write_csv(summary_dir / "run_results.csv", run_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row["status"] == "COMPLETED":
            grouped[row["policy"]].append(row)
    policy_rows = []
    for policy, rows in sorted(grouped.items()):
        total_requests = sum(int(row["requests"]) for row in rows)
        policy_rows.append({
            "policy": policy,
            "n_replays": len(rows),
            "n_trace_blocks": len({int(row["block"]) for row in rows}),
            "n_request_seeds": len({int(row["request_seed"]) for row in rows}),
            "n_physical_gpus": len({int(row["physical_gpu"]) for row in rows}),
            "requests": total_requests,
            "cache_hits": sum(int(row["cache_hits"]) for row in rows),
            "cache_misses": sum(int(row["cache_misses"]) for row in rows),
            "eviction_misses": sum(int(row["eviction_misses"]) for row in rows),
            "evictions": sum(int(row["evictions"]) for row in rows),
            "cache_hit_rate": sum(int(row["cache_hits"]) for row in rows) / total_requests,
            "cumulative_recovery_to_first_token_s": sum(float(row["cumulative_recovery_to_first_token_s"]) for row in rows),
            "slo_violations": sum(int(row["slo_violations"]) for row in rows),
            "slo_violation_rate": sum(int(row["slo_violations"]) for row in rows) / total_requests,
            "absolute_gpu_energy_kj": sum(float(row["absolute_gpu_energy_kj"]) for row in rows),
            "median_replay_hit_rate": median([float(row["cache_hit_rate"]) for row in rows]),
            "all_memory_returned": all(bool(row["memory_returned"]) for row in rows),
        })
    write_csv(summary_dir / "policy_summary.csv", policy_rows)

    # Paired comparisons use the same trace block and GPU/request-seed block.
    by_unit: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        if row["status"] == "COMPLETED":
            key = (int(row["block"]), int(row["physical_gpu"]), int(row["request_seed"]))
            by_unit[key][row["policy"]] = row
    contrast_rows = []
    for comparator in config["policies"]:
        if comparator == "coldguard_risk":
            continue
        pairs = [(key, rows["coldguard_risk"], rows[comparator]) for key, rows in by_unit.items() if "coldguard_risk" in rows and comparator in rows]
        miss_deltas = [float(other["cache_misses"]) - float(cg["cache_misses"]) for _, cg, other in pairs]
        hit_deltas = [float(cg["cache_hit_rate"]) - float(other["cache_hit_rate"]) for _, cg, other in pairs]
        recovery_deltas = [float(other["cumulative_recovery_to_first_token_s"]) - float(cg["cumulative_recovery_to_first_token_s"]) for _, cg, other in pairs]
        slo_deltas = [float(other["slo_violations"]) - float(cg["slo_violations"]) for _, cg, other in pairs]
        energy_deltas = [float(other["absolute_gpu_energy_kj"]) - float(cg["absolute_gpu_energy_kj"]) for _, cg, other in pairs]
        contrast_rows.append({
            "reference_policy": "coldguard_risk",
            "comparator_policy": comparator,
            "n_paired_block_seed_gpu_units": len(pairs),
            "mean_comparator_minus_coldguard_cache_misses": mean(miss_deltas),
            "median_comparator_minus_coldguard_cache_misses": median(miss_deltas),
            "coldguard_miss_wins": sum(value > 0 for value in miss_deltas),
            "miss_ties": sum(value == 0 for value in miss_deltas),
            "coldguard_miss_losses": sum(value < 0 for value in miss_deltas),
            "mean_coldguard_minus_comparator_hit_rate": mean(hit_deltas),
            "mean_comparator_minus_coldguard_recovery_s": mean(recovery_deltas),
            "mean_comparator_minus_coldguard_slo_violations": mean(slo_deltas),
            "mean_comparator_minus_coldguard_energy_kj": mean(energy_deltas),
        })
    write_csv(summary_dir / "paired_policy_contrasts.csv", contrast_rows)

    planned_runs = int(config["guardrails"]["expected_run_count"])
    expected_per_policy = len(config["blocks"]) * len(config["physical_gpus"])
    schedule_counts = Counter(
        (int(row["block"]), int(row["physical_gpu"]), row["policy"])
        for row in run_rows
    )
    pilot = read_json_if_present(args.result_dir / "capacity_pilot" / "completion.json")
    audit = {
        "schema": "coldguard.a100_stateful_evidence_audit.v2",
        "generated_utc": utc_now(),
        "planned_runs": planned_runs,
        "observed_runs": len(completions),
        "failed_run_ids": [row["run_id"] for row in completions if row["status"] != "COMPLETED"],
        "all_guardrails_passed": len(completions) == planned_runs and all(row.get("guardrails_passed") is True for row in completions),
        "all_memory_returned": len(completions) == planned_runs and all(row.get("gpu_memory_returned_to_baseline") is True for row in completions),
        "all_requests_completed": len(completions) == planned_runs and all(
            len(row.get("events") or [])
            == expected_event_counts[(int(row["schedule"]["block"]), int(row["schedule"]["replicate"]))]
            and all(event.get("request", {}).get("status") == "COMPLETED" for event in row.get("events") or [])
            for row in completions
        ),
        "capacity_pilot_passed_all_four_gpus": (
            pilot.get("passed") is True
            and len(pilot.get("gpu_results") or []) == len(config["physical_gpus"])
            and {int(row["physical_gpu"]) for row in pilot.get("gpu_results") or []} == set(config["physical_gpus"])
            and all(row.get("passed") is True for row in pilot.get("gpu_results") or [])
        ),
        "preflight_passed": read_json_if_present(args.result_dir / "preflight.json").get("passed") is True,
        "balanced_policy_by_block_and_gpu": len(schedule_counts) == planned_runs and set(schedule_counts.values()) == {1},
        "policies_with_expected_block_seed_gpu_replays": sorted(policy for policy, rows in grouped.items() if len(rows) == expected_per_policy),
        "inference_note": "The four GPU/request-order repetitions are blocked robustness repetitions of the same underlying trace, not four independent production traces. Cluster inferential claims by the three trace blocks.",
    }
    audit["complete"] = bool(
        audit["observed_runs"] == planned_runs
        and not audit["failed_run_ids"]
        and audit["all_guardrails_passed"]
        and audit["all_memory_returned"]
        and audit["all_requests_completed"]
        and audit["capacity_pilot_passed_all_four_gpus"]
        and audit["preflight_passed"]
        and audit["balanced_policy_by_block_and_gpu"]
        and len(audit["policies_with_expected_block_seed_gpu_replays"]) == len(config["policies"])
    )
    atomic_json(summary_dir / "evidence_audit.json", audit)
    lines = [
        "# ColdGuard A100 four-GPU stateful competition",
        "",
        f"Generated UTC: {utc_now()}",
        "",
        f"- Planned replays: {planned_runs}",
        f"- Completed replays: {len(completions) - len(audit['failed_run_ids'])}",
        f"- Four-GPU simultaneous capacity pilot passed: {audit['capacity_pilot_passed_all_four_gpus']}",
        f"- Balanced policy × block × GPU design: {audit['balanced_policy_by_block_and_gpu']}",
        f"- Evidence complete: {audit['complete']}",
        "",
        "Each physical GPU is an independent 4-of-6 residency pool. A 5x4 cyclic Latin rectangle balances policy over physical GPU and parallel round within each trace block.",
        "The four request-order seeds are blocked robustness repetitions of the same trace; they are not reported as independent production traces. Requests, evictions, loads, and 200 ms telemetry are nested observations.",
        "The six anonymous logical deployments use one fixed Qwen2.5-7B artifact so policy-induced residency differences are not confounded by architecture or weight size.",
    ]
    (summary_dir / "completion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
