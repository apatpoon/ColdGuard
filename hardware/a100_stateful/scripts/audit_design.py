from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = load(args.workload_dir / "request_sequence.csv")
    schedule = load(args.workload_dir / "run_schedule.csv")
    risks = {
        (int(row["day"]), row["logical_model_id"]): {
            "coldguard_risk": float(row["coldguard_risk_3d"]),
            "direct_risk": float(row["direct_risk_3d"]),
        }
        for row in load(args.workload_dir / "risk_scores.csv")
    }
    if len(schedule) != 60:
        raise RuntimeError(f"expected 60 scheduled runs, found {len(schedule)}")
    if set(Counter((int(row["block"]), int(row["physical_gpu"]), row["policy"]) for row in schedule).values()) != {1}:
        raise RuntimeError("policy is not balanced exactly once per trace block and GPU")

    output = []
    for scheduled in schedule:
        block = int(scheduled["block"])
        replicate = int(scheduled["replicate"])
        policy = scheduled["policy"]
        events = [row for row in requests if int(row["block"]) == block and int(row["replicate"]) == replicate]
        resident: list[str] = []
        last_use: dict[str, tuple[int, int]] = {}
        misses = evictions = 0
        for index, event in enumerate(events):
            target = event["logical_model_id"]
            day = int(event["day"])
            if target not in resident:
                misses += 1
                if len(resident) >= 4:
                    if policy == "lru":
                        victim = min(resident, key=lambda model: last_use.get(model, (-1, -1)))
                    elif policy == "ttl_3d":
                        idle = {model: day - last_use.get(model, (day - 100, 0))[0] for model in resident}
                        expired = [model for model in resident if idle[model] >= 3]
                        pool = expired or resident
                        victim = max(pool, key=lambda model: (idle[model], model))
                    elif policy in {"coldguard_risk", "direct_risk"}:
                        victim = min(resident, key=lambda model: (risks.get((day, model), {}).get(policy, 1.0), last_use.get(model, (-1, -1))))
                    else:
                        def next_use(model: str) -> int:
                            return next((future for future in range(index + 1, len(events)) if events[future]["logical_model_id"] == model), 10**9)
                        victim = max(resident, key=lambda model: (next_use(model), model))
                    resident.remove(victim)
                    evictions += 1
                resident.append(target)
            last_use[target] = (day, int(event["position_within_day"]))
        output.append({
            "run_order": int(scheduled["run_order"]),
            "block": block,
            "parallel_round": int(scheduled["parallel_round"]),
            "physical_gpu": int(scheduled["physical_gpu"]),
            "replicate": replicate,
            "request_seed": int(scheduled["request_seed"]),
            "policy": policy,
            "requests": len(events),
            "logical_cache_misses": misses,
            "logical_evictions": evictions,
            "logical_hit_rate": (len(events) - misses) / len(events),
        })
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    aggregate = Counter()
    for row in output:
        aggregate[(row["block"], row["policy"], "requests")] += row["requests"]
        aggregate[(row["block"], row["policy"], "misses")] += row["logical_cache_misses"]
    for block in (1, 2, 3):
        for policy in ["coldguard_risk", "direct_risk", "lru", "ttl_3d", "oracle"]:
            print({
                "block": block,
                "policy": policy,
                "requests_across_four_repetitions": aggregate[(block, policy, "requests")],
                "logical_cache_misses_across_four_repetitions": aggregate[(block, policy, "misses")],
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
