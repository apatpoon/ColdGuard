from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd


KEYS = ["app_id", "model_id", "site_id", "pd_role"]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    daily = pd.read_parquet(args.daily, columns=[*KEYS, "day", "params_b", "is_cold"])
    daily = daily.loc[daily["day"].between(121, 150) & daily["params_b"].eq(7.0)].copy()

    candidates = []
    for key, group in daily.groupby(KEYS, sort=False):
        group = group.sort_values("day")
        active = (~group["is_cold"]).astype(int).tolist()
        candidates.append({
            **dict(zip(KEYS, key)),
            "params_b": 7.0,
            "observed_days": int(group["day"].nunique()),
            "active_days": int(sum(active)),
            "cold_days": int(group["is_cold"].sum()),
            "state_transitions": int(sum(left != right for left, right in zip(active, active[1:]))),
        })
    frame = pd.DataFrame(candidates)
    frame = frame.loc[
        frame["observed_days"].eq(30)
        & frame["active_days"].ge(5)
        & frame["cold_days"].ge(5)
    ].sort_values(
        ["state_transitions", "active_days", "app_id", "model_id", "site_id"],
        ascending=[False, False, True, True, True],
    ).head(6).reset_index(drop=True)
    if len(frame) != 6:
        raise RuntimeError(f"expected six eligible logical deployments, found {len(frame)}")
    frame["logical_model_id"] = config["logical_models"]

    predictions = pd.read_parquet(args.predictions)
    base_requests: list[dict] = []
    risks: list[dict] = []
    model_rows: list[dict] = []
    for row in frame.to_dict("records"):
        mask = pd.Series(True, index=daily.index)
        pred_mask = pd.Series(True, index=predictions.index)
        for key in KEYS:
            mask &= daily[key].eq(row[key])
            pred_mask &= predictions[key].eq(row[key])
        deployment = daily.loc[mask].sort_values("day")
        scored = predictions.loc[pred_mask & predictions["day"].between(121, 150)]
        model_rows.append(row)
        for day in deployment.loc[~deployment["is_cold"], "day"]:
            base_requests.append({"day": int(day), "logical_model_id": row["logical_model_id"]})
        for score in scored.to_dict("records"):
            risks.append({
                "day": int(score["day"]),
                "logical_model_id": row["logical_model_id"],
                "coldguard_risk_3d": float(score["score_hazard_hierarchical_3d"]),
                "direct_risk_3d": float(score["score_direct_hierarchical_3d"]),
            })

    ordered_requests: list[dict] = []
    for replicate, request_seed in enumerate(config["request_order_seeds"]):
        event_index = 0
        for day in range(121, 151):
            daily_models = [row["logical_model_id"] for row in base_requests if row["day"] == day]
            random.Random(int(request_seed) + day).shuffle(daily_models)
            for position, model_id in enumerate(daily_models, start=1):
                event_index += 1
                ordered_requests.append({
                    "replicate": replicate,
                    "request_seed": int(request_seed),
                    "event_index": event_index,
                    "day": day,
                    "position_within_day": position,
                    "logical_model_id": model_id,
                    "block": 1 + (day - 121) // 10,
                })
        expected = int(config["guardrails"]["expected_requests_per_replicate"])
        if event_index != expected:
            raise RuntimeError(f"replicate {replicate}: expected {expected} requests, built {event_index}")

    # A 5x4 cyclic Latin rectangle balances policy over GPU and parallel round.
    run_schedule: list[dict] = []
    run_order = 0
    policies = list(config["policies"])
    gpus = [int(value) for value in config["physical_gpus"]]
    for block in config["blocks"]:
        for round_index in range(len(policies)):
            for gpu_position, gpu in enumerate(gpus):
                run_order += 1
                policy = policies[(round_index + gpu_position) % len(policies)]
                replicate = gpu_position
                run_schedule.append({
                    "run_order": run_order,
                    "block": int(block["block"]),
                    "start_day": int(block["start_day"]),
                    "end_day": int(block["end_day"]),
                    "parallel_round": round_index + 1,
                    "physical_gpu": gpu,
                    "replicate": replicate,
                    "request_seed": int(config["request_order_seeds"][replicate]),
                    "policy": policy,
                    "latin_policy_index": (round_index + gpu_position) % len(policies),
                })
    if len(run_schedule) != int(config["guardrails"]["expected_run_count"]):
        raise RuntimeError("run schedule size mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "logical_models.csv", model_rows)
    write_csv(args.output_dir / "request_sequence.csv", ordered_requests)
    write_csv(args.output_dir / "risk_scores.csv", risks)
    write_csv(args.output_dir / "run_schedule.csv", run_schedule)
    audit = {
        "schema": "coldguard.a100_stateful_workload.v2",
        "seed": int(config["seed"]),
        "request_order_seeds": config["request_order_seeds"],
        "selection_rule": "six exact-7B logical deployments with 30 observed days, >=5 active and cold days, ordered by state transitions then active days and stable identifiers",
        "logical_models": len(model_rows),
        "requests_per_replicate": int(config["guardrails"]["expected_requests_per_replicate"]),
        "request_rows": len(ordered_requests),
        "risk_rows": len(risks),
        "policy_runs": len(run_schedule),
        "parallel_rounds": len(config["blocks"]) * len(config["policies"]),
        "blocks_per_replicate": {
            str(block): sum(row["block"] == block and row["replicate"] == 0 for row in ordered_requests)
            for block in (1, 2, 3)
        },
    }
    (args.output_dir / "workload_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
