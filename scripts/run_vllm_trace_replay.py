from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import random
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vllm_hil_runner", ROOT / "scripts" / "run_vllm_hil_session.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def load_events(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, object] = dict(row)
            for key in [
                "action_day",
                "revival_day",
                "params_b",
                "representative_params_b",
                "mapping_absolute_error_b",
                "model_gib",
                "exposure_gib_hours",
            ]:
                converted[key] = float(converted[key])
            for key in ["selected_by_coldguard", "selected_by_direct"]:
                converted[key] = parse_bool(str(converted[key]))
            rows.append(converted)
    if not rows or len({row["trace_event_id"] for row in rows}) != len(rows):
        raise ValueError("trace events must be non-empty and uniquely identified")
    return rows


def trace_schedule(
    events: list[dict[str, object]], config: dict[str, object], session: int
) -> list[dict[str, object]]:
    design = config["design"]
    if session < 1 or session > int(design["sessions"]):
        raise ValueError("session is outside configured range")
    cache_regimes = list(design["factors"]["cache_regimes"])
    treatments = [
        {**event, "cache_regime": cache, "pressure_mib": 0}
        for event in events
        for cache in cache_regimes
    ]
    block_seed = int(design["seed"]) + 9_000_000 + session
    random.Random(block_seed).shuffle(treatments)
    schedule = []
    for sequence, row in enumerate(treatments, start=1):
        schedule.append({
            "session": session,
            "role": "trace_measured",
            "block": 1,
            "position_within_block": sequence,
            "sequence": sequence,
            "block_seed": block_seed,
            **row,
        })
    return schedule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # A trace revival is one cold recovery request. Concurrency effects are estimated
    # in the separate full-factorial service matrix, not pseudoreplicated per event.
    config = copy.deepcopy(config)
    config["measurement"]["concurrency_levels"] = [1]
    config["measurement"]["request_repetitions"] = 1
    config["experiment_id"] = "CG-HIL-VLLM-3090-TRACE-03"
    models = RUNNER.load_manifest(args.model_manifest)
    events = load_events(args.events)
    schedule = trace_schedule(events, config, args.session)
    required = {str(row["model_id"]) for row in schedule}
    missing = sorted(required - set(models))
    if missing:
        raise ValueError(f"model manifest is missing: {missing}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "runs").mkdir(exist_ok=True)
    (args.output_root / "experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(args.model_manifest, args.output_root / "model_manifest.json")
    shutil.copy2(args.events, args.output_root / "trace_replay_events.csv")
    RUNNER.write_schedule(args.output_root / "schedule.csv", schedule)
    environment_path = args.output_root / "environment.json"
    if not environment_path.exists():
        RUNNER.write_json(environment_path, RUNNER.environment_snapshot())
    results = []
    for row in schedule:
        result = RUNNER.run_one(row, config, models[str(row["model_id"])], args.output_root)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        RUNNER.write_json(args.output_root / "run_index.json", results)
    failed = sum(row["status"] != "COMPLETED" for row in results)
    RUNNER.write_json(args.output_root / "session_completion.json", {
        "schema": "coldguard.vllm_trace_replay_session.v1",
        "session": args.session,
        "completed_at": RUNNER.utc_now(),
        "unique_trace_events": len(events),
        "planned_runs": len(schedule),
        "completed_runs": len(schedule) - failed,
        "failed_runs": failed,
        "passed": failed == 0,
    })
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

