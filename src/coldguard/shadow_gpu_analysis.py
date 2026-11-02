from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from coldguard.shadow_gpu_experiment import sha256_file


PRIMARY_METRICS = (
    "load_total_s",
    "model_load_s",
    "first_request_s",
    "resident_device_memory_mib",
    "peak_memory_above_baseline_mib",
    "unload_to_baseline_s",
)

NOMINAL_PARAMS_B = {
    "qwen_0_5b": 0.5,
    "qwen_1_5b": 1.5,
    "qwen_3b": 3.0,
    "qwen_7b": 7.0,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def audit_shadow_bundle(run_root: Path) -> dict[str, Any]:
    """Verify acquisition coverage, per-run guardrails, and content-bound manifests."""

    run_root = run_root.resolve()
    issues: list[dict[str, str]] = []

    def issue(code: str, detail: str) -> None:
        issues.append({"code": code, "detail": detail})

    required_root = [
        "schedule.json",
        "experiment_config.yaml",
        "model_manifest.json",
        "environment.json",
        "run_index.json",
    ]
    for name in required_root:
        if not (run_root / name).is_file():
            issue("missing_root_artifact", name)
    if issues:
        return {
            "schema": "coldguard.shadow_gpu_bundle_audit.v1",
            "run_root": str(run_root),
            "passed": False,
            "issues": issues,
        }

    config = yaml.safe_load((run_root / "experiment_config.yaml").read_text(encoding="utf-8"))
    schedule = _read_json(run_root / "schedule.json")
    run_index = _read_json(run_root / "run_index.json")
    model_manifest = _read_json(run_root / "model_manifest.json")
    model_revisions = {
        row["model_id"]: row["resolved_revision"] for row in model_manifest["models"]
    }
    model_ids = [row["model_id"] for row in config["models"]]
    warmups = int(config["measurement"]["warmup_cycles_per_model"])
    measured = int(config["measurement"]["measured_cycles_per_model"])
    expected_total = len(model_ids) * (warmups + measured)
    if len(schedule) != expected_total:
        issue("schedule_count", f"expected {expected_total}, observed {len(schedule)}")
    if len(run_index) != expected_total:
        issue("run_index_count", f"expected {expected_total}, observed {len(run_index)}")

    indexed_ids = [row["run_id"] for row in run_index]
    if len(indexed_ids) != len(set(indexed_ids)):
        issue("duplicate_run_id", "run_index contains duplicate run IDs")
    disk_ids = sorted(path.name for path in (run_root / "runs").iterdir() if path.is_dir())
    if sorted(indexed_ids) != disk_ids:
        issue("run_directory_coverage", "run_index and on-disk run directories differ")

    for role, blocks in [("warmup", warmups), ("measured", measured)]:
        for block in range(1, blocks + 1):
            members = [
                row["model_id"]
                for row in schedule
                if row["role"] == role and int(row["block"]) == block
            ]
            if sorted(members) != sorted(model_ids):
                issue(
                    "incomplete_randomized_block",
                    f"{role} block {block}: expected {sorted(model_ids)}, observed {sorted(members)}",
                )

    required_run_files = {
        "config.json",
        "environment.json",
        "telemetry.jsonl",
        "stdout.log",
        "stderr.log",
        "worker_result.json",
        "stop.requested",
        "raw_metrics.jsonl",
        "predictions.csv",
        "guardrails.json",
        "test_evidence.json",
        "manifest.json",
    }
    verified_artifacts = 0
    guardrail_passes = 0
    completed_runs = 0
    measured_runs = 0
    warmup_runs = 0
    telemetry_samples = 0
    for index_row in run_index:
        run_id = index_row["run_id"]
        run_dir = run_root / "runs" / run_id
        missing = sorted(name for name in required_run_files if not (run_dir / name).is_file())
        for name in missing:
            issue("missing_run_artifact", f"{run_id}/{name}")
        if missing:
            continue

        manifest = _read_json(run_dir / "manifest.json")
        guardrails = _read_json(run_dir / "guardrails.json")
        raw_rows = _read_jsonl(run_dir / "raw_metrics.jsonl")
        telemetry = _read_jsonl(run_dir / "telemetry.jsonl")
        if manifest.get("run_id") != run_id:
            issue("manifest_run_id", run_id)
        if index_row.get("status") != "COMPLETED" or manifest.get("status") != "COMPLETED":
            issue("run_not_completed", run_id)
        else:
            completed_runs += 1
        if not guardrails.get("passed") or not manifest.get("completion", {}).get(
            "all_guardrails_passed"
        ):
            issue("guardrail_failure", run_id)
        else:
            guardrail_passes += 1
        if len(raw_rows) != 1:
            issue("raw_metric_row_count", f"{run_id}: {len(raw_rows)}")
            continue

        raw = raw_rows[0]
        schedule_row = raw.get("schedule", {})
        if raw.get("run_id") != run_id:
            issue("raw_metric_run_id", run_id)
        if raw.get("worker_returncode") != 0:
            issue("worker_returncode", run_id)
        if raw.get("worker", {}).get("status") != "COMPLETED":
            issue("worker_status", run_id)
        if raw.get("model_revision") != model_revisions.get(raw.get("model_id")):
            issue("model_revision", run_id)
        if schedule_row.get("role") == "measured":
            measured_runs += 1
        elif schedule_row.get("role") == "warmup":
            warmup_runs += 1
        else:
            issue("unknown_run_role", run_id)

        times = [float(row["monotonic_s"]) for row in telemetry]
        if not telemetry:
            issue("empty_telemetry", run_id)
        elif any(right <= left for left, right in zip(times, times[1:])):
            issue("nonmonotonic_telemetry", run_id)
        telemetry_samples += len(telemetry)
        if int(raw.get("samples", -1)) != len(telemetry):
            issue("telemetry_sample_count", run_id)

        for name, expected in manifest.get("artifacts", {}).items():
            artifact = run_dir / name
            if not artifact.is_file():
                issue("manifest_missing_artifact", f"{run_id}/{name}")
                continue
            observed_bytes = artifact.stat().st_size
            observed_hash = sha256_file(artifact)
            if observed_bytes != int(expected["bytes"]):
                issue("artifact_size_mismatch", f"{run_id}/{name}")
            if observed_hash != expected["sha256"]:
                issue("artifact_hash_mismatch", f"{run_id}/{name}")
            if observed_bytes == int(expected["bytes"]) and observed_hash == expected["sha256"]:
                verified_artifacts += 1

    failure_files = sorted(str(path.relative_to(run_root)) for path in run_root.rglob("failure.json"))
    if failure_files:
        issue("failure_artifacts_present", ", ".join(failure_files))

    return {
        "schema": "coldguard.shadow_gpu_bundle_audit.v1",
        "run_root": str(run_root),
        "design": config["design"],
        "expected_runs": expected_total,
        "observed_runs": len(run_index),
        "completed_runs": completed_runs,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "guardrail_passes": guardrail_passes,
        "verified_manifest_artifacts": verified_artifacts,
        "telemetry_samples": telemetry_samples,
        "failure_files": failure_files,
        "passed": not issues,
        "issues": issues,
    }


def load_cycle_metrics(run_root: Path) -> pd.DataFrame:
    model_manifest = _read_json(run_root / "model_manifest.json")
    model_bytes = {row["model_id"]: int(row["total_bytes"]) for row in model_manifest["models"]}
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "runs").glob("*/raw_metrics.jsonl")):
        raw_rows = _read_jsonl(path)
        if len(raw_rows) != 1:
            raise ValueError(f"expected exactly one row in {path}")
        raw = raw_rows[0]
        worker = raw["worker"]
        schedule = raw["schedule"]
        total_bytes = model_bytes[raw["model_id"]]
        rows.append(
            {
                "run_id": raw["run_id"],
                "role": schedule["role"],
                "block": int(schedule["block"]),
                "sequence": int(schedule["sequence"]),
                "position_within_block": int(schedule["position_within_block"]),
                "model_id": raw["model_id"],
                "model_repository": raw["model_repository"],
                "model_revision": raw["model_revision"],
                "nominal_params_b": NOMINAL_PARAMS_B[raw["model_id"]],
                "model_artifact_bytes": total_bytes,
                "model_artifact_gib": total_bytes / 1024**3,
                "baseline_memory_mib": float(raw["baseline_memory_mib"]),
                "tokenizer_load_s": float(worker["tokenizer_load_s"]),
                "model_load_s": float(worker["model_load_s"]),
                "load_total_s": float(worker["load_total_s"]),
                "first_request_s": float(worker["first_request_s"]),
                "resident_device_memory_mib": float(
                    worker["after_load"]["device_memory_used_mib"]
                ),
                "peak_torch_allocated_mib": float(worker["peak_torch_allocated_mib"]),
                "peak_torch_reserved_mib": float(worker["peak_torch_reserved_mib"]),
                "peak_memory_above_baseline_mib": float(raw["peak_memory_above_baseline_mib"]),
                "unload_to_baseline_s": float(raw["unload_to_baseline_s"]),
                "peak_gpu_utilization_pct": float(raw["peak_gpu_utilization_pct"]),
                "max_temperature_c": float(raw["max_temperature_c"]),
                "mean_power_w": float(raw["mean_power_w"]),
                "max_power_w": float(raw["max_power_w"]),
                "telemetry_samples": int(raw["samples"]),
                "worker_wall_s": float(worker["worker_wall_s"]),
                "worker_returncode": int(raw["worker_returncode"]),
            }
        )
    return pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True)


def _descriptive_interval(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("descriptive interval requires at least two finite observations")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(array) - 1))
    half_width = critical * sd / math.sqrt(len(array))
    return {
        "n": int(len(array)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "cv_pct": 100.0 * sd / mean if mean != 0 else math.nan,
    }


def summarize_measured_cycles(cycles: pd.DataFrame) -> pd.DataFrame:
    measured = cycles.loc[cycles["role"] == "measured"].copy()
    rows: list[dict[str, Any]] = []
    for model_id, group in measured.groupby("model_id", sort=False):
        row: dict[str, Any] = {
            "model_id": model_id,
            "model_repository": group["model_repository"].iloc[0],
            "model_revision": group["model_revision"].iloc[0],
            "model_artifact_bytes": int(group["model_artifact_bytes"].iloc[0]),
            "model_artifact_gib": float(group["model_artifact_gib"].iloc[0]),
            "n_cycles": int(len(group)),
        }
        for metric in PRIMARY_METRICS:
            for statistic, value in _descriptive_interval(group[metric]).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_artifact_bytes").reset_index(drop=True)


def fit_empirical_service_curve(cycles: pd.DataFrame) -> dict[str, float | int | str]:
    """Fit a descriptive warm-cache service curve with model family as the scope."""

    measured = cycles.loc[cycles["role"] == "measured"].copy()
    x = measured["model_artifact_gib"].to_numpy(dtype=float)
    y = measured["load_total_s"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    slope = float(coefficients[1])
    params = measured["nominal_params_b"].to_numpy(dtype=float)
    params_design = np.column_stack([np.ones(len(params)), params])
    params_coefficients, *_ = np.linalg.lstsq(params_design, y, rcond=None)
    params_fitted = params_design @ params_coefficients
    params_residual = y - params_fitted
    params_ss_res = float(np.sum(params_residual**2))
    return {
        "schema": "coldguard.shadow_gpu_service_curve.v1",
        "status": "descriptive_calibration_only",
        "scope": "Qwen2.5 family, FP16, RTX 3090, warm host cache, isolated process",
        "unit_of_analysis": "complete independent process load-infer-unload cycle",
        "n_cycles": int(len(measured)),
        "n_model_sizes": int(measured["model_id"].nunique()),
        "intercept_s": float(coefficients[0]),
        "slope_s_per_artifact_gib": slope,
        "implied_artifact_gib_per_s": 1.0 / slope if slope > 0 else math.nan,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
        "rmse_s": math.sqrt(ss_res / len(y)),
        "params_intercept_s": float(params_coefficients[0]),
        "slope_s_per_billion_params": float(params_coefficients[1]),
        "params_r_squared": 1.0 - params_ss_res / ss_tot if ss_tot > 0 else math.nan,
        "params_rmse_s": math.sqrt(params_ss_res / len(y)),
        "supported_params_b_min": float(params.min()),
        "supported_params_b_max": float(params.max()),
    }
