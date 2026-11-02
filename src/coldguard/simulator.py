from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import simpy


SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class SimulationConfig:
    """Bounded logical-pool loading scenario for trace-driven replay."""

    bandwidth_gib_per_s: float
    load_slots: int
    fixed_overhead_s: float = 2.0
    arrival_spread_s: float = SECONDS_PER_DAY
    slo_thresholds_s: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 30.0)
    seed: int = 0
    seconds_per_day: float = SECONDS_PER_DAY

    def __post_init__(self) -> None:
        if not np.isfinite(self.bandwidth_gib_per_s) or self.bandwidth_gib_per_s <= 0:
            raise ValueError("bandwidth_gib_per_s must be positive and finite")
        if self.load_slots < 1:
            raise ValueError("load_slots must be at least one")
        if not np.isfinite(self.fixed_overhead_s) or self.fixed_overhead_s < 0:
            raise ValueError("fixed_overhead_s must be non-negative and finite")
        if (
            not np.isfinite(self.arrival_spread_s)
            or self.arrival_spread_s <= 0
            or self.arrival_spread_s > self.seconds_per_day
        ):
            raise ValueError("arrival_spread_s must be in (0, seconds_per_day]")
        if not self.slo_thresholds_s or any(
            (not np.isfinite(value) or value <= 0) for value in self.slo_thresholds_s
        ):
            raise ValueError("slo thresholds must be positive and finite")


def _stable_fraction(seed: int, action_id: object) -> float:
    payload = f"{seed}|{action_id}|revival-offset".encode("utf-8")
    raw = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(raw, "big") / float(2**64)


def _metric_suffix(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(values, probability)) if values else 0.0


def _required_columns(frame: pd.DataFrame) -> None:
    required = {
        "action_id",
        "action_day",
        "model_gib",
        "exposure_gib_hours",
        "revives_within_horizon",
        "revival_day",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"simulation actions are missing columns: {missing}")


def simulate_policy(
    actions: pd.DataFrame,
    config: SimulationConfig,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Replay one policy under a finite logical loading-pool scenario.

    Each selected cold episode contributes one unload action. A revival inside the
    declared horizon creates one representative reload event. The public trace is
    daily, so a stable seed-specific within-day offset supplies event order while
    preserving common random numbers across policies.
    """

    _required_columns(actions)
    frame = actions.copy()
    for column in ["action_day", "model_gib", "exposure_gib_hours"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["action_id"].duplicated().any():
        raise ValueError("action_id must be unique within a policy replay")
    if (frame["model_gib"] < 0).any() or (frame["exposure_gib_hours"] < 0).any():
        raise ValueError("model size and exposure must be non-negative")

    revives = frame["revives_within_horizon"].astype(bool)
    if frame.loc[revives, "revival_day"].isna().any():
        raise ValueError("reviving actions require revival_day")

    result: dict[str, object] = dict(metadata or {})
    result.update(
        {
            "seed": int(config.seed),
            "bandwidth_gib_per_s": float(config.bandwidth_gib_per_s),
            "load_slots": int(config.load_slots),
            "fixed_overhead_s": float(config.fixed_overhead_s),
            "arrival_spread_s": float(config.arrival_spread_s),
            "actions": int(len(frame)),
            "reloads": int(revives.sum()),
            "successful_actions": int((~revives).sum()),
            "simulated_released_gib_hours": float(frame["exposure_gib_hours"].sum()),
            "safe_released_gib_hours": float(
                frame.loc[~revives, "exposure_gib_hours"].sum()
            ),
            "revival_exposure_gib_hours": float(
                frame.loc[revives, "exposure_gib_hours"].sum()
            ),
            "reload_volume_gib": float(frame.loc[revives, "model_gib"].sum()),
            "action_risk": float(revives.mean()) if len(frame) else 0.0,
            "exposure_risk": float(
                frame.loc[revives, "exposure_gib_hours"].sum()
                / frame["exposure_gib_hours"].sum()
            )
            if frame["exposure_gib_hours"].sum() > 0
            else 0.0,
        }
    )

    if not revives.any():
        result.update(
            {
                "mean_queue_wait_s": 0.0,
                "p95_queue_wait_s": 0.0,
                "mean_added_latency_s": 0.0,
                "p50_added_latency_s": 0.0,
                "p95_added_latency_s": 0.0,
                "p99_added_latency_s": 0.0,
                "max_queue_length": 0,
                "load_slot_utilization": 0.0,
                "unfinished_reloads": 0,
            }
        )
        for threshold in config.slo_thresholds_s:
            suffix = _metric_suffix(threshold)
            result[f"reload_slo_violation_rate_{suffix}s"] = 0.0
            result[f"action_slo_violation_rate_{suffix}s"] = 0.0
        return result

    env = simpy.Environment()
    loader = simpy.Resource(env, capacity=config.load_slots)
    origin_day = float(frame["action_day"].min())
    queue_waits: list[float] = []
    added_latencies: list[float] = []
    finishes: list[float] = []
    service_total = 0.0
    max_queue_length = 0

    def reload_process(row: object, arrival_s: float):
        nonlocal max_queue_length, service_total
        yield env.timeout(max(0.0, arrival_s - env.now))
        request_time = float(env.now)
        with loader.request() as request:
            max_queue_length = max(max_queue_length, len(loader.queue))
            yield request
            wait = float(env.now) - request_time
            service = config.fixed_overhead_s + float(row.model_gib) / config.bandwidth_gib_per_s
            service_total += service
            yield env.timeout(service)
            queue_waits.append(wait)
            added_latencies.append(wait + service)
            finishes.append(float(env.now))

    arrivals: list[float] = []
    service_upper = 0.0
    for row in frame.loc[revives].sort_values(["revival_day", "action_id"]).itertuples():
        if hasattr(row, "revival_time_seconds") and np.isfinite(row.revival_time_seconds):
            arrival_s = float(row.revival_time_seconds)
        else:
            arrival_key = getattr(row, "arrival_key", row.action_id)
            within_day = _stable_fraction(config.seed, arrival_key) * config.arrival_spread_s
            arrival_s = (float(row.revival_day) - origin_day) * config.seconds_per_day + within_day
        arrivals.append(arrival_s)
        service_upper = max(
            service_upper,
            config.fixed_overhead_s + float(row.model_gib) / config.bandwidth_gib_per_s,
        )
        env.process(reload_process(row, arrival_s))

    # Finite worst-case FIFO drain bound: all reloads may arrive together and use
    # one slot. The extra second avoids the half-open numeric run boundary.
    stop_time = max(arrivals) + len(arrivals) * service_upper + service_upper + 1.0
    env.run(until=stop_time)
    unfinished = int(len(arrivals) - len(added_latencies))
    if unfinished:
        raise RuntimeError(f"bounded simulation ended with {unfinished} unfinished reloads")

    span = max(finishes) - min(arrivals) if finishes else 0.0
    utilization = service_total / (config.load_slots * span) if span > 0 else 0.0
    result.update(
        {
            "mean_queue_wait_s": float(np.mean(queue_waits)),
            "p95_queue_wait_s": _quantile(queue_waits, 0.95),
            "mean_added_latency_s": float(np.mean(added_latencies)),
            "p50_added_latency_s": _quantile(added_latencies, 0.50),
            "p95_added_latency_s": _quantile(added_latencies, 0.95),
            "p99_added_latency_s": _quantile(added_latencies, 0.99),
            "max_queue_length": int(max_queue_length),
            "load_slot_utilization": float(utilization),
            "unfinished_reloads": unfinished,
        }
    )
    latency_array = np.asarray(added_latencies, dtype=float)
    for threshold in config.slo_thresholds_s:
        violation_count = int(np.sum(latency_array > threshold))
        suffix = _metric_suffix(threshold)
        result[f"reload_slo_violation_rate_{suffix}s"] = violation_count / len(latency_array)
        result[f"action_slo_violation_rate_{suffix}s"] = (
            violation_count / len(frame) if len(frame) else 0.0
        )
    return result


def replay_scenarios(
    actions: pd.DataFrame,
    configs: Iterable[SimulationConfig],
    *,
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [simulate_policy(actions, config, metadata=metadata) for config in configs]
    )
