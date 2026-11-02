from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd


def reconstruct_virtual_blocks(
    instance_rows: pd.DataFrame,
    *,
    count_rule: str = "round",
    allocation_profile: str = "uniform",
) -> pd.DataFrame:
    """Convert instance-level daily capacity into virtual-GPU blocks.

    ``count_rule='single'`` keeps each instance as one indivisible block.
    ``floor``, ``round``, and ``ceil`` derive a block count from the trace's
    average GPU count.  ``uniform`` divides capacity equally; ``concentrated``
    places all capacity on the first reconstructed block and zero on the rest.
    Every variant conserves instance capacity exactly up to floating error.
    """

    required = {"instance_block_gib", "average_gpu_count"}
    missing = sorted(required - set(instance_rows.columns))
    if missing:
        raise ValueError(f"instance rows are missing columns: {missing}")
    if count_rule not in {"single", "floor", "round", "ceil"}:
        raise ValueError(f"unsupported count_rule: {count_rule}")
    if allocation_profile not in {"uniform", "concentrated"}:
        raise ValueError(f"unsupported allocation_profile: {allocation_profile}")

    result = instance_rows.copy()
    capacity = pd.to_numeric(result["instance_block_gib"], errors="raise").to_numpy(float)
    average_count = pd.to_numeric(result["average_gpu_count"], errors="raise").to_numpy(float)
    if np.any(~np.isfinite(capacity)) or np.any(capacity < 0):
        raise ValueError("instance_block_gib must be finite and non-negative")
    if np.any(~np.isfinite(average_count)) or np.any(average_count <= 0):
        raise ValueError("average_gpu_count must be positive and finite")

    if count_rule == "single":
        counts = np.ones(len(result), dtype=np.int64)
    elif count_rule == "floor":
        counts = np.floor(average_count).astype(np.int64)
    elif count_rule == "ceil":
        counts = np.ceil(average_count).astype(np.int64)
    else:
        counts = np.rint(average_count).astype(np.int64)
    counts = np.maximum(counts, 1)
    result["virtual_gpu_count"] = counts
    result = result.loc[result.index.repeat(result["virtual_gpu_count"])].copy()
    group_columns = [column for column in ["cold_definition", "action_id", "day", "instance_id"] if column in result]
    if not group_columns:
        result["_instance_row"] = np.repeat(np.arange(len(instance_rows)), counts)
        group_columns = ["_instance_row"]
    result["virtual_gpu_index"] = result.groupby(group_columns, sort=False).cumcount()
    if allocation_profile == "uniform":
        result["block_gib"] = result["instance_block_gib"] / result["virtual_gpu_count"]
    else:
        result["block_gib"] = np.where(
            result["virtual_gpu_index"].eq(0), result["instance_block_gib"], 0.0
        )
    result.drop(columns=["_instance_row"], errors="ignore", inplace=True)
    return result.reset_index(drop=True)


@dataclass(frozen=True)
class ModelTemplate:
    """Weight-residency scenario used in the placement boundary experiment."""

    name: str
    parameters_billion: float
    bytes_per_parameter: float
    tensor_parallel_degree: int
    runtime_headroom: float = 0.20

    def __post_init__(self) -> None:
        if not np.isfinite(self.parameters_billion) or self.parameters_billion <= 0:
            raise ValueError("parameters_billion must be positive and finite")
        if not np.isfinite(self.bytes_per_parameter) or self.bytes_per_parameter <= 0:
            raise ValueError("bytes_per_parameter must be positive and finite")
        if self.tensor_parallel_degree < 1:
            raise ValueError("tensor_parallel_degree must be at least one")
        if not np.isfinite(self.runtime_headroom) or self.runtime_headroom < 0:
            raise ValueError("runtime_headroom must be non-negative and finite")

    @property
    def footprint_gib(self) -> float:
        raw_bytes = self.parameters_billion * 1e9 * self.bytes_per_parameter
        return float(raw_bytes / (2**30) * (1.0 + self.runtime_headroom))

    @property
    def rank_gib(self) -> float:
        return self.footprint_gib / self.tensor_parallel_degree


def max_distinct_block_replicas(
    capacities_gib: Sequence[float],
    *,
    rank_gib: float,
    tensor_parallel_degree: int,
) -> int:
    """Return the exact count of identical TP replicas supported by the blocks.

    A replica needs one equal-sized rank on each of ``tensor_parallel_degree``
    distinct virtual GPU blocks. A block may host ranks from multiple replicas.
    With identical ranks, feasibility for ``r`` replicas is
    ``sum(min(rank_slots_i, r)) >= r * TP``. Binary search therefore avoids a
    placement-order artifact while retaining the distinct-device constraint.
    """

    if not np.isfinite(rank_gib) or rank_gib <= 0:
        raise ValueError("rank_gib must be positive and finite")
    if tensor_parallel_degree < 1:
        raise ValueError("tensor_parallel_degree must be at least one")
    capacities = np.asarray(capacities_gib, dtype=float)
    if capacities.ndim != 1:
        raise ValueError("capacities_gib must be one-dimensional")
    if np.any(~np.isfinite(capacities)) or np.any(capacities < 0):
        raise ValueError("capacities_gib must be finite and non-negative")
    if capacities.size == 0:
        return 0

    tolerance = max(1e-12, rank_gib * 1e-12)
    slots = np.floor((capacities + tolerance) / rank_gib).astype(np.int64)
    slots = slots[slots > 0]
    if len(slots) < tensor_parallel_degree:
        return 0

    low = 0
    high = int(slots.sum() // tensor_parallel_degree)
    while low < high:
        candidate = (low + high + 1) // 2
        available = int(np.minimum(slots, candidate).sum())
        if available >= candidate * tensor_parallel_degree:
            low = candidate
        else:
            high = candidate - 1
    return low


def evaluate_placement_day(
    blocks: pd.DataFrame,
    template: ModelTemplate,
    *,
    site_columns: tuple[str, ...] = ("site_id", "gpu_type"),
    cluster_columns: tuple[str, ...] = ("site_id", "cluster_id", "gpu_type"),
) -> dict[str, float | int]:
    """Evaluate nested deployment bounds for one time-averaged trace day."""

    required = {"block_gib", *site_columns, *cluster_columns}
    missing = sorted(required - set(blocks.columns))
    if missing:
        raise ValueError(f"placement blocks are missing columns: {missing}")
    capacity = pd.to_numeric(blocks["block_gib"], errors="raise").to_numpy(float)
    if np.any(~np.isfinite(capacity)) or np.any(capacity < 0):
        raise ValueError("block_gib must be finite and non-negative")

    total_gib = float(capacity.sum())
    footprint = template.footprint_gib
    global_pool = int(math.floor((total_gib + 1e-12) / footprint))

    site_pool = 0
    if len(blocks):
        for _, group in blocks.groupby(list(site_columns), dropna=False, sort=False):
            site_pool += int(
                math.floor((float(group["block_gib"].sum()) + 1e-12) / footprint)
            )

    cluster_pool = 0
    block_packed = 0
    if len(blocks):
        for _, group in blocks.groupby(list(cluster_columns), dropna=False, sort=False):
            cluster_pool += int(
                math.floor((float(group["block_gib"].sum()) + 1e-12) / footprint)
            )
            block_packed += max_distinct_block_replicas(
                group["block_gib"].to_numpy(float),
                rank_gib=template.rank_gib,
                tensor_parallel_degree=template.tensor_parallel_degree,
            )

    if not global_pool >= site_pool >= cluster_pool >= block_packed:
        raise AssertionError("nested placement bounds must be monotone")
    used_gib = block_packed * footprint
    return {
        "released_gib": total_gib,
        "virtual_gpu_blocks": int(len(blocks)),
        "global_pool_replicas": global_pool,
        "site_gpu_pool_replicas": site_pool,
        "cluster_gpu_pool_replicas": cluster_pool,
        "block_packed_replicas": block_packed,
        "block_packed_used_gib": used_gib,
        "block_packed_stranded_gib": max(0.0, total_gib - used_gib),
        "block_packed_utilization": used_gib / total_gib if total_gib > 0 else 0.0,
        "placement_retention": block_packed / global_pool if global_pool > 0 else 0.0,
    }
