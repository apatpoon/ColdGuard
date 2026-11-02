import pandas as pd
import pytest

from coldguard.placement import (
    ModelTemplate,
    evaluate_placement_day,
    max_distinct_block_replicas,
    reconstruct_virtual_blocks,
)


def test_model_template_uses_binary_gib_and_headroom() -> None:
    template = ModelTemplate("7B", 7.0, 2.0, 1, runtime_headroom=0.20)
    assert template.footprint_gib == pytest.approx(7e9 * 2 / 2**30 * 1.2)
    assert template.rank_gib == template.footprint_gib


def test_distinct_block_packing_enforces_tensor_parallel_devices() -> None:
    assert max_distinct_block_replicas([10.0, 10.0], rank_gib=5.0, tensor_parallel_degree=2) == 2
    assert max_distinct_block_replicas([20.0, 4.0], rank_gib=5.0, tensor_parallel_degree=2) == 0
    assert max_distinct_block_replicas([10.0, 10.0, 10.0], rank_gib=5.0, tensor_parallel_degree=2) == 3


def test_daily_bounds_are_nested_and_locality_aware() -> None:
    blocks = pd.DataFrame(
        {
            "site_id": ["a", "a", "b", "b"],
            "cluster_id": ["x", "y", "z", "z"],
            "gpu_type": ["g", "g", "g", "g"],
            "block_gib": [6.0, 6.0, 6.0, 6.0],
        }
    )
    template = ModelTemplate("toy", 10 * 2**30 / 1e9, 1.0, 1, runtime_headroom=0.0)
    result = evaluate_placement_day(blocks, template)
    assert result["global_pool_replicas"] == 2
    assert result["site_gpu_pool_replicas"] == 2
    assert result["cluster_gpu_pool_replicas"] == 1
    assert result["block_packed_replicas"] == 0


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError):
        max_distinct_block_replicas([1.0, -1.0], rank_gib=1.0, tensor_parallel_degree=1)


def test_virtual_block_reconstruction_conserves_capacity() -> None:
    instances = pd.DataFrame(
        {
            "action_id": ["a", "b"],
            "instance_id": ["i1", "i2"],
            "instance_block_gib": [12.0, 9.0],
            "average_gpu_count": [2.4, 1.2],
        }
    )
    floor_uniform = reconstruct_virtual_blocks(
        instances, count_rule="floor", allocation_profile="uniform"
    )
    ceil_concentrated = reconstruct_virtual_blocks(
        instances, count_rule="ceil", allocation_profile="concentrated"
    )
    assert len(floor_uniform) == 3
    assert len(ceil_concentrated) == 5
    assert floor_uniform.groupby("action_id")["block_gib"].sum().to_dict() == {
        "a": 12.0,
        "b": 9.0,
    }
    assert ceil_concentrated.groupby("action_id")["block_gib"].sum().to_dict() == {
        "a": 12.0,
        "b": 9.0,
    }
    assert (ceil_concentrated.groupby("action_id")["block_gib"].apply(lambda x: (x > 0).sum()) == 1).all()


def test_single_block_reconstruction_ignores_gpu_count() -> None:
    instances = pd.DataFrame(
        {
            "instance_block_gib": [8.0],
            "average_gpu_count": [4.0],
        }
    )
    blocks = reconstruct_virtual_blocks(instances, count_rule="single")
    assert len(blocks) == 1
    assert blocks.loc[0, "block_gib"] == 8.0
