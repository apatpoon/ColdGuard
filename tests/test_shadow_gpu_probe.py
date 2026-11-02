from __future__ import annotations

import pytest

from coldguard.shadow_gpu_probe import assess_readiness, parse_nvidia_smi_csv


def test_parse_nvidia_smi_csv() -> None:
    rows = parse_nvidia_smi_csv(
        "0, NVIDIA GeForce RTX 3090, GPU-abc, 550.54.14, 24576, 412, 41\n"
    )
    assert rows == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 3090",
            "uuid": "GPU-abc",
            "driver_version": "550.54.14",
            "memory_total_mib": pytest.approx(24576.0),
            "memory_used_mib": pytest.approx(412.0),
            "temperature_c": pytest.approx(41.0),
        }
    ]


def test_parse_nvidia_smi_rejects_wrong_field_count() -> None:
    with pytest.raises(ValueError, match="expected 7"):
        parse_nvidia_smi_csv("0, RTX 3090, 24576")


def test_readiness_accepts_clean_3090_before_torch_setup(monkeypatch) -> None:
    monkeypatch.setattr("coldguard.shadow_gpu_probe.platform.system", lambda: "Linux")
    readiness = assess_readiness(
        [
            {
                "name": "NVIDIA GeForce RTX 3090",
                "memory_total_mib": 24576.0,
                "memory_used_mib": 500.0,
            }
        ],
        {"total_gib": 64.0, "available_gib": 50.0},
        200.0,
        {"status": "UNAVAILABLE"},
    )
    assert readiness["status"] == "READY_FOR_SETUP"
    assert not readiness["blockers"]
    assert any("PyTorch" in item for item in readiness["warnings"])


def test_readiness_blocks_small_disk_and_missing_gpu() -> None:
    readiness = assess_readiness([], None, 20.0, {"status": "UNAVAILABLE"})
    assert readiness["status"] == "NOT_READY"
    assert len(readiness["blockers"]) == 2

