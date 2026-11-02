from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_daily(
    output: str | Path,
    n_deployments: int = 900,
    days: int = 180,
    seed: int = 20261027,
) -> Path:
    """Generate a deployment-day table with heterogeneous recurrent cold spells."""
    rng = np.random.default_rng(seed)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    sites = np.array(["site_0", "site_1", "site_2", "site_3"])
    roles = np.array(["prefill", "decode", "fused"])
    arches = np.array(["dense", "moe"])
    gpu_types = np.array(["gpu_a", "gpu_b", "gpu_c"])

    for deployment in range(n_deployments):
        app_group = deployment % 180
        model_group = deployment % 75
        base_group = model_group % 18
        site = str(rng.choice(sites))
        role = str(rng.choice(roles, p=[0.2, 0.35, 0.45]))
        arch = str(rng.choice(arches, p=[0.78, 0.22]))
        gpu_type = str(rng.choice(gpu_types))
        params_b = float(rng.choice([7, 14, 32, 70, 110]))
        avg_gpu_count = float(rng.choice([1, 1, 1, 2, 4]))
        average_mem = max(8.0, params_b * rng.uniform(1.2, 2.0)) * avg_gpu_count

        # Stable hierarchy effects make unseen time periods predictable without
        # making cold age alone sufficient.
        group_effect = (
            0.8 * np.sin(model_group / 6.0)
            + 0.5 * np.cos(app_group / 17.0)
            + (0.35 if role == "decode" else -0.15)
        )
        is_cold = bool(rng.random() < 0.35)
        cold_age = 1 if is_cold else 0

        for day in range(days):
            seasonal = 0.25 * np.sin(2 * np.pi * day / 28.0)
            if is_cold:
                logit_reactivate = -2.3 + group_effect + seasonal - 0.12 * min(cold_age, 30)
                reactivation_probability = 1.0 / (1.0 + np.exp(-logit_reactivate))
                if rng.random() < reactivation_probability:
                    is_cold = False
                    cold_age = 0
            else:
                enter_cold_probability = 0.045 + 0.02 * (role == "prefill")
                if rng.random() < enter_cold_probability:
                    is_cold = True
                    cold_age = 1

            if is_cold:
                sm_util = float(rng.uniform(0.0, 0.75))
                zero_fraction = float(rng.uniform(0.93, 1.0))
                cold_age += 1
            else:
                sm_util = float(rng.uniform(3.0, 55.0))
                zero_fraction = float(rng.uniform(0.0, 0.65))

            observed_hours = 24.0
            mem_gib_hours = average_mem * observed_hours * rng.uniform(0.92, 1.08)
            rows.append(
                {
                    "day": day,
                    "app_id": f"app_{app_group}",
                    "model_id": f"model_{model_group}",
                    "site_id": site,
                    "pd_role": role,
                    "base_id": f"base_{base_group}",
                    "cluster_id": f"cluster_{deployment % 12}",
                    "gpu_type": gpu_type,
                    "model_arch": arch,
                    "base_arch": arch,
                    "variant_arch": arch,
                    "params_b": params_b,
                    "instance_count": int(max(1, avg_gpu_count)),
                    "cluster_count": 1,
                    "gpu_type_count": 1,
                    "gpu_hours": avg_gpu_count * observed_hours,
                    "observed_hours": observed_hours,
                    "sm_util": sm_util,
                    "zero_fraction": zero_fraction,
                    "mem_gib_hours": mem_gib_hours,
                    "avg_mem_gib_calendar": mem_gib_hours / 24.0,
                    "avg_gpu_count": avg_gpu_count,
                    "is_cold": is_cold,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path, index=False, compression="zstd")
    return output_path

