from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/revision_round2/loading_event_sensitivity/event_level_transfer_scenarios.csv"
CURVE = ROOT / "artifacts/shadow_gpu_analysis/service_curve.json"
OUT = ROOT / "artifacts/shadow_gpu_trace_bridge"


def _p95(values: pd.Series) -> float:
    return float(np.quantile(values.to_numpy(dtype=float), 0.95)) if len(values) else float("nan")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(INPUT)
    curve = json.loads(CURVE.read_text(encoding="utf-8"))
    low = float(curve["supported_params_b_min"])
    high = float(curve["supported_params_b_max"])
    events["hardware_curve_in_domain"] = events["params_b"].between(low, high, inclusive="both")
    events["rtx3090_warm_cache_load_s"] = np.where(
        events["hardware_curve_in_domain"],
        float(curve["params_intercept_s"])
        + float(curve["slope_s_per_billion_params"]) * events["params_b"],
        np.nan,
    )
    # Existing prespecified service-equation value, compared only on the same in-domain events.
    events["prior_fp16_16gibs_2s_s"] = events["fp16_weight_added_s"]

    rows: list[dict[str, object]] = []
    for (definition, method), group in events.groupby(["cold_definition", "method"]):
        known = group[group["params_b"].notna()]
        in_domain = group[group["hardware_curve_in_domain"]]
        rows.append(
            {
                "cold_definition": definition,
                "method": method,
                "events_total": int(len(group)),
                "events_params_known": int(len(known)),
                "events_hardware_curve_in_domain": int(len(in_domain)),
                "in_domain_fraction_total": float(len(in_domain) / len(group)) if len(group) else np.nan,
                "in_domain_fraction_known": float(len(in_domain) / len(known)) if len(known) else np.nan,
                "rtx3090_curve_p50_s": float(in_domain["rtx3090_warm_cache_load_s"].median()),
                "rtx3090_curve_p95_s": _p95(in_domain["rtx3090_warm_cache_load_s"]),
                "rtx3090_curve_max_s": float(in_domain["rtx3090_warm_cache_load_s"].max()),
                "prior_service_equation_p95_same_events_s": _p95(
                    in_domain["prior_fp16_16gibs_2s_s"]
                ),
                "out_of_domain_known_events": int(len(known) - len(in_domain)),
                "missing_parameter_events": int(len(group) - len(known)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["cold_definition", "method"])
    events.to_csv(OUT / "event_bridge.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    design = {
        "schema": "coldguard.shadow_gpu_trace_bridge.v1",
        "analysis_mode": "descriptive sensitivity",
        "hardware_curve_scope": curve["scope"],
        "supported_params_b": [low, high],
        "extrapolation_policy": "no extrapolation outside the measured 0.5B--7B range",
        "queueing": "not included; per-event isolated load primitive only",
        "production_claim": "not authorized",
    }
    (OUT / "design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")

    strict = summary[summary["cold_definition"] == "sm_le_0.1_pct"]
    lines = [
        "# RTX 3090 实测曲线与 trace 恢复事件的桥接",
        "",
        "## 结论",
        "",
        "只在实测的 0.5B--7B 参数范围内应用硬件曲线，不对 30B、32B、82B 或缺失参数事件外推。"
        "因此该分析是局部硬件标定，不替代完整事件集的参数化情景分析。",
        "",
        "| Cold definition | Method | In-domain / total | In-domain p95 (measured curve) | Prior FP16 equation p95 on same events | Out-of-domain known | Missing params |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.cold_definition} | {row.method} | "
            f"{row.events_hardware_curve_in_domain}/{row.events_total} | "
            f"{row.rtx3090_curve_p95_s:.3f} s | "
            f"{row.prior_service_equation_p95_same_events_s:.3f} s | "
            f"{row.out_of_domain_known_events} | {row.missing_parameter_events} |"
        )
    strict_h = strict[strict["method"] == "hazard_hierarchical"].iloc[0]
    strict_d = strict[strict["method"] == "direct_monotone"].iloc[0]
    lines.extend(
        [
            "",
            "严格三天动作集的两个方法均有 5 个恢复事件落在 0.5B--7B 实测域内。"
            f"在这些相同事件上，RTX 3090 曲线给出的 p95 为 {strict_h.rtx3090_curve_p95_s:.3f} s（hazard）"
            f"和 {strict_d.rtx3090_curve_p95_s:.3f} s（projected direct）；"
            "两组域内最大事件均为 7B，但有限事件组成不同，因此 p95 略有差异。",
            "",
            "其余已知事件超出单卡实验的模型尺寸支持域，必须继续保留为参数化情景；"
            "这正是不能把本次单卡证据写成全模型族生产验证的原因。",
            "",
        ]
    )
    (OUT / "report_CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
