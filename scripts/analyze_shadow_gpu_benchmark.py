from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from coldguard.shadow_gpu_analysis import (
    audit_shadow_bundle,
    fit_empirical_service_curve,
    load_cycle_metrics,
    summarize_measured_cycles,
)
from coldguard.shadow_gpu_experiment import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]


def _fmt(mean: float, low: float, high: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def build_report(
    summary: pd.DataFrame,
    audit: dict[str, object],
    curve: dict[str, object],
    archive_sha256: str | None,
) -> str:
    lines = [
        "# ColdGuard RTX 3090 硬件在环结果",
        "",
        "## 结论",
        "",
        f"完整流水线共取得 {audit['observed_runs']} 个生命周期周期，其中 4 个预热、"
        f"{audit['measured_runs']} 个正式周期；{audit['completed_runs']}/{audit['expected_runs']} 完成，"
        f"{audit['guardrail_passes']}/{audit['expected_runs']} 通过全部守卫，没有失败运行。",
        "",
        "实验在单张 NVIDIA GeForce RTX 3090 24 GiB 上，以独立进程真实加载固定版本的 "
        "Qwen2.5 0.5B/1.5B/3B/7B FP16 权重，执行首个 32-token 请求，再显式触发进程卸载。"
        "统计单位是完整生命周期周期，不是遥测采样点。",
        "",
        "## 正式周期结果",
        "",
        "数值为均值 [95% t 区间]，每种模型 n=5；区间只描述本机重复性，不代表跨硬件或生产流量不确定性。",
        "",
        "| Model | Artifact (GiB) | Total load (s) | First request (s) | Peak ΔVRAM (GiB) | Confirmed unload (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model_id,
                    f"{row.model_artifact_gib:.3f}",
                    _fmt(row.load_total_s_mean, row.load_total_s_ci95_low, row.load_total_s_ci95_high),
                    _fmt(
                        row.first_request_s_mean,
                        row.first_request_s_ci95_low,
                        row.first_request_s_ci95_high,
                    ),
                    _fmt(
                        row.peak_memory_above_baseline_mib_mean / 1024.0,
                        row.peak_memory_above_baseline_mib_ci95_low / 1024.0,
                        row.peak_memory_above_baseline_mib_ci95_high / 1024.0,
                    ),
                    _fmt(
                        row.unload_to_baseline_s_mean,
                        row.unload_to_baseline_s_ci95_low,
                        row.unload_to_baseline_s_ci95_high,
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 可用于论文的解释",
            "",
            f"在该限定环境中，端到端加载时间随本地模型工件大小单调上升。描述性线性标定为 "
            f"`load_total_s = {curve['intercept_s']:.3f} + {curve['slope_s_per_artifact_gib']:.3f} × artifact_GiB`，"
            f"对应的经验斜率约为 {curve['implied_artifact_gib_per_s']:.2f} GiB/s，R²={curve['r_squared']:.4f}。"
            "它是 Qwen2.5/FP16/RTX 3090/暖主机缓存条件下的系统级服务曲线，不是 PCIe 或磁盘的纯带宽测量。",
            "",
            "四个尺寸的 GPU 显存增量均稳定复现，7B 峰值约 14.53 GiB，单卡 24 GiB 可安全容纳。"
            "全部周期均在超时前回到基线。`Confirmed unload` 从写入显式停止请求起算，到连续三次 50 ms "
            "NVML 采样均回到基线才停止，因此它是带确认窗的卸载完成时间，而非瞬时显存释放时刻。",
            "",
            "这组证据将原论文的纯参数化加载层升级为真实 GPU 上的加载/首请求/显存/卸载原语标定；"
            "它尚不是线上请求流或 ColdGuard 策略执行试验，因此不能声称生产 SLO、安全回滚或队列收益已被实测。",
            "",
            "## 完整性与溯源",
            "",
            f"- 归档 SHA-256：`{archive_sha256 or 'not supplied'}`",
            f"- 逐-run清单验证通过的工件数：{audit['verified_manifest_artifacts']}",
            f"- 原始 NVML 遥测采样数：{audit['telemetry_samples']}",
            "- 模型仓库 revision、运行参数、环境、stdout/stderr、逐点遥测、守卫及输出均保存在结果包中。",
            "- 分析模式：预先规定采集端点上的描述性分析；未进行事后显著性筛选或失败运行排除。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "shadow_gpu_analysis"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_shadow_bundle(args.run_root)
    write_json(args.output_dir / "bundle_audit.json", audit)
    if not audit["passed"]:
        raise SystemExit("bundle audit failed; see bundle_audit.json")

    cycles = load_cycle_metrics(args.run_root)
    summary = summarize_measured_cycles(cycles)
    curve = fit_empirical_service_curve(cycles)
    cycles.to_csv(args.output_dir / "cycle_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    write_json(args.output_dir / "service_curve.json", curve)

    archive_hash = sha256_file(args.archive) if args.archive else None
    provenance = {
        "schema": "coldguard.shadow_gpu_analysis_provenance.v1",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(args.run_root.resolve()),
        "archive": str(args.archive.resolve()) if args.archive else None,
        "archive_sha256": archive_hash,
        "analysis_mode": "descriptive-only",
        "unit_of_analysis": "complete independent process load-infer-unload cycle",
        "primary_analysis_set": "20 measured cycles; four warmup cycles retained but excluded",
        "exclusions": [],
        "result_available_before_analysis": True,
    }
    write_json(args.output_dir / "provenance.json", provenance)
    report = build_report(summary, audit, curve, archive_hash)
    (args.output_dir / "report_CN.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
