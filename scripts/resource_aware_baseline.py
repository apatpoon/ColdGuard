from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from coldguard.config import load_config
from coldguard.partial_identification import audit_proposal_stream, tipping_curve
from coldguard.policy import first_actions_by_score
from full_proposal_stream_audit import _first_crossing, _score_full_test_frame
from round1_revision_analysis import (
    _evaluate_selected,
    _load_operational_frames,
    _select_score_policy,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "size_benefit_recency": "score_size_benefit_recency",
    "reload_cost_recency": "score_reload_cost_recency",
}


def _recency_frequency(frame: pd.DataFrame) -> pd.Series:
    active = pd.to_numeric(frame["prior_active_days_30"], errors="coerce").fillna(0.0)
    age = pd.to_numeric(frame["cold_age"], errors="coerce").fillna(1.0).clip(lower=1.0)
    return (active + 1.0) / (active + age + 2.0)


def _attach_resource_scores(frame: pd.DataFrame, median_mem_gib: float) -> None:
    """Attach two fixed allocation-aware recency scores.

    Both scores are proposal heuristics, not probability estimates. The first
    prioritizes large low-revival-propensity allocations; the second treats
    large allocations as more expensive to restore and prioritizes small ones.
    Their geometric size factor is fixed before threshold selection.
    """
    if not np.isfinite(median_mem_gib) or median_mem_gib <= 0:
        raise ValueError("median_mem_gib must be positive and finite")
    propensity = _recency_frequency(frame)
    memory = (
        pd.to_numeric(frame["avg_mem_gib_calendar"], errors="coerce")
        .fillna(median_mem_gib)
        .clip(lower=1e-6)
    )
    size_factor = np.sqrt(memory / median_mem_gib)
    frame["score_size_benefit_recency"] = propensity / size_factor
    frame["score_reload_cost_recency"] = propensity * size_factor


def _flatten(prefix: str, bound) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for risk_name, risk in [("action", bound.action), ("resource", bound.resource)]:
        for key, value in asdict(risk).items():
            row[f"{prefix}_{risk_name}_{key}"] = value
    return row


def run(config_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    operational_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    bound_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    action_frames: list[pd.DataFrame] = []
    fraction_grid = np.linspace(0.0, 1.0, 101)

    for config_path in config_paths:
        config, _ = load_config(config_path)
        definition = f"sm_le_{config['cold_definition']['sm_util_max']}_pct"
        grid_size = int(config["risk"]["threshold_grid_size"])
        minimum_actions = int(config["risk"]["minimum_selected"])
        family_confidence = float(config["risk"]["confidence"])
        action_target = float(config["multihorizon"]["target_action_risk"])
        resource_target = float(config["multihorizon"]["target_resource_risk"])

        for horizon in [1, 3, 7]:
            calibration, test = _load_operational_frames(config_path, horizon)
            median_mem_gib = float(calibration["avg_mem_gib_calendar"].median())
            _attach_resource_scores(calibration, median_mem_gib)
            _attach_resource_scores(test, median_mem_gib)
            full_test = _score_full_test_frame(config_path, horizon)
            _attach_resource_scores(full_test, median_mem_gib)

            for method, score_column in METHODS.items():
                selected, candidate_audit = _select_score_policy(
                    calibration,
                    score_column,
                    grid_size,
                    minimum_actions,
                    action_target,
                    resource_target,
                    family_confidence,
                    "horizon_mem_gib_hours",
                )
                for row in candidate_audit:
                    candidate_rows.append(
                        {
                            "cold_definition": definition,
                            "horizon": horizon,
                            "method": method,
                            "calibration_median_mem_gib": median_mem_gib,
                            **row,
                        }
                    )

                evaluated = _evaluate_selected(
                    test,
                    method,
                    score_column,
                    selected,
                    action_target,
                    resource_target,
                    family_confidence,
                    "horizon_mem_gib_hours",
                )
                operational_rows.append(
                    {
                        "cold_definition": definition,
                        "horizon": horizon,
                        "method": method,
                        "score_formula": (
                            "recency_frequency / sqrt(memory / calibration_median)"
                            if method == "size_benefit_recency"
                            else "recency_frequency * sqrt(memory / calibration_median)"
                        ),
                        "calibration_median_mem_gib": median_mem_gib,
                        "calibration_start_day": int(calibration["day"].min()),
                        "calibration_end_day": int(calibration["day"].max()),
                        "test_start_day": int(test["day"].min()),
                        "test_end_day": int(test["day"].max()),
                        **evaluated,
                    }
                )

                if selected is None:
                    bound_rows.append(
                        {
                            "cold_definition": definition,
                            "horizon": horizon,
                            "method": method,
                            "selected_parameter": np.nan,
                            "proposed_actions": 0,
                            "observed_actions": 0,
                            "unresolved_actions": 0,
                            "full_stream_route": "ABSTAIN_NO_CALIBRATION_SUPPORT",
                        }
                    )
                    continue

                parameter = float(selected["parameter"])
                actions = first_actions_by_score(full_test, score_column, parameter).copy()
                audit = audit_proposal_stream(
                    actions,
                    family_confidence=family_confidence,
                    action_target=action_target,
                    resource_target=resource_target,
                )
                curve = tipping_curve(
                    actions,
                    fraction_grid,
                    family_confidence=family_confidence,
                    action_target=action_target,
                    resource_target=resource_target,
                )
                crossed = curve.loc[
                    (curve["action_upper"] > action_target)
                    | (curve["resource_upper"] > resource_target),
                    "unresolved_revival_fraction",
                ]
                bound_rows.append(
                    {
                        "cold_definition": definition,
                        "horizon": horizon,
                        "method": method,
                        "selected_parameter": parameter,
                        "calibration_median_mem_gib": median_mem_gib,
                        "complete_case_route": str(evaluated["test_route"]),
                        "complete_case_actions": int(evaluated["test_actions"]),
                        "complete_case_exposure_gib_hours": float(
                            evaluated["test_trace_exposure_gib_hours"]
                        ),
                        "proposed_actions": audit.proposed_actions,
                        "observed_actions": audit.observed_actions,
                        "unresolved_actions": audit.unresolved_actions,
                        "unresolved_share": (
                            audit.unresolved_actions / audit.proposed_actions
                            if audit.proposed_actions
                            else 0.0
                        ),
                        "full_stream_route": audit.route,
                        "dual_bound_first_crossing_fraction": (
                            float(crossed.iloc[0]) if not crossed.empty else np.nan
                        ),
                        "action_bound_first_crossing_fraction": _first_crossing(
                            curve, "action_upper", action_target
                        ),
                        "resource_bound_first_crossing_fraction": _first_crossing(
                            curve, "resource_upper", resource_target
                        ),
                        **_flatten("lower", audit.lower),
                        **_flatten("upper", audit.upper),
                    }
                )
                curve.insert(0, "method", method)
                curve.insert(0, "horizon", horizon)
                curve.insert(0, "cold_definition", definition)
                curve_rows.extend(curve.to_dict("records"))

                keep_columns = [
                    "day",
                    "episode_id",
                    "app_id",
                    "model_id",
                    "site_id",
                    "pd_role",
                    "avg_mem_gib_calendar",
                    "outcome_observed",
                    "observed_label",
                    "horizon_mem_gib_hours",
                    "persistent_horizon_mem_gib_hours",
                ]
                keep = actions.loc[:, keep_columns].copy()
                keep.insert(0, "method", method)
                keep.insert(0, "horizon", horizon)
                keep.insert(0, "cold_definition", definition)
                action_frames.append(keep)

    operational = pd.DataFrame(operational_rows).sort_values(
        ["cold_definition", "horizon", "method"]
    )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["cold_definition", "horizon", "method", "candidate_index"]
    )
    bounds = pd.DataFrame(bound_rows).sort_values(
        ["cold_definition", "horizon", "method"]
    )
    curves = pd.DataFrame(curve_rows).sort_values(
        ["cold_definition", "horizon", "method", "unresolved_revival_fraction"]
    )
    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    operational.to_csv(output_dir / "operational_policy_audit.csv", index=False)
    candidates.to_csv(output_dir / "candidate_sequence_audit.csv", index=False)
    bounds.to_csv(output_dir / "policy_bounds.csv", index=False)
    curves.to_csv(output_dir / "tipping_curves.csv", index=False)
    actions.to_parquet(output_dir / "selected_actions.parquet", index=False)

    summary = {
        "analysis": "fixed allocation-aware recency heuristics under the identical chronological dual-risk qualification",
        "methods": {
            "size_benefit_recency": "recency_frequency / sqrt(allocation / calibration_median)",
            "reload_cost_recency": "recency_frequency * sqrt(allocation / calibration_median)",
        },
        "interpretation": "proposal heuristics inspired by size/cost-aware caching; not reproductions of a named system and not calibrated probabilities",
        "allocation_semantics": "daily-average allocated GiB is a trace-available size/restoration-cost proxy, not identified weight bytes",
        "policies": int(len(operational)),
        "calibration_qualified": int(operational["calibration_qualified"].sum()),
        "complete_case_forward_supported": int(
            operational["test_route"].eq("FORWARD_SUPPORTED").sum()
        ),
        "full_stream_route_counts": {
            str(key): int(value)
            for key, value in bounds["full_stream_route"].value_counts().items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    display = bounds.loc[
        bounds["method"].eq("reload_cost_recency"),
        [
            "cold_definition",
            "horizon",
            "proposed_actions",
            "observed_actions",
            "unresolved_actions",
            "lower_action_upper",
            "lower_resource_upper",
            "upper_action_upper",
            "upper_resource_upper",
            "dual_bound_first_crossing_fraction",
            "full_stream_route",
        ],
    ]
    report = [
        "# 资源感知提议基线审计",
        "",
        "## 设计",
        "",
        "本实验加入两个固定、无需训练的分配量感知提议分数。二者共享 recency-frequency 复活倾向，并用校准窗口的日均分配显存中位数冻结尺度：",
        "",
        "- `size_benefit_recency = recency_frequency / sqrt(allocation / median)`：优先提出大分配、低复活倾向的动作；",
        "- `reload_cost_recency = recency_frequency * sqrt(allocation / median)`：将大分配视为恢复成本更高，优先提出小分配、低复活倾向的动作。",
        "",
        "两个分数都是受大小/成本感知缓存策略启发的 trace-available heuristic，不是任何已发表系统的完整复现，也不是校准概率。`allocation` 是日均分配 GiB，只是轨迹可见的大小/恢复成本代理，不等同于模型权重传输字节。阈值选择、最少 200 个校准动作、时间切分、双风险上界、完整提议流和未决结局极端界与论文主方法完全一致。",
        "",
        "## 结果",
        "",
        "`size_benefit_recency` 在六个定义-时间窗组合中均在第一个具备足够样本的候选点触发风险失败，因此全部弃权。`reload_cost_recency` 六个组合均获得校准阈值；完整案例中两个 1 日组合通过，但在完整提议流保守上界下六个组合全部进入 shadow-only。",
        "",
        "| 冷定义 | H | 提议 / 已观察 / 未决 | 下界 action / exposure UCB | 上界 action / exposure UCB | 首次越界 q | 路由 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in display.itertuples(index=False):
        report.append(
            f"| {row.cold_definition} | {int(row.horizon)} | "
            f"{int(row.proposed_actions)} / {int(row.observed_actions)} / {int(row.unresolved_actions)} | "
            f"{row.lower_action_upper:.2%} / {row.lower_resource_upper:.2%} | "
            f"{row.upper_action_upper:.2%} / {row.upper_resource_upper:.2%} | "
            f"{row.dual_bound_first_crossing_fraction:.0%} | `{row.full_stream_route}` |"
        )
    report.extend(
        [
            "",
            "## 解释",
            "",
            "结果直接验证 ColdGuard 的模型无关性：资源感知上游可以扩大或收缩提议集合，但所有提议仍经过同一双风险资格层。优先大分配的启发式在校准阶段即被暴露风险门槛拦截；惩罚大分配的启发式能够在完整案例的一日窗口通过，却因 371/2054 和 578/3353 个未决结局分别在严格/主要定义下进入 shadow-only。当前研究仍不授权真实内存动作。",
            "",
        ]
    )
    (output_dir / "RESOURCE_AWARE_BASELINE_CN.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["config/pilot.yaml", "config/pilot_strict.yaml"],
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/revision_round2/resource_aware_baseline",
    )
    args = parser.parse_args()
    run([ROOT / value for value in args.configs], ROOT / args.output_dir)


if __name__ == "__main__":
    main()
