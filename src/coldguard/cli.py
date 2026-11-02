from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import aggregate_daily
from .episodes import build_episodes
from .pilot import run_pilot
from .survival import run_survival
from .multihorizon import run_multihorizon
from .block_ltt import run_block_ltt
from .generalization import run_generalization
from .operational import run_operational
from .robustness import run_robustness
from .synthetic import generate_synthetic_daily
from .weekly_audit import audit_weekly_labels
from .calibration_window import run_calibration_window
from .window_selector import run_window_selector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coldguard",
        description="ColdGuard risk-calibrated model-residency analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate raw Alibaba daily shards")
    aggregate.add_argument("--config", required=True)

    episodes = subparsers.add_parser("episodes", help="Build cold episodes and landmark labels")
    episodes.add_argument("--config", required=True)

    pilot = subparsers.add_parser("pilot", help="Train baselines and run risk-calibrated replay")
    pilot.add_argument("--config", required=True)

    survival = subparsers.add_parser(
        "survival", help="Run rolling discrete-hazard and dual-risk evaluation"
    )
    survival.add_argument("--config", required=True)

    multihorizon = subparsers.add_parser(
        "multihorizon", help="Run common-window multi-horizon survival evaluation"
    )
    multihorizon.add_argument("--config", required=True)

    block_ltt = subparsers.add_parser(
        "block-ltt", help="Run time-block Learn-then-Test risk screening"
    )
    block_ltt.add_argument("--config", required=True)

    generalization = subparsers.add_parser(
        "generalization", help="Audit unseen entities and worst operational groups"
    )
    generalization.add_argument("--config", required=True)

    operational = subparsers.add_parser(
        "operational", help="Use every horizon's full matured calibration and test window"
    )
    operational.add_argument("--config", required=True)

    robustness = subparsers.add_parser(
        "robustness", help="Run frozen-model rolling calibration and hierarchical fallback"
    )
    robustness.add_argument("--config", required=True)

    calibration_window = subparsers.add_parser(
        "calibration-window",
        help="Compare recent and longer calibration histories with a frozen model",
    )
    calibration_window.add_argument("--config", required=True)

    window_selector = subparsers.add_parser(
        "window-selector",
        help="Select a calibration history using calibration data only",
    )
    window_selector.add_argument("--config", required=True)

    weekly_audit = subparsers.add_parser(
        "weekly-audit", help="Validate daily cold labels against the five-minute week"
    )
    weekly_audit.add_argument("--config", required=True)

    all_steps = subparsers.add_parser("all", help="Run aggregate, episodes, and pilot")
    all_steps.add_argument("--config", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Generate a synthetic deployment-day trace")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--deployments", type=int, default=900)
    synthetic.add_argument("--days", type=int, default=180)
    synthetic.add_argument("--seed", type=int, default=20261027)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "aggregate":
        print(aggregate_daily(args.config))
    elif args.command == "episodes":
        episode_path, landmark_path = build_episodes(args.config)
        print(episode_path)
        print(landmark_path)
    elif args.command == "pilot":
        print(run_pilot(args.config))
    elif args.command == "survival":
        print(run_survival(args.config))
    elif args.command == "multihorizon":
        print(run_multihorizon(args.config))
    elif args.command == "block-ltt":
        print(run_block_ltt(args.config))
    elif args.command == "generalization":
        print(run_generalization(args.config))
    elif args.command == "operational":
        print(run_operational(args.config))
    elif args.command == "robustness":
        print(run_robustness(args.config))
    elif args.command == "calibration-window":
        print(run_calibration_window(args.config))
    elif args.command == "window-selector":
        print(run_window_selector(args.config))
    elif args.command == "weekly-audit":
        print(audit_weekly_labels(args.config))
    elif args.command == "all":
        print(aggregate_daily(args.config))
        episode_path, landmark_path = build_episodes(args.config)
        print(episode_path)
        print(landmark_path)
        print(run_pilot(args.config))
    elif args.command == "synthetic":
        print(
            generate_synthetic_daily(
                Path(args.output),
                n_deployments=args.deployments,
                days=args.days,
                seed=args.seed,
            )
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()

