#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from coldguard.freeinference import run_freeinference_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the predeclared FreeInference 7-day probe")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/external/freeinference_7d/raw/history_168h.json"),
    )
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=Path("data/external/freeinference_7d/raw/dashboard.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/freeinference_7d"),
    )
    args = parser.parse_args()
    report = run_freeinference_probe(args.history, args.dashboard, args.output)
    print(report)


if __name__ == "__main__":
    main()
