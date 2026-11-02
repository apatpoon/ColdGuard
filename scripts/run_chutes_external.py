#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from coldguard.chutes_external import run_chutes_external


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ColdGuard on the Chutes yearly trace")
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=Path("data/external/chutes/derived/cold_landmarks.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/chutes_external")
    )
    args = parser.parse_args()
    print(run_chutes_external(args.landmarks, args.output))


if __name__ == "__main__":
    main()
