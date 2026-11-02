#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from coldguard.gentd_external import run_gentd_external


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GenTD26 external GPU-memory validation")
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("data/external/gentd26/extracted"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/gentd_external")
    )
    args = parser.parse_args()
    print(run_gentd_external(args.extracted_root, args.output))


if __name__ == "__main__":
    main()
