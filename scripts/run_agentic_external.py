#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from coldguard.agentic_external import run_agentic_external


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FreeInference Agentic external validation")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(
            "data/external/freeinference_agentic_trace/repository/data/v1/release"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/agentic_external")
    )
    args = parser.parse_args()
    print(run_agentic_external(args.release_root, args.output))


if __name__ == "__main__":
    main()
