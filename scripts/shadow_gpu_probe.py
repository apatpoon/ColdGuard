from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coldguard.shadow_gpu_probe import collect_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect a read-only RTX 3090 readiness report for ColdGuard."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "shadow_gpu" / "environment_probe.json",
    )
    args = parser.parse_args()
    report = collect_probe(args.output)
    print(json.dumps(report["readiness"], indent=2, ensure_ascii=False))
    print(f"report={args.output.resolve()}")
    return 0 if report["readiness"]["status"] == "READY_FOR_SETUP" else 2


if __name__ == "__main__":
    raise SystemExit(main())

