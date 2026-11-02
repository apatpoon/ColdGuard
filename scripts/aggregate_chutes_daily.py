#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coldguard.chutes_external import aggregate_daily, verify_trace_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Chutes into a chute-day table")
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("data/external/chutes_workload/trace/chutes_trace.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/chutes/derived/chute_daily.parquet"),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="12GB")
    args = parser.parse_args()
    audit = verify_trace_metadata(args.trace)
    output = aggregate_daily(args.trace, args.output, args.threads, args.memory_limit)
    audit["daily_output"] = str(output.resolve())
    audit["daily_output_bytes"] = output.stat().st_size
    audit_path = output.parent / "trace_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    print(audit_path)


if __name__ == "__main__":
    main()
