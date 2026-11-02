#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from coldguard.chutes_external import build_daily_landmarks, load_model_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build right-censored Chutes landmarks")
    parser.add_argument(
        "--daily",
        type=Path,
        default=Path("data/external/chutes/derived/chute_daily.parquet"),
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("data/external/chutes/metadata/chute_models.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/chutes/derived/cold_landmarks.parquet"),
    )
    args = parser.parse_args()
    daily = pd.read_parquet(args.daily)
    metadata = load_model_metadata(args.models)
    landmarks, audit = build_daily_landmarks(daily, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    landmarks.to_parquet(args.output, compression="zstd", index=False)
    audit["landmark_output"] = str(args.output.resolve())
    audit["landmark_output_bytes"] = args.output.stat().st_size
    split_audit = (
        landmarks.groupby("split", as_index=False)
        .agg(
            landmarks=("episode_id", "size"),
            episodes=("episode_id", "nunique"),
            chutes=("chute_id", "nunique"),
            mapped_rate=("metadata_mapped", "mean"),
            revival_rate_1d=("revives_1d", "mean"),
            revival_rate_7d=("revives_7d", "mean"),
            revival_rate_30d=("revives_30d", "mean"),
        )
    )
    split_audit.to_csv(args.output.parent / "landmark_split_audit.csv", index=False)
    (args.output.parent / "landmark_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
