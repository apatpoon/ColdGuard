#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {
    "history_168h.json": "https://openinfra.freeinference.org/api/history?hours=168",
    "dashboard.json": "https://openinfra.freeinference.org/api/dashboard",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch an immutable FreeInference 7-day snapshot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/freeinference_7d/raw"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    for filename, url in SOURCES.items():
        target = args.output / filename
        temporary = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "ColdGuard-research/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        os.replace(temporary, target)
        files.append(
            {
                "filename": filename,
                "url": url,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    manifest = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (args.output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(args.output / "download_manifest.json")


if __name__ == "__main__":
    main()
