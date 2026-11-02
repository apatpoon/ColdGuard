from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coldguard.shadow_gpu_experiment import sha256_file, utc_now, write_json  # noqa: E402


def hash_snapshot(path: Path) -> list[dict[str, object]]:
    rows = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        rows.append(
            {
                "path": str(item.relative_to(path)),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.model_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = {
        "schema": "coldguard.shadow_gpu_models.v1",
        "created_at": utc_now(),
        "source_config": str(args.config.resolve()),
        "source_config_sha256": sha256_file(args.config),
        "models": [],
    }
    for model in config["models"]:
        repository = model["repository"]
        requested_revision = model.get("revision", "main")
        if not requested_revision or requested_revision.startswith("TO_BE_"):
            requested_revision = "main"
        info = api.model_info(repository, revision=requested_revision)
        resolved_revision = info.sha
        local_dir = args.model_root / model["model_id"] / resolved_revision
        print(json.dumps({"downloading": repository, "revision": resolved_revision}))
        snapshot_path = Path(
            snapshot_download(
                repo_id=repository,
                revision=resolved_revision,
                local_dir=local_dir,
            )
        )
        files = hash_snapshot(snapshot_path)
        manifest["models"].append(
            {
                **model,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
                "local_path": str(snapshot_path.resolve()),
                "total_bytes": sum(int(row["bytes"]) for row in files),
                "files": files,
            }
        )
        write_json(args.output, manifest)
    print(json.dumps({"status": "OK", "output": str(args.output), "models": len(manifest["models"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

