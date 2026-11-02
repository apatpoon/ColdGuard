from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    if args.mib <= 0:
        raise SystemExit("--mib must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # uint8 makes the requested MiB correspond directly to allocated bytes.
    allocation = torch.empty(args.mib * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    allocation.fill_(1)
    torch.cuda.synchronize()
    payload = {
        "pid": os.getpid(),
        "requested_mib": args.mib,
        "allocated_mib": torch.cuda.memory_allocated(0) / 1024**2,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    args.ready_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    while not args.stop_file.exists():
        time.sleep(0.05)
    del allocation
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

