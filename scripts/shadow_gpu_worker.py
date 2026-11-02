from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--hold-timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    started_wall = utc_now()
    started_monotonic = time.monotonic()
    payload: dict[str, object] = {
        "schema": "coldguard.shadow_gpu_worker.v1",
        "status": "RUNNING",
        "model_id": args.model_id,
        "model_path": str(args.model_path.resolve()),
        "seed": args.seed,
        "dtype": args.dtype,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "started_at": started_wall,
        "pid": os.getpid(),
    }
    try:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

        payload["environment"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        }

        tokenizer_start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=False
        )
        tokenizer_end = time.monotonic()

        model_start = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        model.eval()
        torch.cuda.synchronize()
        model_end = time.monotonic()
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        payload["after_load"] = {
            "torch_allocated_mib": torch.cuda.memory_allocated(0) / 1024**2,
            "torch_reserved_mib": torch.cuda.memory_reserved(0) / 1024**2,
            "device_memory_used_mib": (total_bytes - free_bytes) / 1024**2,
        }

        inputs = tokenizer(args.prompt, return_tensors="pt")
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        generation_start = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        generation_end = time.monotonic()
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        output_tokens = int(generated.shape[-1] - prompt_tokens)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

        payload.update(
            {
                "status": "COMPLETED",
                "tokenizer_load_s": tokenizer_end - tokenizer_start,
                "model_load_s": model_end - model_start,
                "load_total_s": model_end - tokenizer_start,
                "first_request_s": generation_end - generation_start,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "decoded_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
                "decoded_text": decoded,
                "peak_torch_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
                "peak_torch_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    if payload["status"] == "COMPLETED" and args.stop_file is not None:
        payload["worker_state"] = "READY_HELD"
        payload["ready_at"] = utc_now()
        write_json(args.output, payload)
        hold_deadline = time.monotonic() + args.hold_timeout_s
        while not args.stop_file.exists() and time.monotonic() < hold_deadline:
            time.sleep(0.02)
        if not args.stop_file.exists():
            payload.update(
                {
                    "status": "FAILED",
                    "worker_state": "HOLD_TIMEOUT",
                    "error_type": "TimeoutError",
                    "error": "parent did not issue stop within hold timeout",
                }
            )
        else:
            payload["worker_state"] = "STOP_OBSERVED"
            payload["stop_observed_at"] = utc_now()
    payload["ended_at"] = utc_now()
    payload["worker_wall_s"] = time.monotonic() - started_monotonic
    write_json(args.output, payload)
    return 0 if payload["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

