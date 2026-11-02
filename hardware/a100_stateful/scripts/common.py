from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import platform
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_snapshot(argv: list[str], timeout_s: float = 120.0, env: dict[str, str] | None = None) -> dict[str, object]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            env=env,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_s": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.monotonic() - started,
        }


def load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != "coldguard.a100_stateful_competition.v2":
        raise ValueError(f"unexpected config schema: {payload.get('schema')}")
    return payload


def directory_size(path: Path, exclude_cache: bool = False) -> int:
    total = 0
    for item in path.rglob("*"):
        if exclude_cache and ".cache" in item.parts:
            continue
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def validate_model(model: dict[str, Any], model_root: Path) -> dict[str, Any]:
    local_path = (model_root / str(model["directory"])).resolve()
    errors: list[str] = []
    config_path = local_path / "config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        errors.append("missing_or_empty_config_json")
    shards = sorted(local_path.glob("*.safetensors"))
    if not shards:
        errors.append("no_safetensors_weights")
    incomplete = sorted(
        str(path)
        for path in local_path.rglob("*.incomplete")
        if ".cache" not in path.parts
    )
    if incomplete:
        errors.append("incomplete_files_present")
    indexed_shards: set[str] = set()
    index_files = sorted(local_path.glob("*.safetensors.index.json"))
    for index_path in index_files:
        try:
            index = read_json(index_path)
            indexed_shards.update(index.get("weight_map", {}).values())
        except Exception as exc:
            errors.append(f"invalid_index:{index_path.name}:{type(exc).__name__}")
    missing_indexed = sorted(name for name in indexed_shards if not (local_path / name).is_file())
    if missing_indexed:
        errors.append("missing_indexed_shards")
    zero_payload = sorted(
        str(path)
        for path in local_path.iterdir()
        if path.is_file() and path.stat().st_size == 0
    ) if local_path.is_dir() else []
    if zero_payload:
        errors.append("zero_byte_payload_files")
    record_path = local_path / "COLDGUARD_MODEL_RECORD.json"
    record = read_json(record_path) if record_path.is_file() else None
    metadata_revisions: set[str] = set()
    metadata_dir = local_path / ".cache" / "huggingface" / "download"
    if metadata_dir.is_dir():
        for metadata_path in metadata_dir.rglob("*.metadata"):
            try:
                first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
                if len(first_line) == 40 and all(character in "0123456789abcdef" for character in first_line.lower()):
                    metadata_revisions.add(first_line)
            except (OSError, UnicodeDecodeError, IndexError):
                continue
    metadata_hashes = {}
    for name in ["config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"]:
        path = local_path / name
        if path.is_file():
            metadata_hashes[name] = sha256_file(path)
    return {
        **model,
        "local_path": str(local_path),
        "exists": local_path.is_dir(),
        "valid": not errors,
        "errors": errors,
        "total_bytes": directory_size(local_path, exclude_cache=True) if local_path.is_dir() else 0,
        "weight_shards": len(shards),
        "weight_files": [{"name": path.name, "bytes": path.stat().st_size} for path in shards],
        "indexed_shards": len(indexed_shards),
        "missing_indexed_shards": missing_indexed,
        "incomplete_files": incomplete,
        "zero_byte_payload_files": zero_payload,
        "download_record": record,
        "huggingface_metadata_revisions": sorted(metadata_revisions),
        "metadata_sha256": metadata_hashes,
    }


def environment_snapshot(result_root: Path) -> dict[str, Any]:
    commands = {
        "nvidia_smi_query": ["nvidia-smi", "-q"],
        "nvidia_smi_xml": ["nvidia-smi", "-q", "-x"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "nvlink_status": ["nvidia-smi", "nvlink", "--status"],
        "nvcc_version": ["nvcc", "--version"],
        "os_release": ["bash", "-lc", "cat /etc/os-release"],
        "lscpu": ["lscpu"],
        "memory": ["free", "-b"],
        "disk": ["df", "-B1", str(result_root)],
        "mount": ["findmnt", "-T", str(result_root)],
        "block_devices": ["lsblk", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL"],
        "kernel": ["uname", "-a"],
        "vllm_version": ["vllm", "--version"],
        "torch_cuda_version": [
            sys.executable,
            "-c",
            "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu_count':torch.cuda.device_count()}))",
        ],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "processes": ["ps", "-eo", "pid,ppid,pgid,lstart,etime,stat,args"],
        "ulimit": ["bash", "-lc", "ulimit -a"],
    }
    return {
        "schema": "coldguard.a100_environment.v1",
        "captured_utc": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "commands": {name: command_snapshot(argv) for name, argv in commands.items()},
    }


def host_telemetry_snapshot() -> dict[str, Any]:
    """Dependency-free raw host counters for later rate/cache diagnostics."""
    output: dict[str, Any] = {}
    try:
        output["load_average"] = list(os.getloadavg())
    except OSError:
        output["load_average"] = None
    try:
        meminfo = {}
        wanted = {
            "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached",
            "SwapCached", "Active", "Inactive", "Dirty", "Writeback", "SwapTotal", "SwapFree",
        }
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in wanted:
                meminfo[f"{key}_kib"] = int(value.strip().split()[0])
        output["memory"] = meminfo
    except (OSError, ValueError, IndexError):
        output["memory"] = None
    try:
        cpu_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        output["cpu_ticks"] = {
            key: int(value)
            for key, value in zip(
                ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice"],
                cpu_line[1:],
            )
        }
    except (OSError, ValueError, IndexError):
        output["cpu_ticks"] = None
    try:
        network = {"rx_bytes": 0, "rx_errors": 0, "rx_drops": 0, "tx_bytes": 0, "tx_errors": 0, "tx_drops": 0}
        interfaces = {}
        for line in Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]:
            name, payload = line.split(":", 1)
            name = name.strip()
            if name == "lo":
                continue
            fields = [int(value) for value in payload.split()]
            row = {
                "rx_bytes": fields[0], "rx_errors": fields[2], "rx_drops": fields[3],
                "tx_bytes": fields[8], "tx_errors": fields[10], "tx_drops": fields[11],
            }
            interfaces[name] = row
            for key, value in row.items():
                network[key] += value
        output["network"] = {"aggregate": network, "interfaces": interfaces}
    except (OSError, ValueError, IndexError):
        output["network"] = None
    try:
        whole_devices = {path.name for path in Path("/sys/block").iterdir()}
        devices = {}
        for line in Path("/proc/diskstats").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 14 or fields[2] not in whole_devices:
                continue
            devices[fields[2]] = {
                "reads_completed": int(fields[3]),
                "sectors_read": int(fields[5]),
                "read_ms": int(fields[6]),
                "writes_completed": int(fields[7]),
                "sectors_written": int(fields[9]),
                "write_ms": int(fields[10]),
                "io_in_progress": int(fields[11]),
                "io_ms": int(fields[12]),
            }
        output["block_devices"] = devices
    except (OSError, ValueError, IndexError):
        output["block_devices"] = None
    return output


class MultiGpuSampler:
    def __init__(self, output: Path, interval_s: float, gpu_indices: Iterable[int]):
        import pynvml

        self.p = pynvml
        pynvml.nvmlInit()
        self.indices = list(gpu_indices)
        self.handles = {index: pynvml.nvmlDeviceGetHandleByIndex(index) for index in self.indices}
        self.output = output
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.sampling_errors: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)

    def _maybe(self, function: Any, *args: Any, scale: float = 1.0) -> float | int | None:
        try:
            return function(*args) / scale
        except (self.p.NVMLError, AttributeError, TypeError):
            return None

    def _pcie_throughput(self, handle: Any, counter_name: str) -> float | int | None:
        counter = getattr(self.p, counter_name, None)
        if counter is None:
            return None
        return self._maybe(self.p.nvmlDeviceGetPcieThroughput, handle, counter)

    def sample(self) -> dict[str, Any]:
        gpu_rows = []
        for index in self.indices:
            handle = self.handles[index]
            memory = self.p.nvmlDeviceGetMemoryInfo(handle)
            utilization = self.p.nvmlDeviceGetUtilizationRates(handle)
            processes = []
            try:
                process_rows = self.p.nvmlDeviceGetComputeRunningProcesses(handle)
            except self.p.NVMLError:
                process_rows = []
            for process in process_rows:
                used = getattr(process, "usedGpuMemory", None)
                if used is not None and int(used) > int(memory.total):
                    used = None
                processes.append({
                    "pid": int(process.pid),
                    "used_memory_mib": None if used is None else used / 1024**2,
                })
            row = {
                "gpu_index": index,
                "uuid": self._text(self.p.nvmlDeviceGetUUID(handle)),
                "name": self._text(self.p.nvmlDeviceGetName(handle)),
                "memory_used_mib": memory.used / 1024**2,
                "memory_free_mib": memory.free / 1024**2,
                "memory_total_mib": memory.total / 1024**2,
                "gpu_utilization_pct": int(utilization.gpu),
                "memory_utilization_pct": int(utilization.memory),
                "temperature_c": self._maybe(
                    self.p.nvmlDeviceGetTemperature,
                    handle,
                    self.p.NVML_TEMPERATURE_GPU,
                ),
                "power_w": self._maybe(self.p.nvmlDeviceGetPowerUsage, handle, scale=1000.0),
                "graphics_clock_mhz": self._maybe(
                    self.p.nvmlDeviceGetClockInfo, handle, self.p.NVML_CLOCK_GRAPHICS
                ),
                "memory_clock_mhz": self._maybe(
                    self.p.nvmlDeviceGetClockInfo, handle, self.p.NVML_CLOCK_MEM
                ),
                "clock_throttle_reasons_bitmask": self._maybe(
                    self.p.nvmlDeviceGetCurrentClocksThrottleReasons, handle
                ),
                "pcie_tx_kib_s": self._pcie_throughput(handle, "NVML_PCIE_UTIL_TX_BYTES"),
                "pcie_rx_kib_s": self._pcie_throughput(handle, "NVML_PCIE_UTIL_RX_BYTES"),
                "compute_processes": processes,
            }
            gpu_rows.append(row)
        return {
            "wall_time_utc": utc_now(),
            "monotonic_s": time.monotonic(),
            "gpus": gpu_rows,
            "host": host_telemetry_snapshot(),
        }

    def _loop(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as handle:
            while not self._stop.is_set():
                try:
                    row = self.sample()
                    self.samples.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                except Exception as exc:
                    error = {
                        "wall_time_utc": utc_now(),
                        "monotonic_s": time.monotonic(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    self.sampling_errors.append(error)
                    handle.write(json.dumps({"sampling_error": error}, ensure_ascii=False) + "\n")
                    handle.flush()
                self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 10))
        self.p.nvmlShutdown()


def gpu_process_map(sample: dict[str, Any]) -> dict[int, set[int]]:
    return {
        int(row["gpu_index"]): {int(process["pid"]) for process in row["compute_processes"]}
        for row in sample["gpus"]
    }


def wait_for_idle(
    sampler: MultiGpuSampler,
    gpu_indices: Iterable[int],
    maximum_temperature_c: float,
    timeout_s: float,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    required = set(gpu_indices)
    deadline = time.monotonic() + timeout_s
    last = sampler.sample()
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("stop requested while waiting for GPU idle state")
        last = sampler.sample()
        relevant = [row for row in last["gpus"] if int(row["gpu_index"]) in required]
        no_processes = all(not row["compute_processes"] for row in relevant)
        temperatures_ok = all(
            row["temperature_c"] is None or float(row["temperature_c"]) <= maximum_temperature_c
            for row in relevant
        )
        if no_processes and temperatures_ok:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"GPUs did not become safely idle: {last}")


def memory_map(sample: dict[str, Any]) -> dict[int, float]:
    return {int(row["gpu_index"]): float(row["memory_used_mib"]) for row in sample["gpus"]}


def memory_returned(
    samples: list[dict[str, Any]],
    baseline: dict[int, float],
    tolerance_mib: float,
    consecutive: int,
) -> bool:
    if len(samples) < consecutive:
        return False
    for sample in samples[-consecutive:]:
        current = memory_map(sample)
        for index, base in baseline.items():
            if current.get(index, float("inf")) > base + tolerance_mib:
                return False
    return True


def gpu_summary(
    samples: list[dict[str, Any]],
    baseline: dict[int, float],
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[dict[str, Any]]:
    rows = [
        sample
        for sample in samples
        if (start_s is None or float(sample["monotonic_s"]) >= start_s)
        and (end_s is None or float(sample["monotonic_s"]) <= end_s)
    ]
    output = []
    for index in sorted(baseline):
        gpu_rows = [
            (float(sample["monotonic_s"]), next(row for row in sample["gpus"] if int(row["gpu_index"]) == index))
            for sample in rows
        ]
        if not gpu_rows:
            continue
        memory_values = [float(row["memory_used_mib"]) for _, row in gpu_rows]
        powers = [(timestamp, row["power_w"]) for timestamp, row in gpu_rows if row["power_w"] is not None]
        energy_j = None
        if len(powers) >= 2:
            energy_j = sum(
                (right[0] - left[0]) * (float(left[1]) + float(right[1])) / 2.0
                for left, right in zip(powers, powers[1:])
            )
        output.append({
            "gpu_index": index,
            "baseline_memory_mib": baseline[index],
            "peak_memory_mib": max(memory_values),
            "peak_increment_mib": max(memory_values) - baseline[index],
            "max_gpu_utilization_pct": max(int(row["gpu_utilization_pct"]) for _, row in gpu_rows),
            "max_memory_utilization_pct": max(int(row["memory_utilization_pct"]) for _, row in gpu_rows),
            "max_temperature_c": max(
                (float(row["temperature_c"]) for _, row in gpu_rows if row["temperature_c"] is not None),
                default=None,
            ),
            "max_power_w": max(
                (float(row["power_w"]) for _, row in gpu_rows if row["power_w"] is not None),
                default=None,
            ),
            "energy_j": energy_j,
        })
    return output


def warm_model_files(model_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    files = sorted(model_path.glob("*.safetensors"))
    auxiliary = [
        path for path in [model_path / "config.json", model_path / "tokenizer.json"] if path.is_file()
    ]
    total = 0
    for path in files + auxiliary:
        with path.open("rb", buffering=0) as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                total += len(chunk)
    return {
        "requested_regime": "warm",
        "effective_regime": "warm",
        "files": len(files) + len(auxiliary),
        "bytes_read": total,
        "elapsed_s": time.monotonic() - started,
    }


def drop_page_cache() -> dict[str, Any]:
    started = time.monotonic()
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="utf-8")
        time.sleep(1.0)
        return {
            "requested_regime": "cold",
            "effective_regime": "cold",
            "drop_caches_succeeded": True,
            "elapsed_s": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "requested_regime": "cold",
            "effective_regime": "uncontrolled",
            "drop_caches_succeeded": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": time.monotonic() - started,
        }


def health_ok(url: str, timeout_s: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def stream_chat(
    url: str,
    served_model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first_content = None
    output_fragments: list[str] = []
    usage: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        if first_content is None:
                            first_content = time.monotonic()
                        output_fragments.append(str(content))
        ended = time.monotonic()
        text = "".join(output_fragments)
        return {
            "status": "COMPLETED",
            "started_monotonic_s": started,
            "ended_monotonic_s": ended,
            "ttft_s": None if first_content is None else first_content - started,
            "total_s": ended - started,
            "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "output_chars": len(text),
            "usage": usage,
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "started_monotonic_s": started,
            "ended_monotonic_s": time.monotonic(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def request_batch(
    url: str,
    served_model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_chat, url, served_model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        requests = [future.result() for future in futures]
    ended = time.monotonic()
    completed = [row for row in requests if row["status"] == "COMPLETED"]
    output_tokens = sum(int((row.get("usage") or {}).get("completion_tokens", 0)) for row in completed)
    return {
        "concurrency": concurrency,
        "started_monotonic_s": started,
        "ended_monotonic_s": ended,
        "wall_s": ended - started,
        "requests": requests,
        "completed": len(completed),
        "failed": len(requests) - len(completed),
        "aggregate_output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / (ended - started) if output_tokens else None,
    }


def terminate_process_group(process: subprocess.Popen[Any], timeout_s: float) -> dict[str, Any]:
    if process.poll() is not None:
        return {"initially_running": False, "returncode": process.returncode}
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
        return {"initially_running": True, "signal": "SIGTERM", "returncode": process.returncode}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
        return {
            "initially_running": True,
            "signal": "SIGKILL_AFTER_TIMEOUT",
            "returncode": process.returncode,
        }


def process_tree_snapshot(root_pid: int) -> dict[str, Any]:
    return {
        "root_pid": root_pid,
        "captured_utc": utc_now(),
        "ps": command_snapshot(["ps", "-eo", "pid,ppid,pgid,psr,stat,etime,args"]),
        "nvidia_smi_pmon": command_snapshot(["nvidia-smi", "pmon", "-c", "1"]),
    }


def build_schedule(config: dict[str, Any], drop_cache_supported: bool) -> list[dict[str, Any]]:
    design = config["design"]
    models = config["models"]
    seed = int(design["seed"])
    schedule: list[dict[str, Any]] = []
    sequence = 0

    warmups = []
    for model in models:
        for repeat in range(int(design["warmup_lifecycle_per_model"])):
            warmups.append({
                "role": "warmup",
                "block": repeat + 1,
                "model_id": model["model_id"],
                "cache_regime": "warm",
            })
    random.Random(seed + 100).shuffle(warmups)
    for row in warmups:
        sequence += 1
        schedule.append({"sequence": sequence, "schedule_seed": seed + 100, **row})

    regimes = (
        design["cache_regimes_if_drop_supported"]
        if drop_cache_supported
        else design["cache_regimes_fallback"]
    )
    for block in range(1, int(design["measured_blocks"]) + 1):
        treatments = [
            {
                "role": "measured",
                "block": block,
                "model_id": model["model_id"],
                "cache_regime": regime,
            }
            for model in models
            for regime in regimes
        ]
        block_seed = seed + 10_000 + block
        random.Random(block_seed).shuffle(treatments)
        for position, row in enumerate(treatments, start=1):
            sequence += 1
            schedule.append({
                "sequence": sequence,
                "position_within_block": position,
                "schedule_seed": block_seed,
                **row,
            })
    return schedule


def write_schedule_csv(path: Path, schedule: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in schedule for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(schedule)


def vllm_argv(config: dict[str, Any], model: dict[str, Any]) -> list[str]:
    server = config["server"]
    argv = [
        "vllm",
        "serve",
        str(model["local_path"]),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--served-model-name",
        str(model["model_id"]),
        "--dtype",
        str(server["dtype"]),
        "--tensor-parallel-size",
        str(model["tensor_parallel_size"]),
        "--max-model-len",
        str(server["max_model_len"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--generation-config",
        "vllm",
        "--distributed-executor-backend",
        str(server["distributed_executor_backend"]),
        "--disable-log-stats",
    ]
    if bool(server.get("enforce_eager", False)):
        argv.append("--enforce-eager")
    return argv


def sanitized_server_env(model: dict[str, Any]) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, model["physical_gpus"])),
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,COLL",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
    }


def merged_server_env(model: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(sanitized_server_env(model))
    return env


def artifacts_manifest(directory: Path, exclude: set[str] | None = None) -> dict[str, Any]:
    excluded = exclude or set()
    records = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if not path.is_file() or path.name in excluded or relative in excluded:
            continue
        records[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema": "coldguard.artifact_manifest.v1",
        "generated_utc": utc_now(),
        "artifacts": records,
    }
