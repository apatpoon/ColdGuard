# Reproducibility guide

## 1. Environment

```bash
uv sync --all-extras --locked
uv run pytest
```

`uv.lock` freezes the CPU/offline environment. Hardware replay dependencies are intentionally documented separately because the vLLM, CUDA, PyTorch, and driver versions must match the target GPU host.

## 2. Data-independent verification

Generate a deterministic synthetic deployment-day table and exercise the principal pipeline:

```bash
uv run coldguard synthetic --output data/synthetic/deployment_daily.parquet \
  --deployments 200 --days 180 --seed 20261027
uv run coldguard episodes --config config/synthetic.yaml
uv run coldguard pilot --config config/synthetic.yaml
```

The unit suite additionally checks censoring, multi-horizon coherence, clustered risk, qualification policies, simulation, and hardware-design invariants.

## 3. Offline trace analysis

1. Acquire the provider data described in [`DATA.md`](DATA.md).
2. Review and, if necessary, update only the path fields in `config/pilot.yaml` and `config/pilot_strict.yaml`.
3. Run `bash scripts/reproduce_all.sh --from-raw` to rebuild intermediates, or `--from-interim` when the deployment-day and episode tables already exist.
4. Inspect the generated claim-verification JSON before treating any figure as paper evidence.

Some analyses are intentionally separate entry points because Chutes preprocessing and resampling studies have materially different resource requirements. Their names are explicit under `scripts/`.

## 4. Controlled hardware evidence

The frozen A100 package is under `hardware/a100_stateful/`. Follow [`HARDWARE_REPLAY.md`](HARDWARE_REPLAY.md) and the package protocol. The compact results in `results/a100/` match the completed 60-run package and are included to make the manuscript tables auditable without redistributing the large telemetry archive.

## 5. Evidence interpretation

- The 12 block-by-GPU/request-order repetitions are technical repetitions, not 12 independent production traces.
- The low-pressure block is a preregistered negative control.
- The Oracle uses future request order and is descriptive only.
- A tie with direct risk in the frozen A100 replay supports the downstream ranking interface; it does not establish superiority over direct risk.
