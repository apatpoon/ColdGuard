# ColdGuard

ColdGuard is a qualification layer for model-residency decisions in finite-capacity Model-as-a-Service (MaaS) systems. It reconstructs right-censored model lifecycles, estimates coherent multi-horizon revival risk, and turns a proposed eviction into a time-bounded **retain / evict / abstain** decision. The repository accompanies the manuscript **“ColdGuard: Risk-Calibrated Qualification of Model Residency in Finite-Capacity MaaS Systems.”**

## What is included

- `src/coldguard/`: lifecycle reconstruction, survival and direct-risk models, calibration, qualification, and simulation logic.
- `scripts/`: analysis, audit, verification, and figure-generation entry points.
- `config/`: frozen CPU/offline and RTX 3090 configurations.
- `hardware/a100_stateful/`: the exact four-A100 stateful replay package used for the 60-run experiment.
- `results/`: compact, reviewable result summaries and audit records; no model weights or raw third-party traces.
- `tests/`: unit and design tests.

## Key evidence

The offline study uses the Alibaba MaaS trace and the independent year-long Chutes workload. The controlled hardware study comprises 60 balanced stateful replays: five policies, three temporal blocks, and four GPU/request-order repetitions. On the frozen replay workload, ColdGuard reduced eviction-induced misses by 29.2% relative to LRU and by 27.0% relative to TTL-3d; the corresponding aggregate recovery-to-first-token time fell by 14.4% and 13.4%. ColdGuard tied the matched direct-risk baseline on cache outcomes. The Oracle is a descriptive upper-bound reference, not a deployable policy.

These results are bounded to the released traces and frozen hardware design. The repository does not claim universal performance across all models, clusters, or storage paths.

## Quick start

Python 3.11 or later is required. With [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/apatpoon/ColdGuard.git
cd ColdGuard
uv sync --all-extras --locked
uv run pytest
```

Run a small data-independent smoke test:

```bash
uv run coldguard synthetic --output data/synthetic/deployment_daily.parquet \
  --deployments 200 --days 180 --seed 20261027
uv run coldguard episodes --config config/synthetic.yaml
uv run coldguard pilot --config config/synthetic.yaml
```

## Reproducing the paper evidence

Reproduction has three levels because the datasets and hardware have different access and storage requirements:

1. **Unit and synthetic checks:** no external data or GPU required.
2. **Offline trace analyses:** download provider data, preserve its license and citation terms, and follow [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
3. **Hardware replay:** requires four independent NVIDIA A100 GPU pools and the named Qwen2.5-7B BF16 artifact; see [`docs/HARDWARE_REPLAY.md`](docs/HARDWARE_REPLAY.md).

Compact published outputs are indexed in [`results/README.md`](results/README.md). Their role is verification and review, not a substitute for provider datasets.

## Data

The repository does not redistribute the Alibaba or Chutes raw traces. Obtain them from their providers:

- [Alibaba Cluster Trace v2026 MaaS](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-maas)
- [Chutes workload trace](https://github.com/HarvardMadSys/chutes_workload)

See [`docs/DATA.md`](docs/DATA.md) for expected layout and attribution responsibilities.

## Scope of the public release

The code and small derived result tables are released under the MIT License. Dataset licenses, model licenses, and provider terms remain controlling for their respective artifacts. Large raw traces, model weights, credentials, host identifiers, per-run logs, and full 200 ms telemetry are intentionally excluded.

## Citation

The manuscript is being prepared for submission. Use [`CITATION.cff`](CITATION.cff) for the current author and title metadata; a DOI and final bibliographic record will be added after publication.

## Contact

Corresponding author: Zhibin Pan (`zhibin@graduate.utm.my`).
