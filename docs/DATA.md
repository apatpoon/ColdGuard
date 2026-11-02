# Data sources and expected layout

ColdGuard uses provider-hosted datasets. Raw records are not redistributed here.

## Alibaba MaaS trace

Provider repository: [Alibaba Cluster Trace v2026 MaaS](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-maas).

After downloading the provider release, place the trace components under `data/raw/` or update the paths in `config/pilot.yaml`. Preserve the provider's README, license, citation, and checksums. The paper reconstructs deployment-day exposure and right-censored cold episodes from the released lifecycle, invocation, model-identity, and memory-exposure fields.

## Chutes workload trace

Provider repository: [Harvard MadSys Chutes workload](https://github.com/HarvardMadSys/chutes_workload).

The helper script can clone the provider repository and invoke its official data fetcher:

```bash
bash scripts/download_external_validation_suite.sh chutes
```

The default expected Parquet location is `data/external/chutes_workload/trace/chutes_trace.parquet`. The trace is large; verify available storage before downloading and retain the provider's licensing and citation materials.

## Other external validation data

`scripts/download_external_validation_suite.sh` also documents the Alibaba GenAI trace and FreeInference agentic tables used in auxiliary validation. These are not required for the principal ColdGuard claims unless the corresponding analysis is rerun.

## Data-governance boundary

Do not commit raw provider data, model weights, credentials, host names, or full operational telemetry. Derived tables in `results/` are small claim-level summaries and contain no raw request content.
