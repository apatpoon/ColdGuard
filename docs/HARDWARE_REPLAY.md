# Stateful A100 replay

The directory `hardware/a100_stateful/` is the frozen package used for the reported experiment. Its workload and configuration hashes match the completed session.

## Design

- four independent NVIDIA A100 GPU pools;
- six logical deployments competing for four resident slots per pool;
- five independent policies: ColdGuard, matched direct risk, LRU, TTL-3d, and Oracle;
- three temporal blocks and five balanced rounds, yielding 60 replays;
- 200 ms memory, power, host, and event telemetry;
- fixed request-order seeds and policy–GPU–seed rotation.

The replay uses the same Qwen2.5-7B BF16 artifact for every logical deployment. It tests finite-capacity ranking and recovery, not heterogeneous model-size placement.

## Safety and prerequisites

Read `hardware/a100_stateful/PROTOCOL_CN.md` and run the package preflight before executing. Confirm the CUDA driver, vLLM/PyTorch environment, storage capacity, model license, and an isolated four-GPU host. Do not run the shell launcher on a shared production host without an operator review.

## Oracle definition

Oracle is an offline reference that evicts the resident model whose next request lies furthest in the known future sequence. It is not deployable and is not included as evidence that ColdGuard predicts the future. It bounds the frozen replay and checks whether the remaining baselines behave plausibly.

## Published summaries

`results/a100/quantitative_summary.csv` reports policy totals. `telemetry_audit.json` and `evidence_audit.json` record completeness and guardrails. The raw per-sample telemetry and model weights are excluded because they are large and unnecessary for checking the reported aggregates.
