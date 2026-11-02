# Released result summaries

This directory contains compact derived evidence used to audit manuscript claims. It deliberately excludes raw provider data, fitted model objects, model weights, request payloads, and full telemetry.

## `alibaba/`

- `full_proposal_stream/`: risk bounds, tipping curves, summary, and automated claim checks.
- `selection_aware_coverage/`: design, selected-route coverage summary, and claim checks.

## `chutes/`

Temporal and unseen-model split audit, predictive metrics, recalibration summaries, policy frontier, and selected policy records from the independent Chutes workload.

## `rtx3090/`

Primitive load/unload, first-request recovery, service-curve, provenance, and bundle-audit summaries. These measurements support mechanism feasibility and are not treated as the stateful multi-model competition experiment.

## `a100/`

The completed 60-run stateful competition summaries, paired technical-repetition contrasts, telemetry/evidence audits, and scalable SVG figures. `quantitative_summary.csv` is the direct source for the aggregate policy table.

All percentages and comparisons should be recomputed from the released CSV/JSON rather than transcribed from prose.
