#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---from-interim}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

if [[ ! -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  echo "Missing .venv. Create it and install: uv sync --all-extras --locked" >&2
  exit 2
fi

case "${MODE}" in
  --from-raw|--from-interim) ;;
  *) echo "Usage: bash scripts/reproduce_all.sh [--from-raw|--from-interim]" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/coldguard-mpl}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"

for config in config/pilot.yaml config/pilot_strict.yaml; do
  if [[ "${MODE}" == "--from-raw" ]]; then
    run coldguard aggregate --config "${config}"
    run coldguard episodes --config "${config}"
  fi
  run coldguard pilot --config "${config}"
  run coldguard survival --config "${config}"
  run coldguard multihorizon --config "${config}"
  run coldguard generalization --config "${config}"
  run coldguard operational --config "${config}"
  run coldguard robustness --config "${config}"
  run coldguard calibration-window --config "${config}"
done

if compgen -G "data/raw/instance_week_5min/part_*.parquet" >/dev/null; then
  run coldguard weekly-audit --config config/pilot.yaml
  run coldguard weekly-audit --config config/pilot_strict.yaml
fi

run python scripts/round1_revision_analysis.py
run python scripts/validate_clustered_risk.py --replications 300
run python scripts/round1_censoring_audit.py
run python scripts/round1_censoring_sensitivity.py
run python scripts/run_trace_simulation.py
run python scripts/run_placement_analysis.py
run python scripts/figure_round1_revision.py
run python scripts/figure_trace_simulation.py
run python scripts/figure_placement_analysis.py
run pytest -q
run python scripts/verify_round1_claims.py
run python scripts/verify_trace_simulation.py
run python scripts/verify_placement_analysis.py
run python scripts/build_repro_manifest.py --hash-raw

echo "Analysis reproduction complete. See artifacts/reproducibility_manifest.json"
