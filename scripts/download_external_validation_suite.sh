#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERNAL_ROOT="${COLDGUARD_EXTERNAL_DIR:-$PROJECT_ROOT/data/external}"
TARGET="${1:-help}"

usage() {
  cat <<'EOF'
Download ColdGuard external-validation datasets from their official sources.

Usage:
  bash scripts/download_external_validation_suite.sh gentd26
  bash scripts/download_external_validation_suite.sh agentic
  bash scripts/download_external_validation_suite.sh chutes
  bash scripts/download_external_validation_suite.sh all

Optional destination override:
  COLDGUARD_EXTERNAL_DIR=/path/to/coldguard_external \
    bash scripts/download_external_validation_suite.sh chutes

Targets:
  gentd26  Alibaba GenTD26 GPU memory/utilization/loading telemetry (small)
  agentic  FreeInference 16-week pre-generated Parquet tables (~9 GB)
  chutes   Chutes one-year request trace (91 GB; large work disk required)

Set COLDGUARD_ALLOW_LOW_SPACE=1 only if you deliberately accept having too
little room for Chutes preprocessing and DuckDB temporary files.
EOF
}

available_gib() {
  local path="$1"
  mkdir -p "$path"
  df -Pk "$path" | awk 'NR==2 {printf "%.0f\n", $4/1024/1024}'
}

require_space() {
  local path="$1"
  local required="$2"
  local label="$3"
  local available
  available="$(available_gib "$path")"
  echo "$label: ${available} GiB available; ${required} GiB required"
  if [ "$available" -lt "$required" ]; then
    echo "Insufficient free space for $label at $path" >&2
    return 1
  fi
}

clone_or_update() {
  local url="$1"
  local destination="$2"
  if [ -d "$destination/.git" ]; then
    git -C "$destination" pull --ff-only
  elif [ -e "$destination" ]; then
    echo "Refusing to overwrite non-repository path: $destination" >&2
    return 1
  else
    git clone --depth 1 "$url" "$destination"
  fi
}

download_gentd26() {
  local destination="$EXTERNAL_ROOT/gentd26"
  local repository="$destination/clusterdata"
  require_space "$EXTERNAL_ROOT" 2 "GenTD26"
  mkdir -p "$destination"
  if [ -d "$repository/.git" ]; then
    git -C "$repository" pull --ff-only
  elif [ -e "$repository" ]; then
    echo "Refusing to overwrite non-repository path: $repository" >&2
    return 1
  else
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/alibaba/clusterdata.git "$repository"
    git -C "$repository" sparse-checkout set cluster-trace-v2026-GenAI
  fi

  local source="$repository/cluster-trace-v2026-GenAI"
  local extracted="$destination/extracted"
  mkdir -p "$extracted"
  for archive in "$source"/*.tar.gz; do
    tar -xzf "$archive" -C "$extracted"
  done
  (
    cd "$source"
    shasum -a 256 ./*.tar.gz ./lora_request_trace.csv > "$destination/SHA256SUMS.local"
  )
  echo "GenTD26 ready: $destination"
}

download_agentic() {
  local destination="$EXTERNAL_ROOT/freeinference_agentic_trace"
  local repository="$destination/repository"
  require_space "$EXTERNAL_ROOT" 25 "FreeInference Agentic Parquet tables"
  mkdir -p "$destination"
  clone_or_update \
    https://github.com/HarvardMadSys/freeinference_agentic_trace.git \
    "$repository"
  (
    cd "$repository"
    make setup
    .venv/bin/python -m release download --tables
  )
  echo "FreeInference Agentic tables ready: $repository/data/v1"
}

download_chutes() {
  local destination="$EXTERNAL_ROOT/chutes_workload"
  local repository="$destination/repository"
  if [ "${COLDGUARD_ALLOW_LOW_SPACE:-0}" != "1" ]; then
    require_space "$EXTERNAL_ROOT" 220 "Chutes trace plus analysis workspace"
  else
    require_space "$EXTERNAL_ROOT" 100 "Chutes trace download only"
    echo "WARNING: low-space override enabled; downstream analysis may exhaust disk." >&2
  fi
  mkdir -p "$destination"
  clone_or_update https://github.com/HarvardMadSys/chutes_workload.git "$repository"
  (
    cd "$repository"
    CHUTES_TRACE_DIR="$destination/trace" bash scripts/fetch_data.sh
  )
  echo "Chutes trace ready: $destination/trace/chutes_trace.parquet"
}

case "$TARGET" in
  gentd26)
    download_gentd26
    ;;
  agentic)
    download_agentic
    ;;
  chutes)
    download_chutes
    ;;
  all)
    download_gentd26
    download_agentic
    download_chutes
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    usage >&2
    exit 2
    ;;
esac
