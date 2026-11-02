#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-daily}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw"
BASE_URL="https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-v2026-maas/data"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

download_and_extract() {
  local archive="$1"
  echo "Downloading ${archive} with resume support..."
  curl -fL --retry 5 --retry-delay 5 -C - -O "${BASE_URL}/${archive}"
  echo "Extracting ${archive} without overwriting existing files..."
  unzip -n "${archive}"
}

case "${MODE}" in
  daily)
    download_and_extract instance_daily.zip
    ;;
  weekly)
    download_and_extract instance_week_5min.zip
    download_and_extract app_metrics.csv.zip
    ;;
  all)
    download_and_extract instance_daily.zip
    download_and_extract instance_week_5min.zip
    download_and_extract app_metrics.csv.zip
    ;;
  *)
    echo "Usage: bash scripts/download_data.sh [daily|weekly|all]" >&2
    exit 2
    ;;
esac

echo "Data available under ${DATA_DIR}"

