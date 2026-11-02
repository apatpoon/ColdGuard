#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${COLDGUARD_EXTERNAL_DATA_DIR:-${PROJECT_DIR}/data/external}"
DRY_RUN="${DRY_RUN:-0}"
VERIFY_SHA256="${VERIFY_SHA256:-1}"
EXTRACT_HUAWEI="${EXTRACT_HUAWEI:-1}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

CHUTES_COMMIT="dad9a641e0c8ae54b1e9704a1e9a254063b02fb1"
CHUTES_TRACE_URL="https://harvardsys-datasets.s3.us-east-1.amazonaws.com/2026_chutes_anonymized/chutes_trace.parquet"
CHUTES_TRACE_BYTES=91044147663
CHUTES_TRACE_SHA256="74833c1469577c5d1f1ef9d7393d165dab31d29950f1d76fe906d5726594e9ec"
CHUTES_TRACE_ROWS=6122413756
CHUTES_MODELS_URL="https://raw.githubusercontent.com/HarvardMadSys/chutes_workload/${CHUTES_COMMIT}/data/paper/chute_models.csv"
CHUTES_CONFIG_URL="https://raw.githubusercontent.com/HarvardMadSys/chutes_workload/${CHUTES_COMMIT}/config/workload.json"
CHUTES_README_URL="https://raw.githubusercontent.com/HarvardMadSys/chutes_workload/${CHUTES_COMMIT}/README.md"

HUAWEI_COMMIT="84a9d7727aefed486ea69e022b5ed2476d588700"
HUAWEI_README_URL="https://raw.githubusercontent.com/sir-lab/data-release/${HUAWEI_COMMIT}/README_data_release_2023.md"
HUAWEI_NAMES=(
  requests_minute
  function_delay_minute
  platform_delay_minute
  cpu_usage_minute
  memory_usage_minute
  cpu_limit_minute
  memory_limit_minute
  instances_minute
)
HUAWEI_IDS=(
  1W9dQvqRGylfUeHJcYqWwJrpYHIsYCVq_
  1Ooentp5nUeC2qKig8QSvJAaRgk3Kix_v
  1PEHdfOdmMfzWz-MSnj5q5Zy_9kKeDRqD
  1K0zaP6rGi-eaH2yJvPAhEo6zNBOC-TwK
  1S7TnHspI4FnR5ed1eIEJWl6nlqyNw73b
  1OX2zfHJ79ACfwrzUgBoX78r4zv7uoVmN
  1COvoR8VwQnwxstOHuJj_YjlSaMv_jDLT
  15_YwzY7lcUnfytof9D6mY7TAJAdhOfDH
)
HUAWEI_BYTES=(
  40197213
  61180541
  48481148
  29718722
  20327575
  1522928
  1413539
  1600731
)

usage() {
  cat <<'EOF'
Usage: bash scripts/download_generalization_data.sh [chutes|huawei|all|verify]

Modes:
  chutes  Download and verify the full 91 GB Chutes trace and metadata.
  huawei  Download all eight Huawei private-cloud minute-level archives,
          verify them, and extract them.
  all     Download both datasets (default).
  verify  Re-run integrity checks without downloading.

Optional environment variables:
  COLDGUARD_EXTERNAL_DATA_DIR=/path   Store data outside the project.
  DRY_RUN=1                          Print actions without downloading.
  VERIFY_SHA256=0                    Skip the expensive 91 GB SHA-256 pass.
  EXTRACT_HUAWEI=0                   Keep Huawei ZIPs without extracting.
  PYTHON_BIN=/path/to/python         Python with pyarrow installed.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

file_size() {
  local path="$1"
  if stat -f '%z' "${path}" >/dev/null 2>&1; then
    stat -f '%z' "${path}"
  else
    stat -c '%s' "${path}"
  fi
}

available_bytes() {
  local path="$1"
  df -Pk "${path}" | awk 'END {printf "%.0f\n", $4 * 1024}'
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

ensure_space() {
  local directory="$1"
  local remaining_download="$2"
  local safety_bytes="$3"
  local available required
  mkdir -p "${directory}"
  available="$(available_bytes "${directory}")"
  required=$((remaining_download + safety_bytes))
  if (( available < required )); then
    echo "Insufficient free space under ${directory}." >&2
    echo "Available: ${available} bytes; required: ${required} bytes." >&2
    echo "Set COLDGUARD_EXTERNAL_DATA_DIR to a larger disk and rerun." >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local destination="$2"
  local expected_bytes="$3"
  local partial="${destination}.part"
  local current_bytes=0

  mkdir -p "$(dirname "${destination}")"
  if [[ -f "${destination}" ]]; then
    current_bytes="$(file_size "${destination}")"
    if (( current_bytes == expected_bytes )); then
      echo "Already complete: ${destination}"
      return
    fi
    echo "Existing final file has the wrong size: ${destination}" >&2
    echo "Move it aside manually before retrying." >&2
    exit 1
  fi
  if [[ -f "${partial}" ]]; then
    current_bytes="$(file_size "${partial}")"
    if (( current_bytes > expected_bytes )); then
      echo "Partial file is larger than expected: ${partial}" >&2
      echo "Move it aside manually before retrying." >&2
      exit 1
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: curl resume ${url} -> ${destination} (${expected_bytes} bytes)"
    return
  fi

  echo "Downloading with resume support: ${partial}"
  curl \
    --fail \
    --location \
    --retry 8 \
    --retry-all-errors \
    --retry-delay 5 \
    --continue-at - \
    --output "${partial}" \
    "${url}"

  current_bytes="$(file_size "${partial}")"
  if (( current_bytes != expected_bytes )); then
    echo "Size verification failed for ${partial}" >&2
    echo "Got ${current_bytes}; expected ${expected_bytes}." >&2
    exit 1
  fi
  mv "${partial}" "${destination}"
}

download_small_file() {
  local url="$1"
  local destination="$2"
  local partial="${destination}.part"
  mkdir -p "$(dirname "${destination}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: ${url} -> ${destination}"
    return
  fi
  curl --fail --location --retry 5 --retry-all-errors --output "${partial}" "${url}"
  [[ -s "${partial}" ]] || {
    echo "Downloaded metadata file is empty: ${partial}" >&2
    exit 1
  }
  mv "${partial}" "${destination}"
}

verify_chutes() {
  local directory="${DATA_ROOT}/chutes"
  local trace="${directory}/raw/chutes_trace.parquet"
  local got_size got_sha

  [[ -f "${trace}" ]] || {
    echo "Missing Chutes trace: ${trace}" >&2
    exit 1
  }
  got_size="$(file_size "${trace}")"
  [[ "${got_size}" == "${CHUTES_TRACE_BYTES}" ]] || {
    echo "BAD Chutes size: ${got_size}; expected ${CHUTES_TRACE_BYTES}" >&2
    exit 1
  }

  if [[ "${VERIFY_SHA256}" == "1" ]]; then
    echo "Computing Chutes SHA-256; this can take several minutes..."
    got_sha="$(sha256_file "${trace}")"
    [[ "${got_sha}" == "${CHUTES_TRACE_SHA256}" ]] || {
      echo "BAD Chutes SHA-256: ${got_sha}" >&2
      exit 1
    }
  else
    got_sha="SKIPPED"
  fi

  [[ -x "${PYTHON_BIN}" ]] || {
    echo "Python not executable: ${PYTHON_BIN}" >&2
    exit 1
  }
  "${PYTHON_BIN}" -c '
import sys
import pyarrow.parquet as pq

path = sys.argv[1]
expected_rows = int(sys.argv[2])
expected_columns = [
    "invocation_id", "function_name", "chute_id", "user_id",
    "rehash_round", "instance_id", "started_at", "completed_at",
    "it", "ot", "ct", "ttft",
]
metadata = pq.ParquetFile(path).metadata
columns = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
assert metadata.num_rows == expected_rows, (metadata.num_rows, expected_rows)
assert columns == expected_columns, columns
print(f"Chutes metadata OK: {metadata.num_rows} rows, {metadata.num_row_groups} row groups")
' "${trace}" "${CHUTES_TRACE_ROWS}"

  printf '%s  %s\n' "${CHUTES_TRACE_SHA256}" "raw/chutes_trace.parquet" > "${directory}/SHA256SUMS.official"
  printf 'trace_bytes=%s\ntrace_sha256=%s\nsource_commit=%s\n' \
    "${got_size}" "${got_sha}" "${CHUTES_COMMIT}" > "${directory}/download_status.txt"
  echo "Chutes verification PASS."
}

download_chutes() {
  local directory="${DATA_ROOT}/chutes"
  local trace="${directory}/raw/chutes_trace.parquet"
  local existing=0 remaining
  mkdir -p "${directory}/raw" "${directory}/metadata"
  if [[ -f "${trace}" ]]; then
    existing="$(file_size "${trace}")"
  elif [[ -f "${trace}.part" ]]; then
    existing="$(file_size "${trace}.part")"
  fi
  (( existing > CHUTES_TRACE_BYTES )) && existing=0
  remaining=$((CHUTES_TRACE_BYTES - existing))
  ensure_space "${directory}" "${remaining}" 20000000000

  download_file "${CHUTES_TRACE_URL}" "${trace}" "${CHUTES_TRACE_BYTES}"
  download_small_file "${CHUTES_MODELS_URL}" "${directory}/metadata/chute_models.csv"
  download_small_file "${CHUTES_CONFIG_URL}" "${directory}/metadata/workload.json"
  download_small_file "${CHUTES_README_URL}" "${directory}/metadata/README.upstream.md"
  if [[ "${DRY_RUN}" != "1" ]]; then
    verify_chutes
  fi
}

verify_huawei() {
  local directory="${DATA_ROOT}/huawei_faas_2023"
  local archive_dir="${directory}/archives"
  local checksum_file="${directory}/SHA256SUMS.local"
  local i name path got_size

  : > "${checksum_file}"
  for i in "${!HUAWEI_NAMES[@]}"; do
    name="${HUAWEI_NAMES[$i]}"
    path="${archive_dir}/${name}.zip"
    [[ -f "${path}" ]] || {
      echo "Missing Huawei archive: ${path}" >&2
      exit 1
    }
    got_size="$(file_size "${path}")"
    [[ "${got_size}" == "${HUAWEI_BYTES[$i]}" ]] || {
      echo "BAD Huawei size for ${name}: ${got_size}; expected ${HUAWEI_BYTES[$i]}" >&2
      exit 1
    }
    unzip -tq "${path}" >/dev/null
    shasum -a 256 "${path}" >> "${checksum_file}"
  done
  echo "Huawei ZIP verification PASS."
  echo "Local checksums: ${checksum_file}"
}

download_huawei() {
  local directory="${DATA_ROOT}/huawei_faas_2023"
  local archive_dir="${directory}/archives"
  local extracted_dir="${directory}/extracted"
  local total_bytes=0 existing=0 remaining i name id expected path url

  mkdir -p "${archive_dir}" "${directory}/metadata"
  for i in "${!HUAWEI_BYTES[@]}"; do
    total_bytes=$((total_bytes + HUAWEI_BYTES[$i]))
    path="${archive_dir}/${HUAWEI_NAMES[$i]}.zip"
    if [[ -f "${path}" ]]; then
      expected="${HUAWEI_BYTES[$i]}"
      if [[ "$(file_size "${path}")" == "${expected}" ]]; then
        existing=$((existing + expected))
      fi
    fi
  done
  remaining=$((total_bytes - existing))
  ensure_space "${directory}" "${remaining}" 2000000000

  for i in "${!HUAWEI_NAMES[@]}"; do
    name="${HUAWEI_NAMES[$i]}"
    id="${HUAWEI_IDS[$i]}"
    expected="${HUAWEI_BYTES[$i]}"
    path="${archive_dir}/${name}.zip"
    url="https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t"
    download_file "${url}" "${path}" "${expected}"
  done
  download_small_file "${HUAWEI_README_URL}" "${directory}/metadata/README_data_release_2023.md"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi

  verify_huawei
  if [[ "${EXTRACT_HUAWEI}" == "1" ]]; then
    mkdir -p "${extracted_dir}"
    for name in "${HUAWEI_NAMES[@]}"; do
      echo "Extracting ${name}.zip..."
      unzip -q -n "${archive_dir}/${name}.zip" -d "${extracted_dir}"
    done
    echo "Huawei minute-level data extracted under ${extracted_dir}"
  fi
  printf 'source_commit=%s\narchives=8\nmetric_granularity=minute\n' \
    "${HUAWEI_COMMIT}" > "${directory}/download_status.txt"
}

require_command curl
require_command shasum
require_command unzip
mkdir -p "${DATA_ROOT}"

case "${MODE}" in
  chutes)
    download_chutes
    ;;
  huawei)
    download_huawei
    ;;
  all)
    download_chutes
    download_huawei
    ;;
  verify)
    verify_chutes
    verify_huawei
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "Generalization data root: ${DATA_ROOT}"

