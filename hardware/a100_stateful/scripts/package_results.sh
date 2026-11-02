#!/usr/bin/env bash
set -Eeuo pipefail

RESULT_DIR="${1:?result directory required}"
PARENT="$(dirname "${RESULT_DIR}")"
BASE="$(basename "${RESULT_DIR}")"
tar -C "${PARENT}" -czf "${RESULT_DIR}.tar.gz" "${BASE}"
sha256sum "${RESULT_DIR}.tar.gz" > "${RESULT_DIR}.tar.gz.sha256"
echo "${RESULT_DIR}.tar.gz"
