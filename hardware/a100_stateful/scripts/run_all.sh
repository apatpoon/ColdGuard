#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${COLDGUARD_STATEFUL_CONFIG:-${PACKAGE_ROOT}/config/a100_stateful.json}"
RUNTIME_ROOT="${COLDGUARD_RUNTIME_ROOT:-/workspace/coldguard-runtimes/vllm-0.32.0-cu130-py312}"
RESULT_PARENT="${RESULT_PARENT:-/workspace/coldguard-a100-stateful-results-v2}"

if [[ ! -x "${RUNTIME_ROOT}/bin/python" ]]; then
  echo "ERROR: runtime missing: ${RUNTIME_ROOT}" >&2
  exit 20
fi
export PATH="${RUNTIME_ROOT}/bin:${PATH}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true
mkdir -p "${RESULT_PARENT}"
if [[ -n "${RESULT_DIR:-}" ]]; then
  mkdir -p "${RESULT_DIR}"
  RESULT_DIR="$(cd "${RESULT_DIR}" && pwd)"
else
  RESULT_DIR="${RESULT_PARENT}/$(date -u +%Y%m%dT%H%M%SZ)_stateful"
  mkdir -p "${RESULT_DIR}"
fi
printf '%s\n' "${RESULT_DIR}" > "${RESULT_PARENT}/LAST_RESULT_DIR.txt"
cp "${CONFIG}" "${RESULT_DIR}/requested_experiment_config.json"
cp -R "${PACKAGE_ROOT}/workload" "${RESULT_DIR}/requested_workload"
cp "${PACKAGE_ROOT}/PROTOCOL_CN.md" "${RESULT_DIR}/PROTOCOL_CN.md"
cp "${PACKAGE_ROOT}/PACKAGE_MANIFEST.sha256" "${RESULT_DIR}/SOURCE_PACKAGE_MANIFEST.sha256"
exec > >(tee -a "${RESULT_DIR}/master.log") 2>&1

echo "=== ColdGuard A100 four-GPU stateful multi-model competition v2 ==="
echo "Started UTC : $(date -u +%FT%TZ)"
echo "Package     : ${PACKAGE_ROOT}"
echo "Results     : ${RESULT_DIR}"
echo "Automatic Pod stop is disabled."

set +e
"${RUNTIME_ROOT}/bin/python" "${PACKAGE_ROOT}/scripts/preflight.py" --config "${CONFIG}" --package-root "${PACKAGE_ROOT}" --result-dir "${RESULT_DIR}"
preflight_rc=$?
set -e
if [[ ${preflight_rc} -ne 0 ]]; then
  echo "ERROR: preflight failed; no model service was started." >&2
  exit ${preflight_rc}
fi

set +e
"${RUNTIME_ROOT}/bin/python" "${PACKAGE_ROOT}/scripts/run_competition.py" --config "${CONFIG}" --package-root "${PACKAGE_ROOT}" --result-dir "${RESULT_DIR}"
replay_rc=$?
"${RUNTIME_ROOT}/bin/python" "${PACKAGE_ROOT}/scripts/summarize.py" --config "${CONFIG}" --result-dir "${RESULT_DIR}"
summary_rc=$?
set -e
printf 'replay_rc=%s summary_rc=%s\n' "${replay_rc}" "${summary_rc}" > "${RESULT_DIR}/stage_codes.txt"
bash "${PACKAGE_ROOT}/scripts/package_results.sh" "${RESULT_DIR}"
echo "Finished UTC: $(date -u +%FT%TZ)"
echo "Results: ${RESULT_DIR}"
echo "The Pod remains running for manual inspection."
if [[ ${replay_rc} -ne 0 || ${summary_rc} -ne 0 ]]; then
  exit 2
fi
