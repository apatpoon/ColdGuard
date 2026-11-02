#!/usr/bin/env bash
set -Eeuo pipefail

RESULT_PARENT="${RESULT_PARENT:-/workspace/coldguard-a100-stateful-results-v2}"
if [[ $# -ge 1 ]]; then
  RESULT_DIR="$1"
elif [[ -s "${RESULT_PARENT}/LAST_RESULT_DIR.txt" ]]; then
  RESULT_DIR="$(cat "${RESULT_PARENT}/LAST_RESULT_DIR.txt")"
else
  echo "No stateful result directory exists yet."
  exit 0
fi
echo "=== ColdGuard A100 four-GPU stateful competition v2 ==="
echo "Result: ${RESULT_DIR}"
date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'
python3 - "${RESULT_DIR}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
runs=[]
for path in (root/'policy_runs').glob('*/completion.json'):
    try:runs.append(json.loads(path.read_text()))
    except Exception:pass
print(f"Completed policy runs: {sum(row.get('status') == 'COMPLETED' for row in runs)} / 60")
print(f"Failed policy runs:    {sum(row.get('status') != 'COMPLETED' for row in runs)}")
pilot=root/'capacity_pilot'/'completion.json'
if pilot.exists():
    row=json.loads(pilot.read_text());print(f"Four-GPU capacity pilot passed: {row.get('passed')}")
state=root/'current_state.json'
if state.exists():
    row=json.loads(state.read_text());print(f"Current phase: {row.get('phase')}");print(f"Block/round:   {row.get('block')} / {row.get('parallel_round')}")
for gpu in range(4):
    path=root/f'current_state_gpu{gpu}.json'
    if path.exists():
        row=json.loads(path.read_text());print(f"GPU {gpu}: {row.get('phase')} | {row.get('run_id','-')} | {row.get('policy','-')}")
audit=root/'summary'/'evidence_audit.json'
if audit.exists():print(f"Evidence complete: {json.loads(audit.read_text()).get('complete')}")
PY
echo
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits || true
echo
tail -n 20 "${RESULT_DIR}/master.log" 2>/dev/null || true
