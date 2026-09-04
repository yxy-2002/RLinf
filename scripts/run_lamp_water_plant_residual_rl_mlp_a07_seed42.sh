#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
CONFIG_NAME="dexjoco_lamp_residual_v4_online_mlp_water_plant_a07_seed42"
ARTIFACT_PATH="${REPO_ROOT}/outputs/lamp_water_plant_base_ckpts/dp_a07_mlp_seed42/artifact"
PYTHON_BIN="${RLINF_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
LOG_ROOT="${LAMP_RL_LOG_ROOT:-${REPO_ROOT}/results/lamp_residual_v4_runs}"
RUN_ID="$(date -u +'%Y%m%d-%H%M%S')"
LOG_DIR="${LOG_ROOT}/${CONFIG_NAME}/${RUN_ID}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "RLinf Python is not executable: ${PYTHON_BIN}" >&2
  echo "Set RLINF_PYTHON=/path/to/python and retry." >&2
  exit 1
fi

for required_file in artifact.json model.safetensors statistics.npz; do
  if [[ ! -f "${ARTIFACT_PATH}/${required_file}" ]]; then
    echo "Incomplete LAMP artifact: missing ${ARTIFACT_PATH}/${required_file}" >&2
    exit 1
  fi
done

export ARTIFACT_PATH
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

metadata = json.loads((Path(os.environ["ARTIFACT_PATH"]) / "artifact.json").read_text())
spec = metadata.get("spec", {})
expected = {
    "model_type": "lamp_dp",
    "task": "water_plant",
}
actual = {
    "model_type": metadata.get("model_type"),
    "task": metadata.get("task"),
}
if actual != expected:
    raise SystemExit(f"Unexpected LAMP artifact identity: expected {expected}, got {actual}")
if spec.get("hand_prior_type") != "mlp" or spec.get("action_horizon") != 16:
    raise SystemExit(f"Expected raw-MLP H=16 artifact, got spec={spec}")
if spec.get("physical_action_dim") != 23 or spec.get("core_action_dim") != 23:
    raise SystemExit(f"Expected 23D physical/raw core artifact, got spec={spec}")
PY

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_DIR}"
COMMAND=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/examples/embodiment/train_async.py"
  --config-path "${REPO_ROOT}/examples/embodiment/config"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "$@"
)
if [[ "${LAMP_RL_DRY_RUN:-0}" == "1" ]]; then
  COMMAND+=(--cfg job --resolve)
fi

printf 'Launching:'
printf ' %q' "${COMMAND[@]}"
printf '\nLogs: %s\n' "${LOG_DIR}"
printf 'Command:' >"${LOG_DIR}/launch_command.txt"
printf ' %q' "${COMMAND[@]}" >>"${LOG_DIR}/launch_command.txt"
printf '\n' >>"${LOG_DIR}/launch_command.txt"

cd "${REPO_ROOT}"
"${COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/run_embodiment.log"
