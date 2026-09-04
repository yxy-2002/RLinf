#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
CONFIG_NAME="dexjoco_lamp_residual_v4_online_cvae_water_plant_a07_seed42_scale030_full_obs_gpu01"
CONFIG_PATH="${REPO_ROOT}/examples/embodiment/config/${CONFIG_NAME}.yaml"
BASE_POLICY_RUN="${REPO_ROOT}/outputs/lamp_prior_modes_z2_selected_lr3e-5_water_plant_pick_bucket/water_plant_dp_cvae_cvae_z2_selected_lr3e-5"
ARTIFACT_PATH="${BASE_POLICY_RUN}/artifact"
PYTHON_BIN="${RLINF_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
LOG_ROOT="${LAMP_RL_LOG_ROOT:-${REPO_ROOT}/results/lamp_residual_v4_runs}"
RUN_ID="$(date -u +'%Y%m%d-%H%M%S')"
LOG_DIR="${LOG_ROOT}/${CONFIG_NAME}/${RUN_ID}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "RLinf Python is not executable: ${PYTHON_BIN}" >&2
  echo "Set RLINF_PYTHON=/path/to/python and retry." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing Hydra config: ${CONFIG_PATH}" >&2
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
    "policy_family": "dp",
    "embodiment": "single",
    "hand_prior_type": "cvae",
    "action_horizon": 16,
    "core_action_dim": 9,
    "physical_action_dim": 23,
    "latent_dims": {"single": 2},
}
actual = {
    "model_type": metadata.get("model_type"),
    "task": metadata.get("task"),
    **{key: spec.get(key) for key in expected if key not in {"model_type", "task"}},
}
if actual != expected:
    raise SystemExit(f"Unexpected LAMP CVAE artifact: expected {expected}, got {actual}")
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
