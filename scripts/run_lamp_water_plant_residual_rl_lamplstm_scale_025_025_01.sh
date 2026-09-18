#!/usr/bin/env bash
# Launch the two-GPU LSTM FiLM residual SAC configuration from any directory.
# Usage: bash scripts/run_lamp_water_plant_residual_rl_lamplstm_scale_025_025_01.sh [Hydra overrides...]
# Optional: RLINF_PYTHON=/path/to/python, LAMP_RL_LOG_ROOT=/path/to/logs.
# Set LAMP_RL_DRY_RUN=1 to print the resolved config without starting training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
CONFIG_NAME="dexjoco_lamp_residual_sac_lamplstm_water_plant_scale_025_025_01"
PYTHON_BIN="${RLINF_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
LOG_ROOT="${LAMP_RL_LOG_ROOT:-${REPO_ROOT}/outputs/lamp_residual_v5_runs}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "RLinf Python is not executable: ${PYTHON_BIN}" >&2
  echo "Set RLINF_PYTHON=/path/to/python and retry." >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}/${CONFIG_NAME}"
LOG_DIR="$(mktemp -d "${LOG_ROOT}/${CONFIG_NAME}/$(date -u +'%Y%m%d-%H%M%S')-XXXXXX")"
COMMAND=(
  "${PYTHON_BIN}" -u
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
printf '%q ' "${COMMAND[@]}" >"${LOG_DIR}/launch_command.txt"
printf '\n' >>"${LOG_DIR}/launch_command.txt"
"${COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/run_embodiment.log"
