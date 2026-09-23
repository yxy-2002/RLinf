#!/bin/bash
set -euo pipefail

export REWARD_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "${REWARD_PATH}")")"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
# Some inherited configs refer to EMBODIED_PATH.
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
CONFIG_NAME="reward_training"
if [[ $# -gt 0 && "$1" != *=* && "$1" != --* ]]; then
    CONFIG_NAME="$1"
    shift
fi
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
CMD=(python "${REWARD_PATH}/train_reward_model.py"
     --config-path "${REWARD_PATH}/config" --config-name "${CONFIG_NAME}"
     "runner.logger.log_path=${LOG_DIR}" "$@")
printf '%q ' "${CMD[@]}" > "${LOG_DIR}/run_reward_training.log"
printf '\n' >> "${LOG_DIR}/run_reward_training.log"
"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/run_reward_training.log"
