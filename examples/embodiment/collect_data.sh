#!/bin/bash
set -euo pipefail

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
# Some inherited configs refer to EMBODIED_PATH.
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
CONFIG_NAME="realworld_collect_data"
if [[ $# -gt 0 && "$1" != *=* && "$1" != --* ]]; then
    CONFIG_NAME="$1"
    shift
fi
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
CMD=(python "${EMBODIED_PATH}/collect_real_data.py"
     --config-path "${EMBODIED_PATH}/config" --config-name "${CONFIG_NAME}"
     "runner.logger.log_path=${LOG_DIR}" "$@")
printf '%q ' "${CMD[@]}" > "${LOG_DIR}/run_embodiment.log"
printf '\n' >> "${LOG_DIR}/run_embodiment.log"
"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/run_embodiment.log"
