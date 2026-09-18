#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export REWARD_PATH="$REPO_PATH/examples/reward"
export SRC_FILE="${SCRIPT_DIR}/collect_reward_data.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME="realworld_collect_ruiyan_dataset"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_collect_process.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${REWARD_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
