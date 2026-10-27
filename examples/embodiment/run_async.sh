#! /bin/bash

set -o pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_async.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

CONFIG_NAME=${1:-maniskill_sac_mlp_async}
if [ "$#" -gt 0 ]; then
    shift
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
# Keep the optional positional platform; Hydra overrides and flags follow it.
if [ "$#" -gt 0 ] && [[ "$1" != *=* && "$1" != -* ]]; then
    ROBOT_PLATFORM=$1
    shift
fi

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}" "runner.logger.log_path=${LOG_DIR}" "$@")
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
# Keep the log pipe open while Python handles Ctrl+C and closes remote workers.
"${CMD[@]}" 2>&1 | (trap '' INT TERM; exec tee -a "${MEGA_LOG_FILE}")
