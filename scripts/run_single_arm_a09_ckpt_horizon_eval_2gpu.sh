#!/usr/bin/env bash
# Remote-host partition: hammer_nail and fold_glasses on two GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export TASKS_TEXT="hammer_nail fold_glasses"
export GPU_IDS="${GPU_IDS:-0 1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/lamp_single_arm_default_a09_2gpu/eval_ckpt_horizon_split/remote_2gpu}"

exec "${SCRIPT_DIR}/run_single_arm_a09_ckpt_horizon_eval_4gpu.sh" "$@"
