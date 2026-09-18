#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export REWARD_PATH="$REPO_PATH/examples/reward"
cd "$REPO_PATH"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"
export RLINF_NODE_RANK=0
DEMO_LOG_DIR="$REPO_PATH/logs/$(date +'%Y%m%d-%H%M%S')-ruiyan-demos"
mkdir -p "$DEMO_LOG_DIR"
python examples/embodiment/franka_ruiyan/collect_demos.py runner.logger.log_path="$DEMO_LOG_DIR" "$@" 2>&1 | tee "$DEMO_LOG_DIR/run.log"
