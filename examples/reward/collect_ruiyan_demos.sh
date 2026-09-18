#!/usr/bin/env bash
set -euo pipefail
export REWARD_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$REWARD_PATH")")"
cd "$REPO_PATH"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"
export RLINF_NODE_RANK=0
DEMO_LOG_DIR="$REPO_PATH/logs/$(date +'%Y%m%d-%H%M%S')-ruiyan-demos"
mkdir -p "$DEMO_LOG_DIR"
python examples/reward/collect_ruiyan_demos.py runner.logger.log_path="$DEMO_LOG_DIR" "$@" 2>&1 | tee "$DEMO_LOG_DIR/run.log"
