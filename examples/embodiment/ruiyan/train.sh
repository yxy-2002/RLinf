#!/usr/bin/env bash
set -euo pipefail
cd /workspace/RLinf
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=enx6c1ff7bb8d22
export GLOO_SOCKET_IFNAME=$RLINF_COMM_NET_DEVICES
export NCCL_SOCKET_IFNAME=$RLINF_COMM_NET_DEVICES
export RAY_ADDRESS=192.168.10.11:6380
/opt/venv/openvla/bin/ray status --address="$RAY_ADDRESS"
log_dir="logs/$(date +%Y%m%d-%H%M%S)-ruiyan-rlpd"
mkdir -p "$log_dir"
/opt/venv/openvla/bin/python examples/embodiment/ruiyan/preflight.py \
  "$log_dir" "$@" "runner.logger.log_path=$log_dir"
/opt/venv/openvla/bin/python examples/embodiment/train_async.py \
  --config-name realworld_ruiyan_rlpd_local \
  "$@" "runner.logger.log_path=$log_dir" 2>&1 | tee "$log_dir/train.log"
