#!/usr/bin/env bash
set -euo pipefail
cd /workspace/RLinf
export PYTHONPATH=/workspace/RLinf
role="${1:-}"
if [ "$#" -gt 0 ]; then shift; fi
case "$role" in
  host)
    export RLINF_NODE_RANK=0
    export RLINF_COMM_NET_DEVICES=enx6c1ff7bb8d22
    export GLOO_SOCKET_IFNAME=$RLINF_COMM_NET_DEVICES
    export NCCL_SOCKET_IFNAME=$RLINF_COMM_NET_DEVICES
    exec /opt/venv/openvla/bin/ray start --head --node-ip-address=192.168.10.11 --port=6380 --disable-usage-stats "$@"
    ;;
  nuc)
    set +u
    source /usr/local/bin/switch_env franka-0.15.0
    set -u
    export RLINF_NODE_RANK=1
    export RLINF_COMM_NET_DEVICES=enx6c1ff7bcd3e4
    export GLOO_SOCKET_IFNAME=$RLINF_COMM_NET_DEVICES
    export NCCL_SOCKET_IFNAME=$RLINF_COMM_NET_DEVICES
    exec /opt/venv/franka-0.15.0/bin/ray start --address=192.168.10.11:6380 --node-ip-address=192.168.10.10 --num-gpus=0 --disable-usage-stats "$@"
    ;;
  *) echo 'Usage: bash examples/embodiment/ruiyan/start_node.sh host|nuc' >&2; exit 2 ;;
esac
