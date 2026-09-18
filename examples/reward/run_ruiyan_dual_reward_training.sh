#!/usr/bin/env bash
# Single-machine GPU reward training; keep NUC teleoperation networking separate.
set -euo pipefail
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_PATH"
# Override explicitly for another GPU host. Never inherit the NUC interface.
REWARD_TRAIN_IFACE="${REWARD_TRAIN_IFACE:-enp5s0}"
if ! ip -4 addr show dev "$REWARD_TRAIN_IFACE" | grep -q 'inet '; then
    echo "No IPv4 address on $REWARD_TRAIN_IFACE; set REWARD_TRAIN_IFACE to this GPU host's interface." >&2
    exit 1
fi
export RLINF_COMM_NET_DEVICES="$REWARD_TRAIN_IFACE"
export GLOO_SOCKET_IFNAME="$REWARD_TRAIN_IFACE"
export NCCL_SOCKET_IFNAME="$REWARD_TRAIN_IFACE"
export RLINF_NODE_RANK=0
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"
echo "Reward training interface: $REWARD_TRAIN_IFACE"
exec python examples/reward/train_reward_model.py \
    --config-name reward_training_ruiyan_dual "$@"
