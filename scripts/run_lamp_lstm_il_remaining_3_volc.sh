#!/usr/bin/env bash
# Single-instance Volcengine custom-task entrypoint. Never detach into tmux.
set -euo pipefail
# Preserve shared-group write access for new logs, caches and checkpoints.
umask 0002
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$REPO_ROOT/examples/embodiment"
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl RLINF_NODE_RANK=0
if [[ "${1:-}" == --prepare && $# == 1 ]]; then
    exec "$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_lstm_il_remaining --group 3 --prepare
fi
if (( $# != 0 )); then
    echo 'Usage: bash run_lamp_lstm_il_remaining_3_volc.sh [--prepare]' >&2
    exit 2
fi
if [[ "${MLP_WORKER_NUM:-1}" != 1 || "${MLP_ROLE_INDEX:-0}" != 0 || "${WORLD_SIZE:-1}" != 1 ]]; then
    echo 'Use one instance and one entry process, without torchrun/DDP.' >&2
    exit 2
fi
unset RAY_ADDRESS MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE
export LAMP_CLOUD_GPUS="${LAMP_CLOUD_GPUS:-0,1,2,3}"
export LAMP_RAY_PORT="${LAMP_RAY_PORT:-6379}"
export LAMP_RAY_OBJECT_STORE_BYTES="${LAMP_RAY_OBJECT_STORE_BYTES:-8589934592}"
OUTPUT="$REPO_ROOT/outputs/lamp_lstm_il/remaining_three/group_3"
# Fail before starting Ray if mounted files, CUDA, or shared memory are missing.
"$REPO_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import shutil

import torch
import yaml

from scripts.lamp_lstm_il_remaining import PLAN, check_rows

rows = json.loads(PLAN.read_text())["groups"]["3"]
check_rows(rows)
for row in rows:
    directory = Path(row["directory"])
    for stage in ("dp",):
        cfg = yaml.safe_load((directory / f"{stage}.yaml").read_text())
        for raw in (cfg["data"]["dataset_root"], cfg["actor"]["model"]["resnet_path"]):
            if not Path(raw).exists():
                raise FileNotFoundError(f"Missing mount or symlink target: {raw}")
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {directory}")
gpus = [int(x) for x in os.environ["LAMP_CLOUD_GPUS"].split(",")]
if not gpus or min(gpus) < 0 or len(set(gpus)) != len(gpus):
    raise ValueError("LAMP_CLOUD_GPUS must contain distinct nonnegative indices")
if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
    raise RuntimeError(f"Requested GPUs {gpus}; visible CUDA GPUs: {torch.cuda.device_count()}")
shm = shutil.disk_usage("/dev/shm").free
store = int(os.environ["LAMP_RAY_OBJECT_STORE_BYTES"])
if store < 80 * 1024**2 or shm < store * 1.2:
    raise RuntimeError(f"Insufficient /dev/shm: free={shm}, object store={store}; use >=16 GiB shared memory for the default 8 GiB store")
print(f"Cloud preflight passed: GPUs={gpus}, /dev/shm free={shm}", flush=True)
PY
mkdir -p "$OUTPUT"
# Mirror the foreground driver's output to persistent storage and platform logs.
exec > >(tee -a "$OUTPUT/cloud_launcher.log") 2>&1
# This entry owns a fresh single-node Ray cluster; never stop an existing cluster.
if "$REPO_ROOT/.venv/bin/ray" status >/dev/null 2>&1; then
    echo 'A Ray cluster already exists. Use a fresh single-instance task container.' >&2
    exit 2
fi
RAY_NODE_IP="$("$REPO_ROOT/.venv/bin/python" -c 'from ray._private.services import get_node_ip_address; print(get_node_ip_address())')"
RAY_GPU_COUNT="$("$REPO_ROOT/.venv/bin/python" -c 'import torch; print(torch.cuda.device_count())')"
"$REPO_ROOT/.venv/bin/ray" start --head --node-ip-address="$RAY_NODE_IP" --port="$LAMP_RAY_PORT" --num-gpus="$RAY_GPU_COUNT" --object-store-memory="$LAMP_RAY_OBJECT_STORE_BYTES" --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS="$RAY_NODE_IP:$LAMP_RAY_PORT"
"$REPO_ROOT/.venv/bin/ray" status
exec "$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_lstm_il_remaining --group 3 --gpus "$LAMP_CLOUD_GPUS"
