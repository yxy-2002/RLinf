#!/usr/bin/env bash
set -euo pipefail
umask 0002
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$REPO_ROOT/examples/embodiment"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline
SESSION="lamp-dp-regularization-sweep"
OUTPUT="$REPO_ROOT/outputs/lamp_lstm_il/dp_regularization_sweep"
ARGS=("$@")
for arg in "$@"; do
    case "$arg" in
        --dry-run|--prepare)
            exec "$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_dp_regularization_sweep "$@"
            ;;
    esac
done
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "Already running: tmux $SESSION" >&2
    exit 1
fi
# Prepare and validate synchronously before allocating any GPU job.
"$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_dp_regularization_sweep --prepare "${ARGS[@]}"
"$REPO_ROOT/.venv/bin/ray" status >/dev/null
mkdir -p "$OUTPUT"
printf -v launch '%q ' env WANDB_MODE=offline OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" EMBODIED_PATH="$EMBODIED_PATH" "$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_dp_regularization_sweep "${ARGS[@]}"
printf -v log '%q' "$OUTPUT/launcher.log"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$launch >> $log 2>&1"
echo "Started: tmux $SESSION"
echo "Log: $OUTPUT/launcher.log"
