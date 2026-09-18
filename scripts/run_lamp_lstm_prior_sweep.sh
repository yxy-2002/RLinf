#!/usr/bin/env bash
set -euo pipefail
# Preserve shared output writability and keep each training process CPU-light.
umask 0002

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$REPO_ROOT/examples/embodiment"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline

SESSION="lamp-lstm-prior-sweep"
OUTPUT="$REPO_ROOT/outputs/lamp_lstm_prior_sweep"
ARGS=()
GPUS="0,1,2,3"
PER_GPU=4
DRY_RUN=0
PREPARE=0
SUMMARIZE=0
FOREGROUND=0

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --prepare) PREPARE=1 ;;
        --summarize) SUMMARIZE=1 ;;
        --foreground) FOREGROUND=1 ;;
        --gpus)
            shift
            GPUS="${1:?missing GPU list}"
            ;;
        --gpus=*) GPUS="${1#*=}" ;;
        --per-gpu)
            shift
            PER_GPU="${1:?missing parallel jobs per GPU}"
            ;;
        --per-gpu=*) PER_GPU="${1#*=}" ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

ARGS+=(--gpus "$GPUS" --per-gpu "$PER_GPU")
if (( DRY_RUN )); then ARGS+=(--dry-run); fi
if (( PREPARE )); then ARGS+=(--prepare); fi
if (( SUMMARIZE )); then ARGS+=(--summarize); fi

if (( DRY_RUN || PREPARE || SUMMARIZE || FOREGROUND )); then
    exec "$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_lstm_prior_sweep "${ARGS[@]}"
fi

if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "Already running: tmux $SESSION" >&2
    exit 1
fi
mkdir -p "$OUTPUT"
printf -v launch '%q ' env \
    PYTHONPATH="$PYTHONPATH" \
    EMBODIED_PATH="$EMBODIED_PATH" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    WANDB_MODE=offline \
    "$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_lstm_prior_sweep "${ARGS[@]}"
printf -v log '%q' "$OUTPUT/launcher.log"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$launch >> $log 2>&1"
echo "Started: tmux $SESSION; log: $OUTPUT/launcher.log"
