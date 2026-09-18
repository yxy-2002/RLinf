#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$REPO_ROOT/examples/embodiment"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline
GROUP=a
SESSION="lamp-lstm-il-$GROUP"
OUTPUT="$REPO_ROOT/outputs/lamp_lstm_il/group_$GROUP"
ARGS=()
FOREGROUND=0
PREPARE=0
for arg in "$@"; do
    case "$arg" in
        --foreground) FOREGROUND=1 ;;
        --prepare) PREPARE=1; ARGS+=("$arg") ;;
        --group|--group=*) echo "Group is fixed by this launcher" >&2; exit 2 ;;
        *) ARGS+=("$arg") ;;
    esac
done
if (( FOREGROUND || PREPARE )); then
    exec "$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_lstm_il --group "$GROUP" "${ARGS[@]}"
fi
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "Already running: tmux $SESSION"
    exit 1
fi
# Prepare synchronously: reject changed frozen configs before background dispatch.
"$REPO_ROOT/.venv/bin/python" -m scripts.lamp_lstm_il --group "$GROUP" --prepare
"$REPO_ROOT/.venv/bin/ray" status >/dev/null
mkdir -p "$OUTPUT"
printf -v launch '%q ' env WANDB_MODE=offline OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" EMBODIED_PATH="$EMBODIED_PATH" "$REPO_ROOT/.venv/bin/python" -u -m scripts.lamp_lstm_il --group "$GROUP" "${ARGS[@]}"
printf -v log '%q' "$OUTPUT/launcher.log"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$launch >> $log 2>&1"
echo "Started: tmux $SESSION; log: $OUTPUT/launcher.log"
