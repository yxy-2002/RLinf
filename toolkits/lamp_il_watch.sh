#!/usr/bin/env bash
# Lightweight local queue for the DexJoCo LAMP IL experiment matrix.
# It intentionally only launches work in empty per-GPU slots and evaluates a
# completed policy before selecting its next training job.
set -u

ROOT="/vepfs-mlp2/c20250301/240906020/RLinf"
PYTHON_BIN="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs/lamp"
EVAL_ROOT="$ROOT/outputs/lamp_eval"
POLL_SECONDS=120

declare -A GPU=( [hammer_nail]=0 [pick_bucket]=1 [pinch_tongs]=2 [water_plant]=3 )
declare -A DESCRIPTION=(
  [hammer_nail]='Use the hammer to drive the nail into the wooden board.'
  [pick_bucket]='Place the boxed food into the bucket and then lift the bucket.'
  [pinch_tongs]='Grasp the tongs and perform three consecutive open-close motions.'
  [water_plant]='Grasp the watering can and apply water to the plant.'
)
TASKS=(hammer_nail pick_bucket pinch_tongs water_plant)
MODES=(vq pca mlp cvae decoder_only)

mkdir -p "$LOG_DIR" "$EVAL_ROOT"
cd "$ROOT"

has_final_checkpoint() {
  [[ -d "$ROOT/outputs/$1/checkpoints/global_step_30000" ]]
}

training_active() {
  pgrep -f "train_lamp_il.py --config-name $1" >/dev/null 2>&1
}

evaluation_active() {
  pgrep -f "runner.logger.experiment_name=$1_eval_50" >/dev/null 2>&1
}

gpu_slot_count() {
  local gpu="$1"
  ps -eo args | grep -E "cluster\.component_placement\.actor=${gpu}-${gpu}|config-name dexjoco_lamp_dp_eval_50_gpu${gpu}" | grep -v grep | wc -l
}

prior_ready() {
  local task="$1" mode="$2"
  case "$mode" in
    vq) has_final_checkpoint "dexjoco_lamp_prior_vq_$task" ;;
    # PCA is a closed-form one-step fit, so its completed artifact is saved at
    # global_step_1 rather than the 30k-step neural-prior convention.
    pca) [[ -d "$ROOT/outputs/dexjoco_lamp_prior_pca_dim6_$task/checkpoints/global_step_1" ]] ;;
    cvae|decoder_only) has_final_checkpoint "dexjoco_lamp_prior_cvae_dim6_$task" ;;
    mlp) return 0 ;;
  esac
}

launch_train() {
  local cfg="$1" gpu="$2"
  echo "$(date -Is) launch training: $cfg on GPU $gpu" >> "$LOG_DIR/lamp_il_watch.log"
  setsid env MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT" "$PYTHON_BIN" \
    examples/embodiment/train_lamp_il.py --config-name "$cfg" \
    "cluster.component_placement.actor=${gpu}-${gpu}" \
    >"$LOG_DIR/${cfg}.g${gpu}.log" 2>&1 < /dev/null &
}

launch_evaluation() {
  local cfg="$1" task="$2" gpu="$3" marker="$4"
  local output="$EVAL_ROOT/$cfg"
  local running_marker="${marker}.running"
  mkdir -p "$output"
  # Keep an explicit lock across the short interval between evaluator exit and
  # result collection, when the evaluator process itself is no longer visible.
  touch "$running_marker"
  echo "$(date -Is) launch evaluation: $cfg on GPU $gpu" >> "$LOG_DIR/lamp_il_watch.log"
  (
    # Hydra resolves a relative config path from evaluations/eval_embodied_agent.py.
    env MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT" \
      EMBODIED_PATH="$ROOT/examples/embodiment" "$PYTHON_BIN" \
      evaluations/eval_embodied_agent.py \
      --config-path "$ROOT/examples/embodiment/config" \
      --config-name "dexjoco_lamp_dp_eval_50_gpu${gpu}" \
      "env.eval.task_name=${task}" \
      "env.eval.task_description=${DESCRIPTION[$task]}" \
      "rollout.model.model_path=$ROOT/outputs/$cfg/artifact" \
      "runner.logger.log_path=$output" \
      "runner.logger.experiment_name=${cfg}_eval_50" \
      >"$LOG_DIR/${cfg}.eval.log" 2>&1
    status=$?
    if [[ $status -eq 0 ]]; then
      if "$PYTHON_BIN" toolkits/collect_lamp_eval_result.py \
        --cfg "$cfg" --task "$task" --eval-root "$EVAL_ROOT" \
        >>"$LOG_DIR/${cfg}.eval.log" 2>&1; then
        touch "$marker"
        rm -f "$running_marker"
        echo "$(date -Is) evaluation complete: $cfg" >> "$LOG_DIR/lamp_il_watch.log"
      else
        rm -f "$running_marker"
        echo "$(date -Is) evaluation result collection failed: $cfg" >> "$LOG_DIR/lamp_il_watch.log"
      fi
    else
      rm -f "$running_marker"
      echo "$(date -Is) evaluation failed ($status): $cfg" >> "$LOG_DIR/lamp_il_watch.log"
    fi
  ) &
}

while true; do
  # The user explicitly requested that a newly-created zk workload be removed.
  zk_pids="$(pgrep -f '(^|/)zk\.py( |$)' || true)"
  if [[ -n "$zk_pids" ]]; then
    kill $zk_pids || true
    sleep 5
    kill -KILL $zk_pids 2>/dev/null || true
    echo "$(date -Is) removed zk processes: $zk_pids" >> "$LOG_DIR/lamp_il_watch.log"
  fi

  for task in "${TASKS[@]}"; do
    gpu="${GPU[$task]}"

    # Evaluation is always preferred once a final DP checkpoint is available.
    for mode in "${MODES[@]}"; do
      cfg="dexjoco_lamp_dp_il_${mode}_${task}"
      marker="$EVAL_ROOT/$cfg/.eval_50_complete"
      if has_final_checkpoint "$cfg" && [[ ! -e "$marker" ]] && [[ ! -e "${marker}.running" ]] && ! evaluation_active "$cfg" && ! training_active "$cfg"; then
        if [[ "$(gpu_slot_count "$gpu")" -lt 2 ]]; then
          launch_evaluation "$cfg" "$task" "$gpu" "$marker"
          break
        fi
      fi
    done

    [[ "$(gpu_slot_count "$gpu")" -ge 2 ]] && continue

    for mode in "${MODES[@]}"; do
      # pick_bucket MLP is trained on the other development machine by request.
      [[ "$task" == pick_bucket && "$mode" == mlp ]] && continue
      cfg="dexjoco_lamp_dp_il_${mode}_${task}"
      if ! has_final_checkpoint "$cfg" && ! training_active "$cfg" && prior_ready "$task" "$mode"; then
        launch_train "$cfg" "$gpu"
        break
      fi
    done
  done
  sleep "$POLL_SECONDS"
done
