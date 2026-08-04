#!/usr/bin/env bash
# Keep the two-task LAMP IL experiment moving without polling it interactively.
# This script only starts processes whose config names are listed below; it never
# kills processes (in particular, it deliberately leaves monitor_gpus.py alone).
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
source .venv/bin/activate

state_dir="${repo_dir}/outputs/lamp_supervisor"
mkdir -p "${state_dir}/logs" "${state_dir}/eval"
summary_csv="${state_dir}/evaluation_summary.csv"
if [[ ! -f "${summary_csv}" ]]; then
  printf 'policy,task,seed_start,seed_end,episodes,success_rate,eval_log\n' >"${summary_csv}"
fi

declare -A gpu_of=(
  [dexjoco_lamp_dp_il_pca_click_mouse]=0
  [dexjoco_lamp_dp_il_cvae_click_mouse]=0
  [dexjoco_lamp_dp_il_vq_click_mouse]=0
  [dexjoco_lamp_dp_il_pca_fold_glasses]=0
  [dexjoco_lamp_dp_il_mlp_fold_glasses]=0
  [dexjoco_lamp_dp_il_mlp_click_mouse]=1
  [dexjoco_lamp_dp_il_cvae_fold_glasses]=1
  [dexjoco_lamp_prior_vq_fold_glasses]=1
  [dexjoco_lamp_dp_il_decoder_only_click_mouse]=1
  [dexjoco_lamp_dp_il_decoder_only_fold_glasses]=1
  [dexjoco_lamp_dp_il_vq_fold_glasses]=1
)

# Only unfinished work belongs here. The order gives evaluation priority, then
# starts prerequisites early enough to keep both GPUs occupied.
queue=(
  dexjoco_lamp_prior_vq_fold_glasses
  dexjoco_lamp_dp_il_vq_click_mouse
  dexjoco_lamp_dp_il_pca_fold_glasses
  dexjoco_lamp_dp_il_mlp_fold_glasses
  dexjoco_lamp_dp_il_decoder_only_click_mouse
  dexjoco_lamp_dp_il_decoder_only_fold_glasses
  dexjoco_lamp_dp_il_vq_fold_glasses
)

final_ckpt() {
  local name="$1"
  [[ -f "outputs/${name}/checkpoints/global_step_30000/actor/training_state.pt" ]]
}

training_active() {
  local name="$1"
  ! final_ckpt "${name}" && pgrep -f "train_lamp_il.py --config-name ${name}" >/dev/null
}

eval_active() {
  local name="$1"
  [[ -f "${state_dir}/eval/${name}.started" && ! -f "${state_dir}/eval/${name}.done" && ! -f "${state_dir}/eval/${name}.failed" ]]
}

gpu_load() {
  local gpu="$1" count=0 name
  for name in "${!gpu_of[@]}"; do
    [[ "${gpu_of[$name]}" == "${gpu}" ]] || continue
    if training_active "${name}" || eval_active "${name}"; then
      ((count += 1))
    fi
  done
  printf '%s' "${count}"
}

start_eval() {
  local name="$1" task config log_file gpu
  [[ -f "${state_dir}/eval/${name}.started" ]] && return
  case "${name}" in
    *_click_mouse) task=click_mouse ;;
    *_fold_glasses) task=fold_glasses ;;
    *) return ;;
  esac
  config="dexjoco_lamp_dp_50seed_${task}_eval"
  gpu="${gpu_of[$name]}"
  log_file="${state_dir}/eval/${name}.log"
  : >"${state_dir}/eval/${name}.started"
  (
    set -o pipefail
    # Preserve the exit status so a failed evaluation records a retryable
    # marker instead of exiting this subshell early under `set -e`.
    set +e
    LAMP_EVAL_GPU="${gpu}-${gpu}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      bash evaluations/run_eval.sh dexjoco "${config}" \
      "runner.logger.experiment_name=${name}_50seed" \
      "rollout.model.model_path=./outputs/${name}/artifact" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ ${status} -eq 0 ]]; then
      success_rate="$(python - "${log_file}" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
matches = re.findall(r"eval/success['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text)
if not matches:
    matches = re.findall(r"success_once\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
print(matches[-1] if matches else "NA")
PY
)"
      printf '%s,%s,0,49,50,%s,%s\n' "${name}" "${task}" "${success_rate}" "${log_file}" >>"${summary_csv}"
      : >"${state_dir}/eval/${name}.done"
    else
      printf 'exit=%s\n' "${status}" >"${state_dir}/eval/${name}.failed"
    fi
  ) &
  printf '%s started eval %s\n' "$(date -u +'%F %T')" "${name}" >>"${state_dir}/supervisor.log"
}

can_start() {
  local name="$1"
  case "${name}" in
    dexjoco_lamp_dp_il_vq_fold_glasses)
      final_ckpt dexjoco_lamp_prior_vq_fold_glasses ;;
    dexjoco_lamp_dp_il_decoder_only_click_mouse)
      final_ckpt dexjoco_lamp_prior_cvae_dim6_click_mouse ;;
    dexjoco_lamp_dp_il_decoder_only_fold_glasses)
      final_ckpt dexjoco_lamp_prior_cvae_dim6_fold_glasses ;;
    *) true ;;
  esac
}

start_training() {
  local name="$1" gpu="${gpu_of[$1]}" log_file="${state_dir}/logs/${name}.log"
  (
    MUJOCO_GL=egl bash examples/embodiment/run_lamp_il.sh "${name}" \
      "cluster.component_placement.actor=${gpu}-${gpu}" 2>&1 | tee "${log_file}"
  ) &
  printf '%s started training %s on gpu%s\n' "$(date -u +'%F %T')" "${name}" "${gpu}" >>"${state_dir}/supervisor.log"
}

while true; do
  # A DP policy is always evaluated before this script starts another job in
  # the same slot.  The marker persists across restarts for reproducibility.
  for name in "${!gpu_of[@]}"; do
    case "${name}" in dexjoco_lamp_dp_il_*) final_ckpt "${name}" && start_eval "${name}" ;; esac
  done

  for name in "${queue[@]}"; do
    final_ckpt "${name}" && continue
    training_active "${name}" && continue
    can_start "${name}" || continue
    gpu="${gpu_of[$name]}"
    # Keep at most two active training/evaluation jobs per GPU.
    [[ "$(gpu_load "${gpu}")" -lt 2 ]] || continue
    start_training "${name}"
  done
  sleep 120
done
