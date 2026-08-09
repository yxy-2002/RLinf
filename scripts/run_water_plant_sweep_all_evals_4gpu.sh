#!/usr/bin/env bash
# Run all failed water_plant sweep evaluations from part2 and part3 on 4 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/lamp_water_plant_hparam}"
EVAL_ENVS="${EVAL_ENVS:-50}"
WANDB_MODE="${WANDB_MODE:-offline}"
N_GPUS=4

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

artifact_ready() {
  [[ -f "$1/model.safetensors" && -f "$1/artifact.json" && -f "$1/statistics.npz" ]]
}

run_eval() {
  local gpu="$1" part="$2" policy_name="$3"
  local part_root="${OUTPUT_BASE}/${part}"
  local artifact="${part_root}/${policy_name}/artifact"
  local eval_name="${policy_name}_eval50"
  local eval_dir="${part_root}/eval/${eval_name}"
  local done_marker="${eval_dir}/.complete"
  local log_file="${part_root}/logs/${eval_name}.log"

  if [[ -f "${done_marker}" ]]; then
    log "reuse completed eval: ${part}/${eval_name}"
    return 0
  fi
  if ! artifact_ready "${artifact}"; then
    log "FAILED missing policy artifact: ${artifact}"
    return 1
  fi

  mkdir -p "${eval_dir}" "${part_root}/logs"
  log "start eval gpu=${gpu} part=${part} policy=${policy_name}"
  if ! "${PYTHON_BIN}" evaluations/eval_embodied_agent.py \
    --config-path "${REPO_ROOT}/evaluations/dexjoco" \
    --config-name dexjoco_lamp_dp_50seed_water_plant_eval \
    "cluster.component_placement={env\, rollout:${gpu}-${gpu}}" \
    "env.eval.seed=20260803" \
    "env.eval.total_num_envs=${EVAL_ENVS}" \
    "rollout.model.model_path=${artifact}" \
    "runner.logger.log_path=${eval_dir}" \
    "runner.logger.experiment_name=${eval_name}" >"${log_file}" 2>&1; then
    log "FAILED eval: ${part}/${eval_name} (see ${log_file})"
    return 1
  fi

  touch "${done_marker}"
  log "finish eval: ${part}/${eval_name}"
}

# Entries are part:policy_name. They are assigned to GPUs 0..3 in batches.
EVALS=(
  "part2:dp_decoder_only_z2_loose_lr3e-5_bb0p1"
  "part2:dp_decoder_only_z2_loose_lr1e-4_bb0p1"
  "part2:dp_decoder_only_z2_loose_lr2e-4_bb0p1"
  "part2:dp_pca_z2_lr3e-5"
  "part2:dp_pca_z2_lr1e-4"
  "part2:dp_pca_z2_lr2e-4"
  "part2:dp_vq"
  "part3:dp_decoder_only_z2_default_lr2e-4_bb0p1"
  "part3:dp_cvae_z2_selected_lr3e-5"
  "part3:dp_cvae_z2_selected_lr1e-4"
)

log "begin combined water_plant evals: host=$(hostname) jobs=${#EVALS[@]} n_gpus=${N_GPUS}"

pids=()
descriptions=()
failed=0
for i in "${!EVALS[@]}"; do
  gpu=$((i % N_GPUS))
  part="${EVALS[$i]%%:*}"
  policy_name="${EVALS[$i]#*:}"
  run_eval "${gpu}" "${part}" "${policy_name}" &
  pids+=("$!")
  descriptions+=("${part}/${policy_name}")

  if (( ${#pids[@]} == N_GPUS || i + 1 == ${#EVALS[@]} )); then
    for j in "${!pids[@]}"; do
      if ! wait "${pids[$j]}"; then
        log "eval job failed: ${descriptions[$j]}"
        failed=1
      fi
    done
    pids=()
    descriptions=()
  fi
done

if (( failed )); then
  log "combined evals completed with failures"
  exit 1
fi
log "complete combined water_plant evals"
