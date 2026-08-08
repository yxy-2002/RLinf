#!/usr/bin/env bash
# Shared scheduler for the three water_plant LAMP hyperparameter sweep parts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${PART:?PART must be part1, part2, or part3}"
: "${N_GPUS:?N_GPUS must be set by the part launcher}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/lamp_water_plant_sweep/${PART}}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/outputs/lamp_cache}"
EVAL_ENVS="${EVAL_ENVS:-50}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the RLinf environment interpreter." >&2
  exit 2
fi
if [[ ! -f "${DATASET_ROOT}/water_plant/meta/info.json" ]]; then
  echo "water_plant dataset not found under: ${DATASET_ROOT}" >&2
  exit 2
fi
if (( N_GPUS < 1 )); then
  echo "N_GPUS must be >= 1" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/eval"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUTPUT_ROOT}/logs/sweep.log"
}

artifact_ready() {
  [[ -f "$1/model.safetensors" && -f "$1/artifact.json" && -f "$1/statistics.npz" ]]
}

run_train() {
  local gpu="$1" config="$2" name="$3"
  shift 3
  local log_file="${OUTPUT_ROOT}/logs/${name}.log"
  local artifact="${OUTPUT_ROOT}/${name}/artifact"
  if artifact_ready "${artifact}"; then
    log "reuse completed training artifact: ${name}"
    return 0
  fi
  log "start train gpu=${gpu} config=${config} name=${name}"
  "${PYTHON_BIN}" examples/embodiment/train_lamp_il.py \
    --config-name "${config}" \
    "cluster.component_placement.actor=${gpu}-${gpu}" \
    "data.dataset_root=${DATASET_ROOT}" \
    "data.cache_root=${CACHE_ROOT}" \
    "runner.logger.log_path=${OUTPUT_ROOT}" \
    "runner.logger.experiment_name=${name}" \
    "$@" >"${log_file}" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    log "FAILED train rc=${rc}: ${name} (see ${log_file})"
    return "${rc}"
  fi
  if ! artifact_ready "${artifact}"; then
    log "FAILED train produced no final artifact: ${name}"
    return 1
  fi
  log "finish train: ${name}"
}

run_eval() {
  local gpu="$1" policy_name="$2"
  local artifact="${OUTPUT_ROOT}/${policy_name}/artifact"
  local eval_name="${policy_name}_eval50"
  local eval_dir="${OUTPUT_ROOT}/eval/${eval_name}"
  local done_marker="${eval_dir}/.complete"
  local log_file="${OUTPUT_ROOT}/logs/${eval_name}.log"
  if [[ -f "${done_marker}" ]]; then
    log "reuse completed eval: ${eval_name}"
    return 0
  fi
  if ! artifact_ready "${artifact}"; then
    log "FAILED eval missing policy artifact: ${artifact}"
    return 1
  fi
  mkdir -p "${eval_dir}"
  log "start eval gpu=${gpu} policy=${policy_name}"
  "${PYTHON_BIN}" evaluations/eval_embodied_agent.py \
    --config-path "${REPO_ROOT}/evaluations/dexjoco" \
    --config-name dexjoco_lamp_dp_50seed_water_plant_eval \
    "cluster.component_placement.env,rollout=${gpu}-${gpu}" \
    "env.eval.total_num_envs=${EVAL_ENVS}" \
    "rollout.model.model_path=${artifact}" \
    "runner.logger.log_path=${eval_dir}" \
    "runner.logger.experiment_name=${eval_name}" >"${log_file}" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    log "FAILED eval rc=${rc}: ${eval_name} (see ${log_file})"
    return "${rc}"
  fi
  touch "${done_marker}"
  log "finish eval: ${eval_name}"
}

# Run shell function invocations in batches of N_GPUS. Each argument is a
# shell-escaped command beginning with run_train or run_eval.
run_batched() {
  local stage="$1"
  shift
  local -a pids=() names=()
  local gpu=0 command pid i rc failed=0
  for command in "$@"; do
    (
      eval "${command}"
    ) &
    pid=$!
    pids+=("${pid}")
    names+=("${command}")
    gpu=$((gpu + 1))
    if (( gpu == N_GPUS )); then
      for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
          log "${stage} job failed: ${names[$i]}"
          failed=1
        fi
      done
      pids=()
      names=()
      gpu=0
    fi
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      log "${stage} job failed: ${names[$i]}"
      failed=1
    fi
  done
  if (( failed )); then
    return 1
  fi
}

cvae_prior_name() { printf 'prior_cvae_z%s' "$1"; }
pca_prior_name() { printf 'prior_pca_z%s' "$1"; }
cvae_artifact() { printf '%s/%s/artifact' "${OUTPUT_ROOT}" "$(cvae_prior_name "$1")"; }
pca_artifact() { printf '%s/%s/artifact' "${OUTPUT_ROOT}" "$(pca_prior_name "$1")"; }

declare -a PRIOR_COMMANDS=()
declare -a DP_COMMANDS=()
declare -a POLICY_NAMES=()

case "${PART}" in
  part1)
    PRIOR_COMMANDS+=(
      "run_train 0 dexjoco_lamp_prior_cvae_dim6_water_plant prior_cvae_z2 actor.model.hand_prior.latent_dim=2"
    )
    DP_COMMANDS+=(
      "run_train 0 dexjoco_lamp_dp_il_mlp_water_plant dp_mlp"
      "run_train 1 dexjoco_lamp_dp_il_cvae_water_plant dp_cvae_z2 actor.model.hand_prior.latent_dim=2 actor.model.hand_prior.artifact_path=$(cvae_artifact 2)"
      "run_train 0 dexjoco_lamp_dp_il_decoder_only_water_plant dp_decoder_only_z2 actor.model.hand_prior.latent_dim=2 actor.model.hand_prior.artifact_path=$(cvae_artifact 2)"
    )
    POLICY_NAMES=(dp_mlp dp_cvae_z2 dp_decoder_only_z2)
    ;;
  part2)
    for dim in 2 4 6 8; do
      gpu=$(( (${#PRIOR_COMMANDS[@]}) % N_GPUS ))
      PRIOR_COMMANDS+=(
        "run_train ${gpu} dexjoco_lamp_prior_pca_dim6_water_plant prior_pca_z${dim} actor.model.hand_prior.latent_dim=${dim}"
      )
    done
    PRIOR_COMMANDS+=(
      "run_train 0 dexjoco_lamp_prior_vq_water_plant prior_vq"
    )
    for dim in 2 4 6 8; do
      gpu=$(( (${#DP_COMMANDS[@]}) % N_GPUS ))
      DP_COMMANDS+=(
        "run_train ${gpu} dexjoco_lamp_dp_il_pca_water_plant dp_pca_z${dim} actor.model.hand_prior.latent_dim=${dim} actor.model.hand_prior.artifact_path=$(pca_artifact "${dim}")"
      )
      POLICY_NAMES+=("dp_pca_z${dim}")
    done
    DP_COMMANDS+=(
      "run_train 0 dexjoco_lamp_dp_il_vq_water_plant dp_vq actor.model.hand_prior.artifact_path=${OUTPUT_ROOT}/prior_vq/artifact"
    )
    POLICY_NAMES+=(dp_vq)
    ;;
  part3)
    for dim in 4 6 8; do
      gpu=$(( (${#PRIOR_COMMANDS[@]}) % N_GPUS ))
      PRIOR_COMMANDS+=(
        "run_train ${gpu} dexjoco_lamp_prior_cvae_dim6_water_plant prior_cvae_z${dim} actor.model.hand_prior.latent_dim=${dim}"
      )
    done
    for dim in 4 6 8; do
      gpu=$(( (${#DP_COMMANDS[@]}) % N_GPUS ))
      DP_COMMANDS+=(
        "run_train ${gpu} dexjoco_lamp_dp_il_cvae_water_plant dp_cvae_z${dim} actor.model.hand_prior.latent_dim=${dim} actor.model.hand_prior.artifact_path=$(cvae_artifact "${dim}")"
      )
      POLICY_NAMES+=("dp_cvae_z${dim}")
      gpu=$(( (${#DP_COMMANDS[@]}) % N_GPUS ))
      DP_COMMANDS+=(
        "run_train ${gpu} dexjoco_lamp_dp_il_decoder_only_water_plant dp_decoder_only_z${dim} actor.model.hand_prior.latent_dim=${dim} actor.model.hand_prior.artifact_path=$(cvae_artifact "${dim}")"
      )
      POLICY_NAMES+=("dp_decoder_only_z${dim}")
    done
    ;;
  *)
    echo "Unknown PART=${PART}; expected part1, part2, or part3" >&2
    exit 2
    ;;
esac

log "begin ${PART}: host=$(hostname) n_gpus=${N_GPUS} output=${OUTPUT_ROOT}"
log "training priors locally; no checkpoint roots are shared across machines"
run_batched prior "${PRIOR_COMMANDS[@]}"
run_batched dp "${DP_COMMANDS[@]}"

declare -a EVAL_COMMANDS=()
for i in "${!POLICY_NAMES[@]}"; do
  gpu=$((i % N_GPUS))
  EVAL_COMMANDS+=("run_eval ${gpu} ${POLICY_NAMES[$i]}")
done
run_batched eval "${EVAL_COMMANDS[@]}"
log "complete ${PART}"
