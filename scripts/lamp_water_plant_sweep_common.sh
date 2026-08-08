#!/usr/bin/env bash
# Shared scheduler for water_plant CVAE hyperparameter search.
# latent_dim is fixed at 2. Search focuses on CVAE KL and DP learning rate.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${PART:?PART must be part1, part2, or part3}"
: "${N_GPUS:?N_GPUS must be set by the part launcher}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/lamp_water_plant_hparam/${PART}}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/outputs/lamp_cache}"
EVAL_ENVS="${EVAL_ENVS:-50}"
WANDB_MODE="${WANDB_MODE:-offline}"
LATENT_DIM=2

# CVAE KL recipes (posterior fixed at 1e-4; latent_dim fixed at 2).
KL_DEFAULT_POST=1e-4
KL_DEFAULT_PRIOR=1e-3
KL_SELECTED_POST=1e-4
KL_SELECTED_PRIOR=3e-4
KL_LOOSE_POST=1e-4
KL_LOOSE_PRIOR=1e-4

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
    "env.eval.seed=20260803" \
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

run_batched() {
  local stage="$1"
  shift
  local -a pids=() names=()
  local gpu=0 command pid i failed=0
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

cvae_name() { printf 'prior_cvae_z%s_%s' "${LATENT_DIM}" "$1"; }
cvae_artifact() { printf '%s/%s/artifact' "${OUTPUT_ROOT}" "$(cvae_name "$1")"; }

cvae_overrides() {
  local recipe="$1"
  local post prior
  case "${recipe}" in
    default)
      post="${KL_DEFAULT_POST}"
      prior="${KL_DEFAULT_PRIOR}"
      ;;
    selected)
      post="${KL_SELECTED_POST}"
      prior="${KL_SELECTED_PRIOR}"
      ;;
    loose)
      post="${KL_LOOSE_POST}"
      prior="${KL_LOOSE_PRIOR}"
      ;;
    *)
      echo "Unknown CVAE KL recipe: ${recipe}" >&2
      return 2
      ;;
  esac
  printf 'actor.model.hand_prior.latent_dim=%s actor.model.hand_prior.posterior_kl_weight=%s actor.model.hand_prior.prior_kl_weight=%s' \
    "${LATENT_DIM}" "${post}" "${prior}"
}

declare -a PRIOR_COMMANDS=()
declare -a DP_COMMANDS=()
declare -a POLICY_NAMES=()

add_cvae_prior() {
  local gpu="$1" recipe="$2"
  PRIOR_COMMANDS+=(
    "run_train ${gpu} dexjoco_lamp_prior_cvae_dim6_water_plant $(cvae_name "${recipe}") $(cvae_overrides "${recipe}")"
  )
}

add_decoder_only_dp() {
  local gpu="$1" recipe="$2" lr="$3"
  local backbone_ratio=0.1
  local name
  name="$(printf 'dp_decoder_only_z%s_%s_lr%s_bb%s' "${LATENT_DIM}" "${recipe}" "${lr}" "${backbone_ratio}")"
  name="${name//./p}"
  DP_COMMANDS+=(
    "run_train ${gpu} dexjoco_lamp_dp_il_decoder_only_water_plant ${name} actor.model.hand_prior.latent_dim=${LATENT_DIM} actor.model.hand_prior.artifact_path=$(cvae_artifact "${recipe}") actor.optim.lr=${lr} actor.optim.backbone_lr_ratio=${backbone_ratio}"
  )
  POLICY_NAMES+=("${name}")
}

add_cvae_dp() {
  local gpu="$1" recipe="$2" lr="$3"
  local name
  name="$(printf 'dp_cvae_z%s_%s_lr%s' "${LATENT_DIM}" "${recipe}" "${lr}")"
  name="${name//./p}"
  DP_COMMANDS+=(
    "run_train ${gpu} dexjoco_lamp_dp_il_cvae_water_plant ${name} actor.model.hand_prior.latent_dim=${LATENT_DIM} actor.model.hand_prior.artifact_path=$(cvae_artifact "${recipe}") actor.optim.lr=${lr}"
  )
  POLICY_NAMES+=("${name}")
}

case "${PART}" in
  part1)
    # This 2-GPU machine: KL selected/default + DP lr sweep on selected + partial default.
    add_cvae_prior 0 selected
    add_cvae_prior 1 default
    DP_COMMANDS+=(
      "run_train 0 dexjoco_lamp_dp_il_mlp_water_plant dp_mlp actor.optim.lr=3e-5"
    )
    POLICY_NAMES+=(dp_mlp)
    add_decoder_only_dp 1 selected 3e-5
    add_decoder_only_dp 0 selected 1e-4
    add_decoder_only_dp 1 selected 2e-4
    add_decoder_only_dp 0 default 3e-5
    add_decoder_only_dp 1 default 1e-4
    ;;
  part2)
    # Remote 4-GPU: loose KL + PCA/VQ baselines.
    add_cvae_prior 0 loose
    PRIOR_COMMANDS+=(
      "run_train 1 dexjoco_lamp_prior_pca_dim6_water_plant prior_pca_z${LATENT_DIM} actor.model.hand_prior.latent_dim=${LATENT_DIM}"
    )
    PRIOR_COMMANDS+=(
      "run_train 2 dexjoco_lamp_prior_vq_water_plant prior_vq"
    )
    add_decoder_only_dp 0 loose 3e-5
    add_decoder_only_dp 1 loose 1e-4
    add_decoder_only_dp 2 loose 2e-4
    DP_COMMANDS+=(
      "run_train 3 dexjoco_lamp_dp_il_pca_water_plant dp_pca_z${LATENT_DIM} actor.model.hand_prior.latent_dim=${LATENT_DIM} actor.model.hand_prior.artifact_path=${OUTPUT_ROOT}/prior_pca_z${LATENT_DIM}/artifact actor.optim.lr=3e-5"
    )
    POLICY_NAMES+=("dp_pca_z${LATENT_DIM}")
    DP_COMMANDS+=(
      "run_train 0 dexjoco_lamp_dp_il_vq_water_plant dp_vq actor.model.hand_prior.artifact_path=${OUTPUT_ROOT}/prior_vq/artifact actor.optim.lr=3e-5"
    )
    POLICY_NAMES+=(dp_vq)
    ;;
  part3)
    # Remote 4-GPU: remaining default-LR point plus DP+CVAE points.
    add_cvae_prior 0 selected
    add_cvae_prior 1 default
    add_decoder_only_dp 0 default 2e-4
    add_cvae_dp 1 selected 3e-5
    add_cvae_dp 2 selected 1e-4
    ;;
  *)
    echo "Unknown PART=${PART}; expected part1, part2, or part3" >&2
    exit 2
    ;;
esac

log "begin ${PART}: host=$(hostname) n_gpus=${N_GPUS} output=${OUTPUT_ROOT}"
log "latent_dim fixed at ${LATENT_DIM}; searching CVAE KL and DP lr (no latent-dim grid)"
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
