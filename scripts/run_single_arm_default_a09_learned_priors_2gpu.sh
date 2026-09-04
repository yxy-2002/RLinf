#!/usr/bin/env bash
# Six-task A09 transfer: CVAE, decoder-only, and AE on an independent 2-GPU host.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HOST_TAG="learned_priors"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/lamp_single_arm_default_a09_2gpu/${HOST_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/cache}"
RLINF_ARTIFACT_STAGING_DIR="${RLINF_ARTIFACT_STAGING_DIR:-/tmp}"
TRAIN_START_STAGGER_SECONDS="${TRAIN_START_STAGGER_SECONDS:-8}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
TRAIN_SEED=42
EVAL_SEED=0
POLICY_NOISE_SEED=0
EVAL_ENVS=50
N_GPUS=2
TRAIN_JOBS_PER_GPU=2
EVAL_JOBS_PER_GPU=1
GPU_IDS=(2 3)

TASKS=(click_mouse pinch_tongs hammer_nail fold_glasses water_plant pick_bucket)
MODES=(cvae decoder_only ae)

[[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 2; }
(( ${#GPU_IDS[@]} == N_GPUS )) || { echo "GPU_IDS must contain exactly ${N_GPUS} ranks" >&2; exit 2; }
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" ]] || { echo "GPU_IDS must be unique" >&2; exit 2; }
for task in "${TASKS[@]}"; do
  [[ -f "${DATASET_ROOT}/${task}/meta/info.json" ]] || {
    echo "Dataset is missing: ${DATASET_ROOT}/${task}" >&2; exit 2;
  }
done

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/eval" "${CACHE_ROOT}"
for path in "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/eval" "${CACHE_ROOT}"; do
  [[ -w "${path}" ]] || { echo "Not writable by uid=$(id -u): ${path}" >&2; exit 2; }
done
exec 9>"${OUTPUT_ROOT}/.launcher.lock"
flock -n 9 || { echo "Another ${HOST_TAG} launcher is already running: ${OUTPUT_ROOT}" >&2; exit 3; }

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE RLINF_ARTIFACT_STAGING_DIR
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
SESSION_LOG="${OUTPUT_ROOT}/logs/launcher_${SESSION_ID}.log"
log() {
  printf '[%s] [%s] [%s] %s\n' "$(date -Is)" "${HOST_TAG}" "${SESSION_ID}" "$*" |
    tee -a "${OUTPUT_ROOT}/logs/launcher.log" "${SESSION_LOG}"
}
CURRENT_PHASE=initialization
on_signal() {
  local signal="$1" status="$2" pid
  trap - HUP INT TERM
  log "received ${signal} during phase=${CURRENT_PHASE}; stopping children"
  while read -r pid; do [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true; done < <(jobs -pr)
  exit "${status}"
}
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

artifact_ready() {
  local root="$1" header
  [[ -s "${root}/model.safetensors" && -s "${root}/artifact.json" && -s "${root}/statistics.npz" ]] || return 1
  header="$(od -An -tu8 -N8 "${root}/model.safetensors" 2>/dev/null | tr -d '[:space:]')"
  [[ "${header}" =~ ^[0-9]+$ ]] && (( header > 1 ))
}

latest_checkpoint() {
  local run_dir="$1" checkpoint
  checkpoint="$(find "${run_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null |
    sort -V | while read -r candidate; do
      [[ -s "${candidate}/actor/training_state.pt" ]] && printf '%s\n' "${candidate}"
    done | tail -n 1)"
  [[ -n "${checkpoint}" ]] && printf '%s' "${checkpoint}"
}

check_contract_or_write() {
  local run_dir="$1" expected="$2" contract_file="${run_dir}/launch_contract.txt"
  [[ "${DRY_RUN}" == 1 ]] && return 0
  mkdir -p "${run_dir}"
  if [[ -f "${contract_file}" && "$(<"${contract_file}")" != "${expected}" ]]; then
    log "CONTRACT MISMATCH: ${run_dir}"
    log "expected: ${expected}"
    log "found: $(<"${contract_file}")"
    return 1
  fi
  printf '%s\n' "${expected}" >"${contract_file}"
}

run_prior() {
  local gpu="$1" task="$2" prior="$3" config name run_dir artifact contract checkpoint
  config="dexjoco_lamp_prior_${prior}_${task}"
  case "${prior}" in
    cvae_dim2_selected) name="${task}_prior_cvae_z2_selected" ;;
    ae_dim2) name="${task}_prior_ae_z2" ;;
    *) log "Unknown prior: ${prior}"; return 2 ;;
  esac
  run_dir="${OUTPUT_ROOT}/train/${name}"
  artifact="${run_dir}/artifact"
  contract="prior|${config}|task=${task}|seed=${TRAIN_SEED}|hand=0.0|data=${DATASET_ROOT}|cache=${CACHE_ROOT}"
  check_contract_or_write "${run_dir}" "${contract}" || return 1
  if artifact_ready "${artifact}"; then log "reuse prior ${name}"; return 0; fi
  if [[ "${DRY_RUN}" == 1 ]]; then log "would train prior gpu=${gpu} ${name}"; return 0; fi
  checkpoint="$(latest_checkpoint "${run_dir}" || true)"
  local -a resume=()
  [[ -z "${checkpoint}" ]] || resume+=("runner.resume_dir=${checkpoint}")
  log "train prior gpu=${gpu} ${name}${checkpoint:+ resume=${checkpoint}}"
  if ! "${PYTHON_BIN}" examples/embodiment/train_lamp_il.py \
      --config-name "${config}" \
      "cluster.component_placement.actor=${gpu}-${gpu}" \
      "data.dataset_root=${DATASET_ROOT}" "data.cache_root=${CACHE_ROOT}" \
      "algorithm.bc_loss.hand=0.0" "actor.seed=${TRAIN_SEED}" \
      "runner.logger.log_path=${OUTPUT_ROOT}/train" "runner.logger.experiment_name=${name}" \
      "${resume[@]}" >"${OUTPUT_ROOT}/logs/${name}.log" 2>&1; then
    log "FAILED prior ${name}"; return 1
  fi
  artifact_ready "${artifact}" || { log "FAILED prior artifact validation ${name}"; return 1; }
  log "finish prior ${name}"
}

prior_artifact() {
  local task="$1" mode="$2"
  case "${mode}" in
    cvae|decoder_only) printf '%s' "${OUTPUT_ROOT}/train/${task}_prior_cvae_z2_selected/artifact" ;;
    ae) printf '%s' "${OUTPUT_ROOT}/train/${task}_prior_ae_z2/artifact" ;;
  esac
}

run_dp() {
  local gpu="$1" task="$2" mode="$3" config name run_dir artifact prior contract checkpoint prior_sha
  config="dexjoco_lamp_dp_il_${mode}_${task}_a09"
  name="${task}_dp_a09_${mode}_seed${TRAIN_SEED}"
  run_dir="${OUTPUT_ROOT}/train/${name}"
  artifact="${run_dir}/artifact"
  prior="$(prior_artifact "${task}" "${mode}")"
  if [[ "${DRY_RUN}" == 1 ]]; then
    prior_sha=dry_run
  else
    artifact_ready "${prior}" || { log "missing prior for ${name}: ${prior}"; return 1; }
    prior_sha="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' "${prior}/artifact.json")"
  fi
  contract="dp|${config}|task=${task}|mode=${mode}|prior=${prior}|prior_sha=${prior_sha}|z=2|lr=1e-4|bb=0.3|wd=1e-4|warm=500|hand=0.0|seed=${TRAIN_SEED}|data=${DATASET_ROOT}|cache=${CACHE_ROOT}"
  check_contract_or_write "${run_dir}" "${contract}" || return 1
  if artifact_ready "${artifact}"; then log "reuse DP ${name}"; return 0; fi
  if [[ "${DRY_RUN}" == 1 ]]; then log "would train DP gpu=${gpu} ${name}"; return 0; fi
  checkpoint="$(latest_checkpoint "${run_dir}" || true)"
  local -a resume=()
  [[ -z "${checkpoint}" ]] || resume+=("runner.resume_dir=${checkpoint}")
  log "train DP gpu=${gpu} ${name}${checkpoint:+ resume=${checkpoint}}"
  if ! "${PYTHON_BIN}" examples/embodiment/train_lamp_il.py \
      --config-name "${config}" \
      "cluster.component_placement.actor=${gpu}-${gpu}" \
      "data.dataset_root=${DATASET_ROOT}" "data.cache_root=${CACHE_ROOT}" \
      "algorithm.bc_loss.hand=0.0" "actor.seed=${TRAIN_SEED}" \
      "actor.model.hand_prior.latent_dim=2" "actor.model.hand_prior.artifact_path=${prior}" \
      "runner.logger.log_path=${OUTPUT_ROOT}/train" "runner.logger.experiment_name=${name}" \
      "${resume[@]}" >"${OUTPUT_ROOT}/logs/${name}.log" 2>&1; then
    log "FAILED DP ${name}"; return 1
  fi
  artifact_ready "${artifact}" || { log "FAILED DP artifact validation ${name}"; return 1; }
  log "finish DP ${name}"
}

run_eval() {
  local gpu="$1" task="$2" mode="$3" name artifact eid edir log_file model_sha contract
  name="${task}_dp_a09_${mode}_seed${TRAIN_SEED}"
  artifact="${OUTPUT_ROOT}/train/${name}/artifact"
  eid="${name}_k8_ddim16_env0-49"
  edir="${OUTPUT_ROOT}/eval/${eid}"
  log_file="${OUTPUT_ROOT}/logs/${eid}.log"
  if [[ "${DRY_RUN}" == 1 ]]; then
    model_sha=dry_run
  else
    artifact_ready "${artifact}" || { log "missing policy for eval ${name}"; return 1; }
    model_sha="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' "${artifact}/artifact.json")"
  fi
  contract="${model_sha}|${artifact}|k=8|ddim=16|env=0-49|noise=0-49|no_temporal"
  if [[ -f "${edir}/.complete" && -f "${edir}/contract.txt" && "$(<"${edir}/contract.txt")" == "${contract}" ]]; then
    log "reuse eval ${eid}"; return 0
  fi
  if [[ "${DRY_RUN}" == 1 ]]; then log "would eval gpu=${gpu} ${eid}"; return 0; fi
  mkdir -p "${edir}"
  rm -f "${edir}/.complete"
  printf '%s\n' "${contract}" >"${edir}/contract.txt"
  log "eval gpu=${gpu} ${eid}"
  if ! "${PYTHON_BIN}" evaluations/eval_embodied_agent.py \
      --config-path "${REPO_ROOT}/evaluations/dexjoco" \
      --config-name "dexjoco_lamp_dp_50seed_${task}_eval" \
      "cluster.component_placement={env\, rollout:${gpu}-${gpu}}" \
      "env.eval.seed=${EVAL_SEED}" "env.eval.total_num_envs=${EVAL_ENVS}" \
      "rollout.model.model_path=${artifact}" \
      "rollout.model.use_temporal_ensemble=false" \
      "rollout.model.execution_horizon_override=8" \
      "rollout.model.num_inference_steps_override=16" \
      "rollout.model.eval_base_noise_seed=${POLICY_NOISE_SEED}" \
      "runner.logger.log_path=${edir}" "runner.logger.experiment_name=${eid}" \
      >"${log_file}" 2>&1; then
    log "FAILED eval ${eid}"; return 1
  fi
  touch "${edir}/.complete"
  log "finish eval ${eid}"
}

run_batched() {
  local phase="$1" jobs_per_gpu="$2" function="$3"; shift 3
  local capacity=$((N_GPUS * jobs_per_gpu)) slot=0 failed=0 gpu record i
  local -a pids=() labels=()
  CURRENT_PHASE="${phase}"
  log "phase start ${phase}: jobs=$# capacity=${capacity}"
  for record in "$@"; do
    gpu="${GPU_IDS[$(((slot / jobs_per_gpu) % N_GPUS))]}"
    IFS='|' read -r -a fields <<<"${record}"
    ( sleep $((slot * TRAIN_START_STAGGER_SECONDS)); "${function}" "${gpu}" "${fields[@]}" ) &
    pids+=("$!"); labels+=("${record}"); slot=$((slot + 1))
    if (( slot == capacity )); then
      for i in "${!pids[@]}"; do wait "${pids[$i]}" || { log "FAILED ${phase} ${labels[$i]}"; failed=1; }; done
      pids=(); labels=(); slot=0
    fi
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { log "FAILED ${phase} ${labels[$i]}"; failed=1; }; done
  (( failed == 0 )) || return 1
  log "phase complete ${phase}"
}

preflight_compose() {
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
tasks = ['click_mouse', 'pinch_tongs', 'hammer_nail', 'fold_glasses', 'water_plant', 'pick_bucket']
with initialize_config_dir(version_base=None, config_dir=str(Path('examples/embodiment/config').resolve())):
    for task in tasks:
        compose(config_name=f'dexjoco_lamp_prior_cvae_dim2_selected_{task}', overrides=['algorithm.bc_loss.hand=0.0', 'actor.seed=42'])
        compose(config_name=f'dexjoco_lamp_prior_ae_dim2_{task}', overrides=['algorithm.bc_loss.hand=0.0', 'actor.seed=42'])
        for mode in ('cvae', 'decoder_only', 'ae'):
            cfg = compose(config_name=f'dexjoco_lamp_dp_il_{mode}_{task}_a09', overrides=['algorithm.bc_loss.hand=0.0', 'actor.seed=42', 'actor.model.hand_prior.latent_dim=2'])
            assert float(cfg.algorithm.bc_loss.hand) == 0.0
            assert (float(cfg.actor.optim.lr), float(cfg.actor.optim.backbone_lr_ratio), float(cfg.actor.optim.weight_decay), int(cfg.actor.optim.warmup_steps)) == (1e-4, 0.3, 1e-4, 500)
with initialize_config_dir(version_base=None, config_dir=str(Path('evaluations/dexjoco').resolve())):
    for task in tasks:
        cfg = compose(config_name=f'dexjoco_lamp_dp_50seed_{task}_eval')
        assert cfg.rollout.model.use_temporal_ensemble is False
        assert int(cfg.rollout.model.execution_horizon_override) == 8
PY
}

log "launcher started gpus=${GPU_IDS[*]} output=${OUTPUT_ROOT} cache=${CACHE_ROOT}"
preflight_compose
log "preflight passed: tasks=6 priors=12 policies=18 evals=18"
[[ "${PREFLIGHT_ONLY}" == 1 ]] && exit 0

PRIOR_JOBS=(); DP_JOBS=(); EVAL_JOBS=()
for task in "${TASKS[@]}"; do
  PRIOR_JOBS+=("${task}|cvae_dim2_selected" "${task}|ae_dim2")
  for mode in "${MODES[@]}"; do
    DP_JOBS+=("${task}|${mode}")
    EVAL_JOBS+=("${task}|${mode}")
  done
done

run_batched prior_training "${TRAIN_JOBS_PER_GPU}" run_prior "${PRIOR_JOBS[@]}"
run_batched dp_training "${TRAIN_JOBS_PER_GPU}" run_dp "${DP_JOBS[@]}"
TRAIN_START_STAGGER_SECONDS=0 run_batched evaluation "${EVAL_JOBS_PER_GPU}" run_eval "${EVAL_JOBS[@]}"

CURRENT_PHASE=summary
SUMMARY="${OUTPUT_ROOT}/evaluation_summary.tsv"
printf 'task\tmode\tk\tenv_seeds\tsuccess_once\tmetrics\n' >"${SUMMARY}"
for task in "${TASKS[@]}"; do for mode in "${MODES[@]}"; do
  eid="${task}_dp_a09_${mode}_seed${TRAIN_SEED}_k8_ddim16_env0-49"
  metrics="${OUTPUT_ROOT}/eval/${eid}/metrics.log"
  [[ -f "${metrics}" ]] || continue
  success="$("${PYTHON_BIN}" -c 'import re,sys; s=open(sys.argv[1],errors="ignore").read(); s=re.sub(r"\x1b\[[0-9;]*[A-Za-z]","",s); m=re.findall(r"success_once=([0-9.]+)",s); print(m[-1] if m else "NA")' "${metrics}")"
  printf '%s\t%s\t8\t0-49\t%s\t%s\n' "${task}" "${mode}" "${success}" "${metrics}" >>"${SUMMARY}"
done; done
log "complete summary=${SUMMARY}"
