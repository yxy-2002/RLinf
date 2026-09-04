#!/usr/bin/env bash
# Current-host partition of the A09 checkpoint and horizon evaluation matrix.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/outputs/lamp_single_arm_default_a09_2gpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/eval_ckpt_horizon_split/current_4gpu}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2 3}"
TASKS_TEXT="${TASKS_TEXT:-click_mouse pinch_tongs water_plant pick_bucket}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-5}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"

TRAIN_SEED=42
EVAL_SEED_START=0
EVAL_SEED_END=49
POLICY_NOISE_SEED=0
EVAL_ENVS=50
DDIM_STEPS=16

read -r -a TASKS <<<"${TASKS_TEXT}"
MODES=(pca mlp vq cvae decoder_only ae)
CHECKPOINT_STEPS=(10000 20000 30000)
HORIZONS=(8 12 16)
read -r -a GPU_IDS <<<"${GPU_IDS_TEXT}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 2; }
(( ${#GPU_IDS[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU rank" >&2; exit 2; }
(( ${#TASKS[@]} > 0 )) || { echo "TASKS_TEXT must contain at least one task" >&2; exit 2; }
declare -A GPU_SEEN=()
for gpu in "${GPU_IDS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU rank: ${gpu}" >&2; exit 2; }
  [[ -z "${GPU_SEEN[$gpu]:-}" ]] || { echo "Duplicate GPU rank: ${gpu}" >&2; exit 2; }
  GPU_SEEN[$gpu]=1
done

mkdir -p "${OUTPUT_ROOT}/eval" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/.matplotlib"
exec 9>"${OUTPUT_ROOT}/.launcher.lock"
flock -n 9 || { echo "Another eval launcher is using ${OUTPUT_ROOT}" >&2; exit 3; }

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"
export TASKS_TEXT
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/.matplotlib}"

SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
SESSION_LOG="${OUTPUT_ROOT}/logs/launcher_${SESSION_ID}.log"
SUMMARY_CSV="${OUTPUT_ROOT}/evaluation_summary.csv"
CURRENT_PHASE=initialization

log() {
  local line
  line="[$(date -Is)] [a09_eval_4gpu] [${SESSION_ID}] $*"
  if [[ -n "${WORKER_LOG_FILE:-}" ]]; then
    printf '%s\n' "${line}" >>"${WORKER_LOG_FILE}"
  else
    printf '%s\n' "${line}" | tee -a "${OUTPUT_ROOT}/logs/launcher.log" "${SESSION_LOG}"
  fi
}

on_signal() {
  local signal="$1" status="$2" pid
  trap - HUP INT TERM
  log "received ${signal} during phase=${CURRENT_PHASE}; stopping child drivers"
  while read -r pid; do [[ -z "${pid}" ]] || kill "${pid}" 2>/dev/null || true; done < <(jobs -pr)
  write_summary || true
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

train_root_for_mode() {
  case "$1" in
    pca|mlp|vq) printf '%s' "${SOURCE_ROOT}/baselines/train" ;;
    cvae|decoder_only|ae) printf '%s' "${SOURCE_ROOT}/learned_priors/train" ;;
    *) return 2 ;;
  esac
}

original_eval_root_for_mode() {
  case "$1" in
    pca|mlp|vq) printf '%s' "${SOURCE_ROOT}/baselines/eval" ;;
    cvae|decoder_only|ae) printf '%s' "${SOURCE_ROOT}/learned_priors/eval" ;;
    *) return 2 ;;
  esac
}

policy_artifact() {
  local task="$1" mode="$2" step="$3" run_dir
  run_dir="$(train_root_for_mode "${mode}")/${task}_dp_a09_${mode}_seed${TRAIN_SEED}"
  printf '%s' "${run_dir}/checkpoints/global_step_${step}/actor/artifact"
}

declare -A MODEL_SHA_CACHE=()
MODEL_SHA_RESULT=""
get_model_sha() {
  local artifact="$1" line value=""
  if [[ -n "${MODEL_SHA_CACHE[$artifact]:-}" ]]; then
    MODEL_SHA_RESULT="${MODEL_SHA_CACHE[$artifact]}"
    return 0
  fi
  while IFS= read -r line; do
    if [[ "${line}" =~ \"model_sha256\"[[:space:]]*:[[:space:]]*\"([0-9a-fA-F]+)\" ]]; then
      value="${BASH_REMATCH[1]}"
      break
    fi
  done <"${artifact}/artifact.json"
  [[ "${value}" =~ ^[0-9a-fA-F]{64}$ ]] || return 1
  MODEL_SHA_CACHE[$artifact]="${value}"
  MODEL_SHA_RESULT="${value}"
}

eval_contract() {
  local sha="$1" artifact="$2" step="$3" horizon="$4"
  printf '%s' "${sha}|${artifact}|step=${step}|k=${horizon}|ddim=${DDIM_STEPS}|env=${EVAL_SEED_START}-${EVAL_SEED_END}|noise=${POLICY_NOISE_SEED}-${EVAL_SEED_END}|no_temporal"
}

try_import_existing_30k_k8() {
  local task="$1" mode="$2" artifact="$3" sha="$4" target_dir="$5"
  local old_dir old_contract
  old_dir="$(original_eval_root_for_mode "${mode}")/${task}_dp_a09_${mode}_seed${TRAIN_SEED}_k8_ddim${DDIM_STEPS}_env0-49"
  [[ -f "${old_dir}/.complete" && -s "${old_dir}/metrics.log" && -f "${old_dir}/contract.txt" ]] || return 1
  old_contract="$(<"${old_dir}/contract.txt")"
  [[ "${old_contract}" == "${sha}|"* && "${old_contract}" == *"|k=8|ddim=${DDIM_STEPS}|env=0-49|noise=0-49|no_temporal" ]] || return 1
  mkdir -p "${target_dir}"
  printf '%s\n' "${old_dir}/metrics.log" >"${target_dir}/source_metrics.txt"
  eval_contract "${sha}" "${artifact}" 30000 8 >"${target_dir}/contract.txt"
  touch "${target_dir}/.complete"
}

metrics_for_dir() {
  local dir="$1" source
  if [[ -s "${dir}/source_metrics.txt" ]]; then
    source="$(<"${dir}/source_metrics.txt")"
    [[ -s "${source}" ]] && { printf '%s' "${source}"; return 0; }
  fi
  [[ -s "${dir}/metrics.log" ]] && printf '%s' "${dir}/metrics.log"
}

evaluation_complete() {
  local task="$1" mode="$2" step="$3" horizon="$4"
  local artifact sha eid edir expected existing
  artifact="$(policy_artifact "${task}" "${mode}" "${step}")"
  eid="${task}_dp_a09_${mode}_seed${TRAIN_SEED}_step${step}_k${horizon}_ddim${DDIM_STEPS}_env0-49"
  edir="${OUTPUT_ROOT}/eval/${eid}"
  # Most pending jobs have no completion marker. Avoid opening large artifact
  # metadata files on shared storage unless a result actually claims complete.
  if [[ ! -f "${edir}/.complete" ]]; then
    if [[ "${step}" == 30000 && "${horizon}" == 8 ]]; then
      artifact_ready "${artifact}" || return 1
      get_model_sha "${artifact}" || return 1
      sha="${MODEL_SHA_RESULT}"
      if try_import_existing_30k_k8 "${task}" "${mode}" "${artifact}" "${sha}" "${edir}"; then
        log "import existing eval ${eid}"
        return 0
      fi
    fi
    return 1
  fi
  artifact_ready "${artifact}" || return 1
  get_model_sha "${artifact}" || return 1
  sha="${MODEL_SHA_RESULT}"
  expected="$(eval_contract "${sha}" "${artifact}" "${step}" "${horizon}")"
  existing="$(metrics_for_dir "${edir}" || true)"
  if [[ -f "${edir}/.complete" && -f "${edir}/contract.txt" && "$(<"${edir}/contract.txt")" == "${expected}" && -n "${existing}" ]]; then
    return 0
  fi
  return 1
}

run_eval() {
  local gpu="$1" task="$2" mode="$3" step="$4" horizon="$5"
  local method artifact sha eid edir log_file expected existing
  [[ "${horizon}" == 8 ]] && method=checkpoint_sweep || method=horizon_sweep
  artifact="$(policy_artifact "${task}" "${mode}" "${step}")"
  eid="${task}_dp_a09_${mode}_seed${TRAIN_SEED}_step${step}_k${horizon}_ddim${DDIM_STEPS}_env0-49"
  edir="${OUTPUT_ROOT}/eval/${eid}"
  log_file="${OUTPUT_ROOT}/logs/${eid}.log"

  if [[ "${DRY_RUN}" == 1 ]]; then
    log "would eval gpu=${gpu} method=${method} ${eid}"
    return 0
  fi
  artifact_ready "${artifact}" || { log "FAILED missing/corrupt artifact ${artifact}"; return 1; }
  get_model_sha "${artifact}" || { log "FAILED invalid artifact metadata ${artifact}"; return 1; }
  sha="${MODEL_SHA_RESULT}"
  expected="$(eval_contract "${sha}" "${artifact}" "${step}" "${horizon}")"
  existing="$(metrics_for_dir "${edir}" || true)"
  if [[ -f "${edir}/.complete" && -f "${edir}/contract.txt" && "$(<"${edir}/contract.txt")" == "${expected}" && -n "${existing}" ]]; then
    log "reuse eval ${eid}"
    return 0
  fi
  if [[ "${step}" == 30000 && "${horizon}" == 8 ]] && try_import_existing_30k_k8 "${task}" "${mode}" "${artifact}" "${sha}" "${edir}"; then
    log "import existing eval ${eid}"
    return 0
  fi

  mkdir -p "${edir}"
  rm -f "${edir}/.complete" "${edir}/source_metrics.txt"
  printf '%s\n' "${expected}" >"${edir}/contract.txt"
  log "start eval gpu=${gpu} method=${method} ${eid}"
  if ! "${PYTHON_BIN}" evaluations/eval_embodied_agent.py \
      --config-path "${REPO_ROOT}/evaluations/dexjoco" \
      --config-name "dexjoco_lamp_dp_50seed_${task}_eval" \
      "cluster.component_placement={env\, rollout:${gpu}-${gpu}}" \
      "env.eval.seed=${EVAL_SEED_START}" "env.eval.total_num_envs=${EVAL_ENVS}" \
      "rollout.model.model_path=${artifact}" \
      "rollout.model.use_temporal_ensemble=false" \
      "rollout.model.execution_horizon_override=${horizon}" \
      "rollout.model.num_inference_steps_override=${DDIM_STEPS}" \
      "rollout.model.eval_base_noise_seed=${POLICY_NOISE_SEED}" \
      "runner.logger.log_path=${edir}" "runner.logger.experiment_name=${eid}" \
      >"${log_file}" 2>&1; then
    log "FAILED eval ${eid}; see ${log_file}"
    return 1
  fi
  [[ -s "${edir}/metrics.log" ]] || { log "FAILED missing metrics ${eid}"; return 1; }
  touch "${edir}/.complete"
  log "finish eval ${eid}"
}

write_summary() {
  local tmp="${SUMMARY_CSV}.tmp.$$"
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${SOURCE_ROOT}" "${tmp}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

output_root, source_root, destination = map(Path, sys.argv[1:])
pattern = re.compile(
    r"^(?P<task>.+)_dp_a09_(?P<mode>pca|mlp|vq|cvae|decoder_only|ae)_seed42_"
    r"step(?P<step>\d+)_k(?P<k>\d+)_ddim(?P<ddim>\d+)_env0-49$"
)
metric_patterns = {
    "success_once": re.compile(r"success_once=([0-9.eE+-]+)"),
    "return": re.compile(r"(?:^|[│|\s])return=([0-9.eE+-]+)"),
    "episode_len": re.compile(r"episode_len=([0-9.eE+-]+)"),
    "num_trajectories": re.compile(r"num_trajectories=([0-9.eE+-]+)"),
}
rows = []
for eval_dir in sorted((output_root / "eval").glob("*")):
    match = pattern.match(eval_dir.name)
    if not match:
        continue
    source_file = eval_dir / "source_metrics.txt"
    metrics = Path(source_file.read_text().strip()) if source_file.is_file() else eval_dir / "metrics.log"
    complete = (eval_dir / ".complete").is_file()
    values = {key: "" for key in metric_patterns}
    if metrics.is_file():
        text = metrics.read_text(errors="ignore")
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        for key, regex in metric_patterns.items():
            found = regex.findall(text)
            values[key] = found[-1] if found else ""
    contract = (eval_dir / "contract.txt").read_text().strip() if (eval_dir / "contract.txt").is_file() else ""
    fields = match.groupdict()
    artifact = contract.split("|")[1] if contract.count("|") >= 1 else ""
    sha = contract.split("|")[0] if contract else ""
    rows.append({
        "task": fields["task"],
        "mode": fields["mode"],
        "evaluation_method": "checkpoint_sweep" if fields["k"] == "8" else "horizon_sweep",
        "checkpoint_step": fields["step"],
        "execution_horizon": fields["k"],
        "ddim_steps": fields["ddim"],
        "train_seed": 42,
        "env_seed_start": 0,
        "env_seed_end": 49,
        "policy_noise_seed_start": 0,
        "num_envs": 50,
        "success_once": values["success_once"],
        "return": values["return"],
        "episode_len": values["episode_len"],
        "num_trajectories": values["num_trajectories"],
        "status": "complete" if complete and values["success_once"] else ("invalid" if complete else "pending"),
        "model_sha256": sha,
        "artifact_path": artifact,
        "metrics_path": str(metrics),
    })
columns = [
    "task", "mode", "evaluation_method", "checkpoint_step", "execution_horizon", "ddim_steps",
    "train_seed", "env_seed_start", "env_seed_end", "policy_noise_seed_start", "num_envs",
    "success_once", "return", "episode_len", "num_trajectories", "status", "model_sha256",
    "artifact_path", "metrics_path",
]
with destination.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
PY
  mv "${tmp}" "${SUMMARY_CSV}"
}

preflight() {
  local task mode step artifact count=0
  for task in "${TASKS[@]}"; do
    [[ -f "${REPO_ROOT}/evaluations/dexjoco/dexjoco_lamp_dp_50seed_${task}_eval.yaml" ]] || {
      echo "Missing evaluation config for ${task}" >&2; return 1;
    }
    for mode in "${MODES[@]}"; do
      for step in "${CHECKPOINT_STEPS[@]}"; do
        artifact="$(policy_artifact "${task}" "${mode}" "${step}")"
        artifact_ready "${artifact}" || { echo "Missing/corrupt artifact: ${artifact}" >&2; return 1; }
        count=$((count + 1))
      done
    done
  done
  "${PYTHON_BIN}" - <<'PY'
import huggingface_hub
from pathlib import Path
from hydra import compose, initialize_config_dir
from transformers import ResNetConfig, ResNetModel

import os
assert tuple(map(int, huggingface_hub.__version__.split(".")[:1])) < (1,), (
    f"incompatible huggingface-hub={huggingface_hub.__version__}; expected >=0.34,<1.0"
)
tasks = os.environ["TASKS_TEXT"].split()
with initialize_config_dir(version_base=None, config_dir=str(Path("evaluations/dexjoco").resolve())):
    for task in tasks:
        cfg = compose(config_name=f"dexjoco_lamp_dp_50seed_{task}_eval")
        assert cfg.rollout.model.use_temporal_ensemble is False
PY
  log "preflight passed: tasks=${#TASKS[@]}, artifacts=${count}, planned_evals=$(( ${#TASKS[@]} * 30 )), gpus=${GPU_IDS[*]}"
}

run_jobs() {
  local -a planned_jobs=() jobs=() pids=() labels=()
  local task mode step horizon record gpu i failed=0 slot=0 skipped=0
  # Import/reuse all existing final-checkpoint k=8 anchors first so they do not
  # leave a GPU idle behind a batch barrier while new evaluations are running.
  for task in "${TASKS[@]}"; do for mode in "${MODES[@]}"; do
    planned_jobs+=("${task}|${mode}|30000|8")
  done; done
  # Method 1: add all preceding checkpoints at the same k=8 horizon.
  for task in "${TASKS[@]}"; do for mode in "${MODES[@]}"; do for step in 10000 20000; do
    planned_jobs+=("${task}|${mode}|${step}|8")
  done; done; done
  # Method 2: horizon sweep at the final checkpoint. k=8 is shared above.
  for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do for horizon in 12 16; do
      planned_jobs+=("${task}|${mode}|30000|${horizon}")
    done; done
  done

  # Remove already completed experiments from the launch queue. Completion is
  # accepted only when the marker, metrics, artifact SHA, and full eval contract
  # all match the current request.
  for record in "${planned_jobs[@]}"; do
    IFS='|' read -r task mode step horizon <<<"${record}"
    if evaluation_complete "${task}" "${mode}" "${step}" "${horizon}"; then
      skipped=$((skipped + 1))
    else
      jobs+=("${record}")
    fi
  done

  write_summary
  CURRENT_PHASE=evaluation
  log "phase start evaluation: planned=${#planned_jobs[@]} completed_removed=${skipped} remaining=${#jobs[@]} capacity=${#GPU_IDS[@]}"
  if (( ${#jobs[@]} == 0 )); then
    log "phase complete evaluation: nothing remaining"
    return 0
  fi
  if [[ "${DRY_RUN}" == 1 ]]; then
    for record in "${jobs[@]}"; do log "would eval ${record}"; done
    return 0
  fi
  for record in "${jobs[@]}"; do
    gpu="${GPU_IDS[$slot]}"
    IFS='|' read -r task mode step horizon <<<"${record}"
    (
      WORKER_LOG_FILE="${OUTPUT_ROOT}/logs/worker_gpu${gpu}.log"
      sleep $((slot * START_STAGGER_SECONDS))
      run_eval "${gpu}" "${task}" "${mode}" "${step}" "${horizon}"
    ) &
    pids+=("$!"); labels+=("${record}"); slot=$((slot + 1))
    if (( slot == ${#GPU_IDS[@]} )); then
      for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
          log "job complete ${labels[$i]}"
        else
          log "FAILED job ${labels[$i]}"; failed=1
        fi
      done
      write_summary
      pids=(); labels=(); slot=0
    fi
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "job complete ${labels[$i]}"
    else
      log "FAILED job ${labels[$i]}"; failed=1
    fi
  done
  write_summary
  (( failed == 0 )) || return 1
  log "phase complete evaluation"
}

log "launcher started source=${SOURCE_ROOT} output=${OUTPUT_ROOT}"
preflight
[[ "${PREFLIGHT_ONLY}" == 1 ]] && { write_summary; exit 0; }
if [[ "${DRY_RUN}" == 1 ]]; then
  run_jobs
  exit 0
fi
run_jobs
CURRENT_PHASE=summary
log "complete summary=${SUMMARY_CSV}"
