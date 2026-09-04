#!/usr/bin/env bash
# Evaluate six selected single-arm DP checkpoints without temporal ensemble.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
TASK="${TASK:-water_plant}"
case "${TASK}" in
  water_plant|pick_bucket)
    DEFAULT_SOURCE_ROOT="${REPO_ROOT}/outputs/lamp_prior_modes_z2_selected_lr3e-5_water_plant_pick_bucket"
    ;;
  hammer_nail|fold_glasses)
    DEFAULT_SOURCE_ROOT="${REPO_ROOT}/outputs/lamp_prior_modes_z2_selected_lr3e-5_hammer_nail_fold_glasses"
    ;;
  click_mouse|pinch_tongs)
    DEFAULT_SOURCE_ROOT="${REPO_ROOT}/outputs/lamp_prior_modes_z2_selected_lr3e-5_click_mouse_pinch_tongs"
    ;;
  *)
    echo "Unsupported TASK: ${TASK}" >&2
    exit 2
    ;;
esac
SOURCE_ROOT="${SOURCE_ROOT:-${DEFAULT_SOURCE_ROOT}}"
AE_SOURCE_ROOT="${AE_SOURCE_ROOT:-${REPO_ROOT}/outputs/lamp_ae_z2_selected_lr3e-5}"
AE_RUN_ROOT="${AE_RUN_ROOT:-${AE_SOURCE_ROOT}/${TASK}_dp_ae_z2_lr3e-5}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-4}"
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ "${TASK}" == water_plant ]]; then
    OUTPUT_ROOT="${SOURCE_ROOT}/eval_selected_no_temporal_k${EXECUTION_HORIZON}_2gpu"
  else
    OUTPUT_ROOT="${SOURCE_ROOT}/eval_${TASK}_selected_no_temporal_k${EXECUTION_HORIZON}_2gpu"
  fi
fi
GPU_IDS_TEXT="${GPU_IDS:-0 1}"
CHECKPOINT_STEPS_TEXT="${CHECKPOINT_STEPS:-30000}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-5}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"

DDIM_STEPS=16
EVAL_SEED_START=0
EVAL_SEED_END=49
POLICY_NOISE_SEED=0
EVAL_ENVS=50
MODES=(cvae decoder_only ae mlp pca vq)

read -r -a GPU_IDS <<<"${GPU_IDS_TEXT}"
read -r -a CHECKPOINT_STEPS <<<"${CHECKPOINT_STEPS_TEXT}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 2; }
command -v pgrep >/dev/null || { echo "pgrep is required" >&2; exit 2; }
(( ${#GPU_IDS[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU rank" >&2; exit 2; }
(( ${#CHECKPOINT_STEPS[@]} > 0 )) || { echo "CHECKPOINT_STEPS must not be empty" >&2; exit 2; }
[[ "${EXECUTION_HORIZON}" =~ ^[1-9][0-9]*$ ]] || {
  echo "EXECUTION_HORIZON must be a positive integer: ${EXECUTION_HORIZON}" >&2
  exit 2
}
(( EXECUTION_HORIZON <= 16 )) || {
  echo "EXECUTION_HORIZON must not exceed the artifact action horizon 16" >&2
  exit 2
}

declare -A GPU_SEEN=()
for gpu in "${GPU_IDS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU rank: ${gpu}" >&2; exit 2; }
  [[ -z "${GPU_SEEN[$gpu]:-}" ]] || { echo "Duplicate GPU rank: ${gpu}" >&2; exit 2; }
  GPU_SEEN[$gpu]=1
done
for step in "${CHECKPOINT_STEPS[@]}"; do
  [[ "${step}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid checkpoint step: ${step}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_ROOT}/eval" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/.matplotlib"
exec 9>"${OUTPUT_ROOT}/.launcher.lock"
flock -n 9 || { echo "Another evaluator is using ${OUTPUT_ROOT}" >&2; exit 3; }

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/.matplotlib}"

SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
SESSION_LOG="${OUTPUT_ROOT}/logs/launcher_${SESSION_ID}.log"
SUMMARY_CSV="${OUTPUT_ROOT}/evaluation_summary.csv"
CURRENT_PHASE=initialization
declare -A ARTIFACT_SHA_CACHE=()
ARTIFACT_SHA_RESULT=""

log() {
  local line
  line="[$(date -Is)] [${TASK}_no_temporal] [${SESSION_ID}] $*"
  if [[ -n "${WORKER_LOG_FILE:-}" ]]; then
    printf '%s\n' "${line}" >>"${WORKER_LOG_FILE}"
  else
    printf '%s\n' "${line}" | tee -a "${OUTPUT_ROOT}/logs/launcher.log" "${SESSION_LOG}"
  fi
}

write_summary() {
  local tmp="${SUMMARY_CSV}.tmp.$$"
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${tmp}" "${TASK}" <<'PY'
import csv
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
destination = Path(sys.argv[2])
task = sys.argv[3]
pattern = re.compile(
    rf"^{re.escape(task)}_(?P<mode>cvae|decoder_only|ae|mlp|pca|vq)_"
    r"step(?P<step>\d+)_k(?P<k>\d+)_ddim(?P<ddim>\d+)_env0-49_no_temporal$"
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
    metrics = eval_dir / "metrics.log"
    values = {key: "" for key in metric_patterns}
    if metrics.is_file():
        text = metrics.read_text(errors="ignore")
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        for key, regex in metric_patterns.items():
            found = regex.findall(text)
            values[key] = found[-1] if found else ""
    contract_path = eval_dir / "contract.txt"
    contract = contract_path.read_text().strip() if contract_path.is_file() else ""
    parts = contract.split("|") if contract else []
    fields = match.groupdict()
    complete = (eval_dir / ".complete").is_file()
    rows.append(
        {
            "task": task,
            "mode": fields["mode"],
            "checkpoint_step": fields["step"],
            "execution_horizon": fields["k"],
            "ddim_steps": fields["ddim"],
            "temporal_ensemble": "false",
            "env_seed_start": 0,
            "env_seed_end": 49,
            "policy_noise_seed_start": 0,
            "num_envs": 50,
            "success_once": values["success_once"],
            "return": values["return"],
            "episode_len": values["episode_len"],
            "num_trajectories": values["num_trajectories"],
            "status": "complete" if complete and values["success_once"] else (
                "invalid" if complete else "pending"
            ),
            "model_sha256": parts[0] if parts else "",
            "artifact_path": parts[1] if len(parts) > 1 else "",
            "metrics_path": str(metrics),
        }
    )

columns = [
    "task",
    "mode",
    "checkpoint_step",
    "execution_horizon",
    "ddim_steps",
    "temporal_ensemble",
    "env_seed_start",
    "env_seed_end",
    "policy_noise_seed_start",
    "num_envs",
    "success_once",
    "return",
    "episode_len",
    "num_trajectories",
    "status",
    "model_sha256",
    "artifact_path",
    "metrics_path",
]
with destination.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
PY
  mv "${tmp}" "${SUMMARY_CSV}"
}

signal_process_tree() {
  local parent="$1" signal="$2" child
  while IFS= read -r child; do
    [[ -z "${child}" ]] || signal_process_tree "${child}" "${signal}"
  done < <(pgrep -P "${parent}" 2>/dev/null || true)
  kill -s "${signal}" "${parent}" 2>/dev/null || true
}

on_signal() {
  local signal="$1" status="$2" pid
  local -a active_pids=()
  trap - HUP INT TERM
  log "received ${signal} during phase=${CURRENT_PHASE}; stopping child evaluators"
  mapfile -t active_pids < <(jobs -pr)
  for pid in "${active_pids[@]}"; do
    [[ -z "${pid}" ]] || signal_process_tree "${pid}" INT
  done
  sleep 1
  for pid in "${active_pids[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      signal_process_tree "${pid}" TERM
    fi
  done
  write_summary || true
  log "interrupted results remain resumable at ${OUTPUT_ROOT}"
  exit "${status}"
}
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

run_name_for_mode() {
  case "$1" in
    cvae) printf '%s' "${TASK}_dp_cvae_cvae_z2_selected_lr3e-5" ;;
    decoder_only) printf '%s' "${TASK}_dp_decoder_only_cvae_z2_selected_lr3e-5" ;;
    ae) printf '%s' "${TASK}_dp_ae_z2_lr3e-5" ;;
    mlp) printf '%s' "${TASK}_dp_mlp_lr3e-5" ;;
    pca) printf '%s' "${TASK}_dp_pca_z2_lr3e-5" ;;
    vq) printf '%s' "${TASK}_dp_vq_lr3e-5" ;;
    *) return 2 ;;
  esac
}

artifact_for() {
  local mode="$1" step="$2" run_name
  if [[ "${mode}" == ae ]]; then
    printf '%s' "${AE_RUN_ROOT}/checkpoints/global_step_${step}/actor/artifact"
    return 0
  fi
  run_name="$(run_name_for_mode "${mode}")"
  printf '%s' "${SOURCE_ROOT}/${run_name}/checkpoints/global_step_${step}/actor/artifact"
}

artifact_ready() {
  local root="$1" header
  [[ -s "${root}/model.safetensors" && -s "${root}/artifact.json" && -s "${root}/statistics.npz" ]] || return 1
  header="$(od -An -tu8 -N8 "${root}/model.safetensors" 2>/dev/null | tr -d '[:space:]')"
  [[ "${header}" =~ ^[0-9]+$ ]] && (( header > 1 ))
}

validate_artifact() {
  local artifact="$1" mode="$2" sha
  if [[ -n "${ARTIFACT_SHA_CACHE[$artifact]:-}" ]]; then
    ARTIFACT_SHA_RESULT="${ARTIFACT_SHA_CACHE[$artifact]}"
    return 0
  fi
  sha="$("${PYTHON_BIN}" - "${artifact}" "${mode}" "${TASK}" <<'PY'
import json
import sys
from pathlib import Path

artifact = Path(sys.argv[1])
mode = sys.argv[2]
task = sys.argv[3]
metadata = json.loads((artifact / "artifact.json").read_text())
spec = metadata.get("spec", {})
expected_prior = {
    "cvae": "cvae",
    "decoder_only": "decoder_only",
    "ae": "ae",
    "mlp": "mlp",
    "pca": "pca",
    "vq": "vq_codebook",
}[mode]
requirements = {
    "model_type": metadata.get("model_type") == "lamp_dp",
    "task": metadata.get("task") == task,
    "policy_family": spec.get("policy_family") == "dp",
    "embodiment": spec.get("embodiment") == "single",
    "action_horizon": int(spec.get("action_horizon", -1)) == 16,
    "physical_action_dim": int(spec.get("physical_action_dim", -1)) == 23,
    "hand_prior_type": spec.get("hand_prior_type") == expected_prior,
}
failed = [name for name, valid in requirements.items() if not valid]
if failed:
    raise SystemExit(f"invalid {mode} artifact {artifact}: failed {failed}; spec={spec}")
sha = metadata.get("model_sha256", "")
if len(sha) != 64 or any(char not in "0123456789abcdefABCDEF" for char in sha):
    raise SystemExit(f"invalid model_sha256 in {artifact}")
print(sha)
PY
  )" || return 1
  ARTIFACT_SHA_CACHE[$artifact]="${sha}"
  ARTIFACT_SHA_RESULT="${sha}"
}

eval_id() {
  local mode="$1" step="$2"
  printf '%s' "${TASK}_${mode}_step${step}_k${EXECUTION_HORIZON}_ddim${DDIM_STEPS}_env0-49_no_temporal"
}

eval_contract() {
  local sha="$1" artifact="$2" step="$3" mode="$4"
  printf '%s' "${sha}|${artifact}|mode=${mode}|step=${step}|k=${EXECUTION_HORIZON}|ddim=${DDIM_STEPS}|env=${EVAL_SEED_START}-${EVAL_SEED_END}|noise=${POLICY_NOISE_SEED}-${EVAL_SEED_END}|temporal=false"
}

evaluation_complete() {
  local mode="$1" step="$2" artifact sha eid edir expected
  artifact="$(artifact_for "${mode}" "${step}")"
  eid="$(eval_id "${mode}" "${step}")"
  edir="${OUTPUT_ROOT}/eval/${eid}"
  [[ -f "${edir}/.complete" && -s "${edir}/metrics.log" && -f "${edir}/contract.txt" ]] || return 1
  artifact_ready "${artifact}" || return 1
  validate_artifact "${artifact}" "${mode}" || return 1
  sha="${ARTIFACT_SHA_RESULT}"
  expected="$(eval_contract "${sha}" "${artifact}" "${step}" "${mode}")"
  [[ "$(<"${edir}/contract.txt")" == "${expected}" ]]
}

run_eval() {
  local gpu="$1" mode="$2" step="$3"
  local artifact sha eid edir log_file expected existing_contract
  artifact="$(artifact_for "${mode}" "${step}")"
  eid="$(eval_id "${mode}" "${step}")"
  edir="${OUTPUT_ROOT}/eval/${eid}"
  log_file="${OUTPUT_ROOT}/logs/${eid}.log"

  artifact_ready "${artifact}" || { log "FAILED missing/corrupt artifact ${artifact}"; return 1; }
  validate_artifact "${artifact}" "${mode}" || { log "FAILED artifact metadata ${artifact}"; return 1; }
  sha="${ARTIFACT_SHA_RESULT}"
  expected="$(eval_contract "${sha}" "${artifact}" "${step}" "${mode}")"
  if [[ -f "${edir}/contract.txt" ]]; then
    existing_contract="$(<"${edir}/contract.txt")"
    if [[ "${existing_contract}" != "${expected}" ]]; then
      log "FAILED contract mismatch for ${eid}; choose a new OUTPUT_ROOT to preserve existing results"
      return 1
    fi
  fi
  if evaluation_complete "${mode}" "${step}"; then
    log "reuse eval ${eid}"
    return 0
  fi
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "would eval gpu=${gpu} mode=${mode} step=${step} artifact=${artifact} temporal=false k=${EXECUTION_HORIZON}"
    return 0
  fi

  mkdir -p "${edir}"
  rm -f "${edir}/.complete"
  printf '%s\n' "${expected}" >"${edir}/contract.txt"
  log "start eval gpu=${gpu} ${eid}"
  if ! "${PYTHON_BIN}" evaluations/eval_embodied_agent.py \
      --config-path "${REPO_ROOT}/evaluations/dexjoco" \
      --config-name "dexjoco_lamp_dp_50seed_${TASK}_eval" \
      "cluster.component_placement={env\, rollout:${gpu}-${gpu}}" \
      "env.eval.seed=${EVAL_SEED_START}" \
      "env.eval.total_num_envs=${EVAL_ENVS}" \
      "rollout.model.model_path=${artifact}" \
      "rollout.model.use_temporal_ensemble=false" \
      "rollout.model.execution_horizon_override=${EXECUTION_HORIZON}" \
      "rollout.model.num_inference_steps_override=${DDIM_STEPS}" \
      "rollout.model.eval_base_noise_seed=${POLICY_NOISE_SEED}" \
      "runner.logger.log_path=${edir}" \
      "runner.logger.experiment_name=${eid}" \
      >"${log_file}" 2>&1; then
    log "FAILED eval ${eid}; see ${log_file}"
    return 1
  fi
  [[ -s "${edir}/metrics.log" ]] || { log "FAILED missing metrics ${eid}"; return 1; }
  touch "${edir}/.complete"
  log "finish eval ${eid}"
}

preflight() {
  local mode step artifact sha count=0
  local total=$((${#MODES[@]} * ${#CHECKPOINT_STEPS[@]}))
  CURRENT_PHASE=preflight
  [[ -f "${REPO_ROOT}/evaluations/dexjoco/dexjoco_lamp_dp_50seed_${TASK}_eval.yaml" ]] || {
    echo "Missing ${TASK} evaluation config" >&2
    return 1
  }
  for mode in "${MODES[@]}"; do
    for step in "${CHECKPOINT_STEPS[@]}"; do
      log "preflight artifact $((count + 1))/${total}: mode=${mode} step=${step}"
      artifact="$(artifact_for "${mode}" "${step}")"
      artifact_ready "${artifact}" || { echo "Missing/corrupt artifact: ${artifact}" >&2; return 1; }
      validate_artifact "${artifact}" "${mode}" || return 1
      sha="${ARTIFACT_SHA_RESULT}"
      [[ -n "${sha}" ]] || return 1
      count=$((count + 1))
    done
  done
  log "preflight passed: artifacts=${count}, temporal=false, k=${EXECUTION_HORIZON}, env=0-49, gpus=${GPU_IDS[*]}"
}

run_jobs() {
  local -a jobs=() completed=() pids=() labels=()
  local mode step record gpu slot=0 i failed=0 skipped=0
  log "resume scan: checking exact-contract completion markers"
  for mode in "${MODES[@]}"; do
    for step in "${CHECKPOINT_STEPS[@]}"; do
      if evaluation_complete "${mode}" "${step}"; then
        skipped=$((skipped + 1))
        completed+=("${mode}|${step}")
      else
        jobs+=("${mode}|${step}")
      fi
    done
  done

  write_summary
  CURRENT_PHASE=evaluation
  log "phase start: planned=$((${#MODES[@]} * ${#CHECKPOINT_STEPS[@]})) reused=${skipped} remaining=${#jobs[@]} capacity=${#GPU_IDS[@]}"
  if (( ${#completed[@]} > 0 )); then
    log "resume reuse complete: ${completed[*]}"
  fi
  if (( ${#jobs[@]} > 0 )); then
    log "resume pending: ${jobs[*]}"
  fi
  for record in "${jobs[@]}"; do
    gpu="${GPU_IDS[$slot]}"
    IFS='|' read -r mode step <<<"${record}"
    (
      WORKER_LOG_FILE="${OUTPUT_ROOT}/logs/worker_gpu${gpu}.log"
      sleep $((slot * START_STAGGER_SECONDS))
      run_eval "${gpu}" "${mode}" "${step}"
    ) &
    pids+=("$!")
    labels+=("${record}")
    slot=$((slot + 1))
    if (( slot == ${#GPU_IDS[@]} )); then
      for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
          log "job complete ${labels[$i]}"
        else
          log "FAILED job ${labels[$i]}"
          failed=1
        fi
      done
      write_summary
      pids=()
      labels=()
      slot=0
    fi
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "job complete ${labels[$i]}"
    else
      log "FAILED job ${labels[$i]}"
      failed=1
    fi
  done
  write_summary
  (( failed == 0 )) || return 1
  log "phase complete"
}

log "launcher started task=${TASK} source=${SOURCE_ROOT} output=${OUTPUT_ROOT} steps=${CHECKPOINT_STEPS[*]}"
preflight
if [[ "${PREFLIGHT_ONLY}" == 1 ]]; then
  write_summary
  exit 0
fi
run_jobs
CURRENT_PHASE=summary
log "complete summary=${SUMMARY_CSV}"
