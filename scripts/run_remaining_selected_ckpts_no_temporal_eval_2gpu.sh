#!/usr/bin/env bash
# Evaluate all still-needed selected single-arm policies with temporal ensemble disabled.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_TASK_LAUNCHER="${SCRIPT_DIR}/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh"
TASKS_TEXT="${TASKS:-click_mouse pinch_tongs hammer_nail fold_glasses}"
BATCH_OUTPUT_ROOT="${BATCH_OUTPUT_ROOT:-}"

read -r -a TASK_LIST <<<"${TASKS_TEXT}"
(( ${#TASK_LIST[@]} > 0 )) || { echo "TASKS must not be empty" >&2; exit 2; }
[[ -x "${SINGLE_TASK_LAUNCHER}" ]] || {
  echo "Single-task launcher is not executable: ${SINGLE_TASK_LAUNCHER}" >&2
  exit 2
}

declare -A TASK_SEEN=()
for task in "${TASK_LIST[@]}"; do
  case "${task}" in
    click_mouse|pinch_tongs|hammer_nail|fold_glasses) ;;
    water_plant|pick_bucket)
      echo "TASKS must exclude already evaluated task: ${task}" >&2
      exit 2
      ;;
    *)
      echo "Unsupported remaining task: ${task}" >&2
      exit 2
      ;;
  esac
  [[ -z "${TASK_SEEN[$task]:-}" ]] || { echo "Duplicate task: ${task}" >&2; exit 2; }
  TASK_SEEN[$task]=1
done

printf '[%s] [remaining_no_temporal] tasks=%s\n' "$(date -Is)" "${TASK_LIST[*]}"
for task in "${TASK_LIST[@]}"; do
  printf '[%s] [remaining_no_temporal] start task=%s\n' "$(date -Is)" "${task}"
  if [[ -n "${BATCH_OUTPUT_ROOT}" ]]; then
    env -u SOURCE_ROOT -u AE_RUN_ROOT \
      TASK="${task}" OUTPUT_ROOT="${BATCH_OUTPUT_ROOT}/${task}" \
      "${SINGLE_TASK_LAUNCHER}"
  else
    env -u SOURCE_ROOT -u AE_RUN_ROOT -u OUTPUT_ROOT \
      TASK="${task}" "${SINGLE_TASK_LAUNCHER}"
  fi
  printf '[%s] [remaining_no_temporal] finish task=%s\n' "$(date -Is)" "${task}"
done
printf '[%s] [remaining_no_temporal] all tasks complete\n' "$(date -Is)"
