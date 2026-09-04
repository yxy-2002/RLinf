#!/usr/bin/env bash
# Run the four-task half of the selected K=8 no-temporal evaluation campaign.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SINGLE_TASK_LAUNCHER="${SCRIPT_DIR}/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2 3}"
TASKS=(click_mouse pinch_tongs hammer_nail fold_glasses)

[[ -x "${SINGLE_TASK_LAUNCHER}" ]] || {
  echo "Single-task launcher is not executable: ${SINGLE_TASK_LAUNCHER}" >&2
  exit 2
}

printf '[%s] [selected_k8_4gpu] tasks=%s gpus=%s\n' \
  "$(date -Is)" "${TASKS[*]}" "${GPU_IDS_TEXT}"
for task in "${TASKS[@]}"; do
  case "${task}" in
    click_mouse|pinch_tongs)
      source_root="${REPO_ROOT}/outputs/lamp_prior_modes_z2_selected_lr3e-5_click_mouse_pinch_tongs"
      ;;
    hammer_nail|fold_glasses)
      source_root="${REPO_ROOT}/outputs/lamp_prior_modes_z2_selected_lr3e-5_hammer_nail_fold_glasses"
      ;;
  esac
  output_root="${source_root}/eval_${task}_selected_no_temporal_k8_4gpu"
  printf '[%s] [selected_k8_4gpu] start task=%s output=%s\n' \
    "$(date -Is)" "${task}" "${output_root}"
  env -u SOURCE_ROOT -u AE_RUN_ROOT \
    TASK="${task}" \
    GPU_IDS="${GPU_IDS_TEXT}" \
    EXECUTION_HORIZON=8 \
    OUTPUT_ROOT="${output_root}" \
    "${SINGLE_TASK_LAUNCHER}"
  printf '[%s] [selected_k8_4gpu] finish task=%s\n' "$(date -Is)" "${task}"
done
printf '[%s] [selected_k8_4gpu] all tasks complete\n' "$(date -Is)"
