#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/train_ruiyan_lamp.sh VARIANT STAGE [--dry-run] [Hydra overrides...]
  VARIANT: lamplstm_film | lamplstm_none | lamplstm_concat | pca | vq | mlp | all
  STAGE:   prior | dp | both
  all both: sequentially run five prior/DP pairs, then the MLP DP.
  MLP has no prior stage; all prior skips MLP.

Environment overrides:
  PYTHON_BIN      Python executable (default: python3 from the active environment)
  DATASET_ROOT    Optional single demo directory (default: LSTM prior config)
  OUTPUT_ROOT     Artifact/log root (default: <repo>/outputs)
  RESNET_PATH     Local Hugging Face ResNet-18 (default: <repo>/pretrained_models/resnet-18)
  PRIOR_ARTIFACT  Existing prior artifact for a single-variant dp run

Examples:
  bash scripts/train_ruiyan_lamp.sh lamplstm_film both
  bash scripts/train_ruiyan_lamp.sh all both --dry-run
  bash scripts/train_ruiyan_lamp.sh mlp dp actor.micro_batch_size=32

Relative environment paths and Hydra paths are resolved from the repository root.
Additional Hydra overrides apply to both stages when STAGE=both.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if (( $# < 2 )); then usage >&2; exit 2; fi
variant=$1
stage=$2
shift 2
case "$variant" in lamplstm_film|lamplstm_none|lamplstm_concat|pca|vq|mlp|all) ;; *) usage >&2; exit 2 ;; esac
case "$stage" in prior|dp|both) ;; *) usage >&2; exit 2 ;; esac
if [[ $variant == mlp && $stage == prior ]]; then
  echo 'MLP has no standalone prior; use mlp dp.' >&2
  exit 2
fi
if [[ -n ${PRIOR_ARTIFACT:-} && ( $stage != dp || $variant == all || $variant == mlp ) ]]; then
  echo 'PRIOR_ARTIFACT is supported only for a single non-MLP variant with stage dp.' >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${PYTHON_BIN:-python3}
# Resolve the shared experiment dataset once, including YAML path lists.
# All prior/DP pairs must use the same collection for artifact compatibility.
if [[ -n ${DATASET_ROOT:-} ]]; then
  dataset_override="data.dataset_root='$DATASET_ROOT'"
else
  dataset_override=$("$python_bin" - <<'PYDATA'
import json
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(
    config_dir=str(Path("examples/embodiment/config").resolve()), version_base="1.1"
):
    cfg = compose(config_name="ruiyan_lamp_prior_lamplstm_stack_cube")
value = cfg.data.dataset_root
if OmegaConf.is_config(value):
    value = OmegaConf.to_container(value, resolve=True)
print("data.dataset_root=" + json.dumps(value))
PYDATA
  )
fi
output_root=${OUTPUT_ROOT:-$repo_root/outputs}
resnet_path=${RESNET_PATH:-$repo_root/pretrained_models/resnet-18}
dry_run=false
overrides=()
for arg in "$@"; do
  if [[ $arg == --dry-run ]]; then
    dry_run=true
  else
    case "$arg" in
      runner.logger.log_path=*|runner.logger.experiment_name=*|actor.model.hand_prior.artifact_path=*|algorithm.stage=*)
        echo 'Use the stage argument, OUTPUT_ROOT and PRIOR_ARTIFACT to preserve prior/DP artifact routing.' >&2
        exit 2 ;;
    esac
    overrides+=("$arg")
  fi
done

config_name() {
  local selected=$1 selected_stage=$2
  [[ $selected == lamplstm_film ]] && selected=lamplstm
  if [[ $selected_stage == prior && $selected != lamplstm_none && $selected != lamplstm_concat ]]; then
    printf 'ruiyan_lamp_prior_%s_stack_cube' "$selected"
  else
    printf 'ruiyan_lamp_%s_%s' "$selected_stage" "$selected"
  fi
}

run_stage() {
  local selected=$1 selected_stage=$2 config artifact
  config=$(config_name "$selected" "$selected_stage")
  local -a command=("$python_bin" examples/embodiment/train_lamp_il.py --config-name "$config"
    "$dataset_override" "runner.logger.log_path='$output_root'")
  if [[ $selected_stage == dp ]]; then
    command+=("actor.model.resnet_path='$resnet_path'")
    if [[ $selected != mlp ]]; then
      artifact=${PRIOR_ARTIFACT:-$output_root/$(config_name "$selected" prior)/artifact}
      command+=("actor.model.hand_prior.artifact_path='$artifact'")
    fi
  fi
  command+=("${overrides[@]}")
  printf 'Launching %s:\n' "$config"
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ $dry_run == false ]]; then
    "${command[@]}"
  fi
}

run_variant() {
  local selected=$1
  if [[ $selected != mlp && ( $stage == prior || $stage == both ) ]]; then
    run_stage "$selected" prior
  fi
  if [[ $stage == dp || $stage == both ]]; then
    run_stage "$selected" dp
  fi
}

if [[ $variant == all ]]; then
  for selected in lamplstm_film lamplstm_none lamplstm_concat pca vq mlp; do
    run_variant "$selected"
  done
else
  run_variant "$variant"
fi
