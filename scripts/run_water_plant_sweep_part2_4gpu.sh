#!/usr/bin/env bash
# Remote 4-GPU part: CVAE KL loose (z=2) + PCA/VQ baselines + decoder_only LR sweep.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PART=part2
export N_GPUS="${N_GPUS:-4}"
source "${SCRIPT_DIR}/lamp_water_plant_sweep_common.sh"
