#!/usr/bin/env bash
# Local 2-GPU part: CVAE KL selected/default (z=2) + decoder_only DP lr sweep + MLP.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PART=part1
export N_GPUS="${N_GPUS:-2}"
source "${SCRIPT_DIR}/lamp_water_plant_sweep_common.sh"
