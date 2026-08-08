#!/usr/bin/env bash
# Local 2-GPU part: MLP plus CVAE/decoder-only at latent_dim=2.
# Retrains the required dim-2 CVAE prior in this repository.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PART=part1
export N_GPUS="${N_GPUS:-2}"
source "${SCRIPT_DIR}/lamp_water_plant_sweep_common.sh"
