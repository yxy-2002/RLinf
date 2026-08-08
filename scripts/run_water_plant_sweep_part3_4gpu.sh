#!/usr/bin/env bash
# Remote 4-GPU part: CVAE/decoder-only at latent_dim={4,6,8}.
# Retrains all three required CVAE priors in this repository.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PART=part3
export N_GPUS="${N_GPUS:-4}"
source "${SCRIPT_DIR}/lamp_water_plant_sweep_common.sh"
