#!/usr/bin/env bash
# Remote 2-GPU part: PCA latent_dim={2,4,6,8} plus fixed VQ.
# Retrains all four PCA priors and the VQ prior in this repository.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PART=part2
export N_GPUS="${N_GPUS:-2}"
source "${SCRIPT_DIR}/lamp_water_plant_sweep_common.sh"
