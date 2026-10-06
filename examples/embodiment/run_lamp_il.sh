#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:?usage: run_lamp_il.sh <config-name> [hydra overrides...]}"
shift
python3 examples/embodiment/train_lamp_il.py --config-name "${CONFIG_NAME}" "$@"
