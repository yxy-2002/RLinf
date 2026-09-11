#!/usr/bin/env bash
# Install into the currently selected Python environment; no ROS installation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
case "${1:-core}" in
  core) "$PYTHON" -m pip install -e "$ROOT/third_party/rlinf-dexhand" ;;
  wuji)
    "$PYTHON" -m pip install -e "$ROOT/third_party/rlinf-dexhand" pin nlopt
    "$PYTHON" -m pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" ;;
  test) "$PYTHON" -m pip install -e "$ROOT/third_party/rlinf-dexhand[test]" gymnasium omegaconf pyzmq ;;
  *) echo 'Usage: install_dexhand.sh core|wuji|test' >&2; exit 2 ;;
esac
