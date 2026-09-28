#!/usr/bin/env bash
# Install into the currently selected Python environment; no ROS installation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
case "${1:-core}" in
  core) "$PYTHON" -m pip install -e "$ROOT/third_party/rlinf-dexhand" ;;
  wuji)
    if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
      "$PYTHON" -m pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
    fi
    "$PYTHON" -m pip install -e "$ROOT/third_party/rlinf-dexhand[wuji]" ;;
  wuji-ros1) exec bash "$ROOT/requirements/install_wuji_ros1.sh" ;;
  test) "$PYTHON" -m pip install -e "$ROOT/third_party/rlinf-dexhand[test]" gymnasium omegaconf ;;
  *) echo 'Usage: install_dexhand.sh core|wuji|wuji-ros1|test' >&2; exit 2 ;;
esac
