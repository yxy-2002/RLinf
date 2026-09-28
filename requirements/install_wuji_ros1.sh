#!/usr/bin/env bash
# Called by install.sh dexhand wuji-ros1. Uses the active Franka environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WUJIHANDCPP_DEB:?Set WUJIHANDCPP_DEB to the official 1.5.1 amd64 deb}"
: "${VIRTUAL_ENV:?Activate the Franka virtual environment first}"
# Keep package installation and catkin on the same activated interpreter.
export PYTHON="$VIRTUAL_ENV/bin/python"
EXPECTED_SHA=d3cfeac37ea2dddfd5b7c9ca78e605c2ab98227781784fa0b18a6b1896872c12
[ "$(uname -m)" = x86_64 ] || { echo 'SDK 1.5.1 package requires x86_64' >&2; exit 1; }
[ "$(dpkg-deb -f "$WUJIHANDCPP_DEB" Package)" = wujihandcpp ]
[ "$(dpkg-deb -f "$WUJIHANDCPP_DEB" Version)" = 1.5.1 ]
[ "$(dpkg-deb -f "$WUJIHANDCPP_DEB" Architecture)" = amd64 ]
[ "$(sha256sum "$WUJIHANDCPP_DEB" | cut -d ' ' -f 1)" = "$EXPECTED_SHA" ] || { echo 'SDK SHA256 mismatch' >&2; exit 1; }
SDK_DIR="$VIRTUAL_ENV/wujihandcpp-1.5.1"
mkdir -p "$SDK_DIR"
dpkg-deb -x "$WUJIHANDCPP_DEB" "$SDK_DIR"
export WUJI_LIBRARY_PATH="$SDK_DIR/usr/lib/libwujihandcpp.so"
"$PYTHON" - <<'PY'
import ctypes, os
ctypes.CDLL(os.environ['WUJI_LIBRARY_PATH'])
PY
bash "$ROOT/requirements/install_dexhand.sh" wuji
WUJI_EMPY_SCRIPT="$("$PYTHON" -c 'import em; print(em.__file__)')"
# Reuse the configured Franka workspace; do not build a second ROS distribution.
WUJI_CATKIN_PATH="${ROS_CATKIN_PATH:-$VIRTUAL_ENV/franka_catkin_ws}"
mkdir -p "$WUJI_CATKIN_PATH/src"
if [ -e "$WUJI_CATKIN_PATH/src/wuji_hand_driver" ] && [ ! -L "$WUJI_CATKIN_PATH/src/wuji_hand_driver" ]; then
    echo 'Refusing to replace an existing wuji_hand_driver source directory' >&2
    exit 1
fi
ln -sfn "$ROOT/third_party/rlinf-dexhand/ros/wuji_hand_driver" "$WUJI_CATKIN_PATH/src/wuji_hand_driver"
set +u
source /opt/ros/noetic/setup.bash
set -u
(cd "$WUJI_CATKIN_PATH" && catkin_make --pkg wuji_hand_driver \
    -DPYTHON_EXECUTABLE="$PYTHON" -DEMPY_SCRIPT="$WUJI_EMPY_SCRIPT" \
    -DWUJI_WITH_SDK=ON -DWUJI_INCLUDE="$SDK_DIR/usr/include" -DWUJI_LIBRARY="$WUJI_LIBRARY_PATH")
activation="source '$WUJI_CATKIN_PATH/devel/setup.bash'"
if ! grep -Fqx "$activation" "$VIRTUAL_ENV/bin/activate"; then
    printf '\n%s\n' "$activation" >> "$VIRTUAL_ENV/bin/activate"
fi
printf 'Wuji installed. Reactivate the environment, or source %s/devel/setup.bash\n' "$WUJI_CATKIN_PATH"
