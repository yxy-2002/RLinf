# Wuji ROS1 driver

ROS1 port of the hardware portion of Wuji Robotics' Apache-2.0 `wujihandros2/wujihand_driver`, using the supplied `wujihandcpp 1.5.1` headers/library. The Catmull–Rom equations follow `psi_glove_ros2/scripts/wujihand_spline_forwarder.py`; no ROS2 runtime is used. The external reference workspace is not a runtime dependency.

The RLinf `WujiHand` adapter supplies the exact joint names/limits from its installed model, sets private parameters and owns this process. Physical startup requires an explicit USB serial and checks handedness; motors remain disabled until the adapter requests enable after feedback arrives.

Private topics: `joint_commands` (20 named radians), `joint_states` (measured radians), `joint_targets` (interpolated SDK input), `diagnostics`. Services: `set_enabled` (SetBool), `hold`, `resume`, `clear_trajectory`, `reset_error` (Trigger). Error reset applies to the entire hand and does not automatically resume. Malformed commands latch holding. Driver-level resume needs enabled motors, valid hardware reads and no motor errors. Glove frame loss is handled by GloveExpert using its last valid target; it does not pause collection or require a resume service.

The default single ROS event loop performs bounded 50 ms health reads at 10 Hz; these can delay interpolation callbacks. The SDK itself runs its control loop separately. Configured callback rates are best-effort, not hard-real-time rates. `joint_states` timestamps reflect the most recent explicit hardware read, not the time of republication. Diagnostics exposes its age. A hardware read error latches a fatal fault requiring driver restart; clearing motor errors does not hide loss of USB communication.

For hardware-free builds, set `-DWUJI_WITH_SDK=OFF`; the resulting binary requires `fake_hardware:=true`. That backend follows the interpolated target and does not emulate motor dynamics or USB timing. Full startup and test commands are in `toolkits/dexhand/README.md`.
