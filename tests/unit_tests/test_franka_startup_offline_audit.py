# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Offline fault injection for the October 2026 Franka startup investigation.

Execute selected, unmodified production methods through AST extraction. Never
import the real-world package, ROS, Ray, or hardware drivers. All process launch,
ROS communication, controller RPC, and time dependencies are test doubles.

Strict xfails document unresolved safety expectations, not successful safeguards.
Use --runxfail to expose their assertion failures. These tests do not model robot
dynamics and cannot certify a trajectory or establish the incident's root cause.
"""

import ast
import os
import sys
import xmlrpc.client
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
FRANKA = ROOT / "rlinf/envs/realworld/franka"
ROS = ROOT / "rlinf/envs/realworld/common/ros/ros_controller.py"


def _load(path: Path, names: list[str], scope: dict, owner: str | None = None):
    """Compile actual source bodies without executing package imports."""
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return {name: scope[name] for name in names}


def _pose(x: float = 0.5) -> np.ndarray:
    return np.array([x, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0])


def _env(frequency: float = 10, initial: np.ndarray | None = None):
    clock = Mock()
    scope = {"np": np, "time": clock}
    _load(FRANKA / "utils.py", ["normalize", "quat_slerp"], scope)
    methods = _load(
        FRANKA / "franka_env.py",
        ["_interpolate_move", "_move_action", "_clear_error"],
        scope,
        "FrankaEnv",
    )
    env = type("OfflineFrankaEnv", (), methods)()
    env.config = SimpleNamespace(step_frequency=frequency, is_dummy=False)
    controller = Mock()
    state = SimpleNamespace(tcp_pose=_pose() if initial is None else initial.copy())
    controller.get_state.return_value.wait.return_value = [state]
    env._controller = controller
    # An actual workspace guard would reject these test targets if called.
    env._clip_position_to_safety_box = Mock(
        side_effect=RuntimeError("outside workspace")
    )
    return env, controller, clock


def _commands(controller: Mock) -> np.ndarray:
    return np.array([call.args[0] for call in controller.move_arm.call_args_list])


@pytest.mark.parametrize("frequency, count", [(10, 15), (100, 150)])
def test_interpolation_frequency_and_valid_quaternions(frequency, count):
    """Measure setpoint spacing; this does not measure actual robot speed."""
    env, controller, clock = _env(frequency)
    target = _pose(0.65)
    target[3:] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    env._interpolate_move(target)
    commands = _commands(controller)
    assert len(commands) == count
    np.testing.assert_allclose(commands[-1], target, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(commands[:, 3:], axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(
        np.diff(np.r_[0.5, commands[:, 0]]), 0.15 / count, atol=1e-7
    )
    assert sum(call.args[0] for call in clock.sleep.call_args_list) == pytest.approx(
        1.5
    )


def test_zero_quaternion_blocks_interpolation():
    """The all-zero default state raises before any arm command is sent."""
    env, controller, _ = _env(initial=np.zeros(7))
    with pytest.raises(ValueError, match="Zero-norm quaternion"):
        env._interpolate_move(_pose())
    controller.move_arm.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="TODO(agent): startup bypasses workspace validation",
)
def test_startup_rejects_outside_workspace():
    env, controller, _ = _env()
    # Synthetic invalid target; no physical workspace or robot is involved.
    try:
        env._interpolate_move(_pose(5.0))
    except (ValueError, RuntimeError):
        pass
    assert controller.move_arm.call_count == 0, "Outside-workspace target was sent"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="TODO(agent): no feedback/fault stop during interpolation",
)
def test_stalled_faulted_robot_stops_receiving_commands():
    env, controller, _ = _env()
    state = controller.get_state.return_value.wait.return_value[0]
    state.robot_mode = "reflex"
    state.current_errors = {"joint_velocity_violation": True}
    # Feedback remains fixed even as the requested pose moves away.
    env._interpolate_move(_pose(0.65))
    assert controller.move_arm.call_count == 0, (
        "All 15 setpoints sent despite faulted feedback"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="TODO(agent): each setpoint automatically requests error recovery",
)
def test_interpolation_does_not_automatically_clear_faults():
    env, controller, _ = _env()
    env._interpolate_move(_pose(0.65))
    assert controller.clear_errors.call_count == 0, "Error recovery requested 15 times"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="TODO(agent): nonfinite position is not rejected before publication",
)
def test_nan_position_never_reaches_controller():
    env, controller, _ = _env()
    target = _pose()
    target[0] = np.nan
    try:
        env._interpolate_move(target)
    except (ValueError, RuntimeError):
        pass
    assert controller.move_arm.call_count == 0, "NaN positions were sent"


def _ros(process_names: list[str], *, online=True, initialize=True):
    processes = []
    for name in process_names:
        process = Mock()
        process.name.return_value = name
        processes.append(process)
    psutil = Mock()
    psutil.process_iter.return_value = processes
    psutil.Popen.return_value.poll.return_value = 1
    rospy = Mock()
    clock = Mock()
    clock.monotonic.return_value = 0.0
    scope = {
        "os": os,
        "sys": sys,
        "pathlib": SimpleNamespace(Path=Path),
        "FileLock": lambda _: nullcontext(),
        "psutil": psutil,
        "rospy": rospy,
        "rosgraph": SimpleNamespace(get_master_uri=lambda: "http://localhost:11311"),
        "_is_ros_master_online": Mock(return_value=online),
        "urlsplit": urlsplit,
        "time": clock,
        "get_logger": Mock(),
    }
    methods = _load(
        ROS,
        [
            "__init__",
            "_ensure_ros_master",
            "connect_ros_channel",
            "get_input_channel_status",
        ],
        scope,
        "ROSController",
    )
    cls = type("OfflineROSController", (), methods)
    ros = object.__new__(cls)
    ros._probe = scope["_is_ros_master_online"]
    if initialize:
        ros.__init__()
    return ros, psutil, rospy, clock


def test_existing_roscore_name_is_reused():
    _, psutil, _, _ = _ros(["roscore", "rosmaster"])
    psutil.Popen.assert_not_called()


def test_existing_master_without_roscore_does_not_launch_duplicate():
    _, psutil, _, _ = _ros(["rosmaster", "rosout"])
    assert psutil.Popen.call_count == 0, "A second roscore was launched"


def test_failed_roscore_launch_is_detected():
    ros, psutil, rospy, _ = _ros([], online=False, initialize=False)
    with pytest.raises(RuntimeError, match="roscore exited with code 1"):
        ros.__init__()
    rospy.init_node.assert_not_called()
    psutil.Popen.assert_called_once()


def test_roscore_waits_for_master_before_initializing_node():
    ros, psutil, rospy, clock = _ros([], initialize=False)
    ros._probe.side_effect = [False, False, True]
    psutil.Popen.return_value.poll.return_value = None
    ros.__init__()
    clock.sleep.assert_called_once_with(0.1)
    rospy.init_node.assert_called_once()


def test_unresponsive_roscore_name_does_not_bypass_master_check():
    ros, _, rospy, _ = _ros(["roscore"], online=False, initialize=False)
    with pytest.raises(RuntimeError, match="ROS master .* is unavailable"):
        ros.__init__()
    rospy.init_node.assert_not_called()


def test_roscore_readiness_timeout_cleans_up_owned_process():
    ros, psutil, rospy, clock = _ros([], online=False, initialize=False)
    psutil.Popen.return_value.poll.return_value = None
    clock.monotonic.side_effect = [0.0, 10.0]
    with pytest.raises(RuntimeError, match="did not become ready within 10s"):
        ros.__init__()
    psutil.Popen.return_value.terminate.assert_called_once()
    psutil.Popen.return_value.wait.assert_called_once_with(timeout=5)
    rospy.init_node.assert_not_called()


@pytest.mark.parametrize(
    "response, error, expected",
    [
        ((1, "", 123), None, True),
        ((-1, "failed", 0), None, False),
        (None, ConnectionRefusedError(), False),
        (None, TimeoutError(), False),
        (None, xmlrpc.client.Fault(1, "failed"), False),
    ],
)
def test_master_probe_validates_xmlrpc_response(response, error, expected):
    proxy = Mock()
    master = proxy.return_value.__enter__ = Mock(return_value=Mock())
    proxy.return_value.__exit__ = Mock(return_value=False)
    master.return_value.getPid.return_value = response
    master.return_value.getPid.side_effect = error
    scope = {
        "xmlrpc": SimpleNamespace(
            client=SimpleNamespace(ServerProxy=proxy, Error=xmlrpc.client.Error)
        ),
        "_MasterTransport": Mock(),
    }
    probe = _load(ROS, ["_is_ros_master_online"], scope)["_is_ros_master_online"]
    assert probe("http://localhost:11311") is expected
    master.return_value.getPid.assert_called_once_with("/rlinf_master_probe")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="TODO(agent): one old state sample leaves readiness latched forever",
)
def test_stale_state_is_not_ready():
    ros, _, rospy, clock = _ros(["roscore"])
    ros.connect_ros_channel("state", object, Mock())
    callback = rospy.Subscriber.call_args.args[2]
    callback(SimpleNamespace(header=SimpleNamespace(stamp=0)))
    clock.monotonic.return_value = 3600
    clock.time.return_value = 3600
    scope = {}
    methods = _load(
        FRANKA / "franka_controller.py", ["is_robot_up"], scope, "FrankaController"
    )
    controller = type("OfflineReadiness", (), methods)()
    controller._ros = ros
    controller._arm_state_channel = "state"
    controller._end_effector_type = SimpleNamespace(is_gripper=False)
    assert not controller.is_robot_up(), "Stale sample still satisfies robot readiness"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="TODO(agent): impedance launch has no duplicate-controller preflight",
)
def test_existing_impedance_controller_is_not_launched_again():
    psutil = Mock()
    existing = Mock()
    existing.name.return_value = "roslaunch"
    existing.cmdline.return_value = [
        "roslaunch",
        "serl_franka_controllers",
        "impedance.launch",
    ]
    psutil.process_iter.return_value = [existing]
    scope = {
        "psutil": psutil,
        "sys": sys,
        "EndEffectorType": SimpleNamespace(FRANKA_GRIPPER="franka_gripper"),
    }
    methods = _load(
        FRANKA / "franka_controller.py", ["start_impedance"], scope, "FrankaController"
    )
    controller = type("OfflineImpedance", (), methods)()
    controller._end_effector_type = "ruiyan_hand"
    controller._ros_pkg = "serl_franka_controllers"
    controller._robot_ip = "offline-test-only"
    controller._wait_robot = Mock()
    controller.log_debug = Mock()
    controller.start_impedance()
    assert psutil.Popen.call_count == 0, "Duplicate impedance.launch was requested"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="TODO(agent): configured compliance is not applied before initial move",
)
def test_constructor_configures_compliance_before_motion():
    events = []
    cfg = SimpleNamespace(
        end_effector_type="ruiyan_hand",
        gripper_type=None,
        task_description="offline",
        is_dummy=False,
        reset_ee_pose=np.array([0.5, 0.1, 0.4, 0, 0, 0]),
        joint_reset_cycle=20000,
        enable_camera_player=False,
        compliance_param={"translational_stiffness": 1000},
    )
    controller = Mock()
    controller.is_robot_up.return_value.wait.return_value = [True]
    controller.get_state.return_value.wait.return_value = [
        SimpleNamespace(tcp_pose=_pose())
    ]

    def configure(*_):
        events.append("configure")
        return Mock()

    controller.reconfigure_compliance_params.side_effect = configure
    scope = {
        "np": np,
        "R": Rotation,
        "time": Mock(),
        "cycle": __import__("itertools").cycle,
        "get_logger": Mock(),
        "FrankaRobotState": Mock(),
        "normalize_end_effector_type": lambda *_: SimpleNamespace(value="ruiyan_hand"),
        "EndEffectorType": SimpleNamespace(WUJI_HAND="wuji_hand"),
        "VideoPlayer": Mock(),
    }
    methods = _load(FRANKA / "franka_env.py", ["__init__"], scope, "FrankaEnv")
    cls = type("OfflineConstructor", (), methods)
    env = object.__new__(cls)
    env.CONFIG_CLS = lambda **_: cfg
    env._ee_type = "ruiyan_hand"
    env._controller = controller
    env._setup_hardware = Mock()
    env._setup_reward_worker = Mock()
    env._build_camera_infos = Mock(return_value=["offline-camera"])
    env._init_action_obs_spaces = Mock()
    env._interpolate_move = Mock(side_effect=lambda *_: events.append("move"))
    env._open_cameras = Mock()
    env.close = Mock()
    cls.__init__(env, {}, None, None, 0)
    assert events == ["configure", "move"], f"Startup sequence was {events}"
