# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in ROS1 tests using the real driver with its simulated hardware backend."""

import os
import signal
import socket
import subprocess
import time
import uuid

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RLINF_TEST_WUJI_ROS1") != "1",
    reason="Requires built wuji_hand_driver and ROS1; no physical hardware",
)


@pytest.fixture(scope="module")
def ros():
    import rosgraph

    from rlinf.envs.realworld.common.ros import ROSController

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    old_uri = os.environ.get("ROS_MASTER_URI")
    os.environ["ROS_MASTER_URI"] = f"http://127.0.0.1:{port}"
    core = subprocess.Popen(
        ["roscore", "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not rosgraph.is_master_online():
            if time.monotonic() > deadline:
                pytest.fail("Isolated test ROS master did not start")
            time.sleep(0.05)
        yield ROSController()
        assert core.poll() is None, "Hand cleanup killed the shared ROS master"
    finally:
        os.killpg(core.pid, signal.SIGINT)
        core.wait(timeout=10)
        if old_uri is None:
            os.environ.pop("ROS_MASTER_URI", None)
        else:
            os.environ["ROS_MASTER_URI"] = old_uri


def test_fake_driver_motion_fault_resume_and_shutdown(ros):
    import rospy
    from rlinf_dexhand.wuji_spec import to_radians
    from std_srvs.srv import Trigger

    from rlinf.envs.realworld.common.hand.wuji_hand import WujiHand

    ns = "/wuji_test/n" + uuid.uuid4().hex
    hand = WujiHand(
        ros,
        serial_number="fake-" + uuid.uuid4().hex,
        namespace=ns,
        fake_hardware=True,
        reset_duration=0.15,
    )
    try:
        hand.initialize()
        hand.get_state()
        for _ in range(25):
            hand.command(np.full(20, 0.4))
            time.sleep(0.02)
        np.testing.assert_allclose(
            hand.get_state(), to_radians(hand.spec, np.full(20, 0.4)), atol=0.02
        )
        time.sleep(0.7)
        assert hand.get_detailed_state()["held"] == "true"
        assert rospy.ServiceProxy(ns + "/resume", Trigger)().success
        time.sleep(0.05)
        assert hand.get_detailed_state()["held"] == "false"
        hand.reset(np.full(20, 0.5))
        np.testing.assert_allclose(
            hand.get_state(), to_radians(hand.spec, np.full(20, 0.5)), atol=0.05
        )
        process = hand._process
    finally:
        hand.shutdown()
        hand.shutdown()
    assert process.poll() is not None


def test_driver_exit_is_not_reported_as_valid_feedback(ros):
    from rlinf.envs.realworld.common.hand.wuji_hand import WujiHand

    hand = WujiHand(
        ros,
        serial_number="fake-" + uuid.uuid4().hex,
        namespace="/wuji_test/n" + uuid.uuid4().hex,
        fake_hardware=True,
    )
    try:
        hand.initialize()
        os.killpg(hand._process.pid, signal.SIGINT)
        hand._process.wait(timeout=5)
        with pytest.raises(RuntimeError, match="driver exited"):
            hand.get_state()
    finally:
        hand.shutdown()


def test_namespace_collision_preserves_existing_owner(ros):
    from filelock import Timeout

    from rlinf.envs.realworld.common.hand.wuji_hand import WujiHand

    ns = "/wuji_test/n" + uuid.uuid4().hex
    hands = [
        WujiHand(ros, serial_number=uuid.uuid4().hex, namespace=ns, fake_hardware=True)
        for _ in range(2)
    ]
    try:
        hands[0].initialize()
        with pytest.raises(Timeout):
            hands[1].initialize()
        hands[1].shutdown()
        for _ in range(10):
            hands[0].command(np.full(20, 0.4))
            time.sleep(0.02)
        assert hands[0].get_detailed_state()["feedback_valid"]
        assert hands[0]._process.poll() is None
    finally:
        for hand in reversed(hands):
            hand.shutdown()


def test_headless_display_publishes_disjoint_target_and_actual_frames(ros):
    import sys

    import rospy
    from rlinf_dexhand.wuji_spec import wuji_spec
    from sensor_msgs.msg import JointState
    from tf2_msgs.msg import TFMessage

    for package in ("robot_state_publisher", "tf2_ros"):
        if subprocess.run(["rospack", "find", package], capture_output=True).returncode:
            pytest.skip(f"Display package unavailable: {package}")
    ns = "/wuji_display_test/n" + uuid.uuid4().hex
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "toolkits.dexhand.rviz_adapter",
            "--no-rviz",
            "--namespace",
            ns,
        ],
        start_new_session=True,
    )
    names = set()
    sub = rospy.Subscriber(
        "/tf",
        TFMessage,
        lambda msg: names.update(t.child_frame_id for t in msg.transforms),
    )
    publishers = [
        rospy.Publisher(ns + suffix, JointState, queue_size=1)
        for suffix in ("/joint_targets", "/joint_states")
    ]
    try:
        deadline = time.monotonic() + 10
        while not all(p.get_num_connections() for p in publishers):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        spec = wuji_spec("left")
        message = JointState()
        message.name = list(spec.joint_names)
        message.position = list(spec.lower)
        while not {"target_left_finger1_link1", "actual_left_finger1_link1"}.issubset(
            names
        ):
            assert time.monotonic() < deadline
            message.header.stamp = rospy.Time.now()
            for pub in publishers:
                pub.publish(message)
            time.sleep(0.05)
    finally:
        sub.unregister()
        for pub in publishers:
            pub.unregister()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def test_background_read_isolation_and_feedback_timeout(ros, tmp_path):
    """Exercise the real driver with blocked fake reads; never connect to USB."""
    import shlex
    import shutil
    from pathlib import Path

    if not shutil.which("g++"):
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[2]
    driver = root / "third_party/rlinf-dexhand/ros/wuji_hand_driver"
    source = tmp_path / "background_driver.cpp"
    source.write_text(r"""
#define main driver_entrypoint
#include "driver.cpp"
#undef main
#include <cassert>
#include <thread>
int main(int argc, char** argv) {
  ros::init(argc, argv, "background_driver_test",
            ros::init_options::AnonymousName);
  ros::NodeHandle nh("~");
  nh.setParam("fake_hardware", true);
  nh.setParam("lower", std::vector<double>(20, -1.0));
  nh.setParam("upper", std::vector<double>(20, 1.0));
  Driver driver(nh);
  driver.enabled_ = true;
  sensor_msgs::JointState::Ptr command(new sensor_msgs::JointState);
  command->name = driver.names_;
  command->position = std::vector<double>(20, 0.0);
  command->header.stamp = ros::Time::now();
  std_srvs::SetBool::Request ownership;
  std_srvs::SetBool::Response owned;
  ownership.data = true;
  driver.set_teleop(ownership, owned);
  assert(owned.success);
  driver.command(command);
  assert(!driver.armed_);
  driver.teleop_command(command);
  assert(driver.armed_);
  ownership.data = false;
  driver.set_teleop(ownership, owned);
  driver.teleop_command(command);  // Late in-flight message after pause ACK.
  assert(!driver.armed_);
  driver.command(command);
  assert(driver.armed_);
  // The worker blocks inside read_hardware while the main thread keeps ticking.
  driver.hardware_mutex_.lock();
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  const double original_feedback = driver.last_health_;
  ros::WallTimerEvent event;
  driver.held_ = true;  // Exercise output with a known target, without interpolation.
  for (int i = 0; i < 10; ++i) {
    driver.target_.fill(0.01 * (i + 1));
    driver.tick(event);
    assert(!driver.fatal_);
    assert(driver.actual_ == driver.target_);
  }
  const auto previous_actual = driver.actual_;
  assert(driver.last_health_ == original_feedback);
  // A stuck reader must not allow indefinite output using stale feedback.
  driver.last_health_ = now() - driver.timeout_ - 1;
  driver.target_.fill(0.5);
  driver.tick(event);
  assert(driver.fatal_ && driver.held_ && !driver.armed_);
  assert(driver.actual_ == previous_actual);
  assert(driver.reason_ == "Hardware feedback timeout");
  driver.hardware_mutex_.unlock();
  driver.health_worker_.stop();  // Wait for the blocked read to finish.
  // A later successful read must not silently clear the fault or resume output.
  driver.health();
  assert(driver.fatal_ && driver.held_);
  std_srvs::Trigger::Request req;
  std_srvs::Trigger::Response res;
  driver.resume(req, res);
  assert(!res.success);
  std_srvs::SetBool::Request enable_req;
  std_srvs::SetBool::Response enable_res;
  enable_req.data = true;
  driver.enable(enable_req, enable_res);
  assert(!enable_res.success);
}
""")
    flags = subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "roscpp"], text=True
    )
    binary = tmp_path / "background_driver"
    # Test access only; compile the actual source without WUJI_WITH_SDK.
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-pthread",
            "-fno-access-control",
            "-I",
            str(driver / "include"),
            "-I",
            str(driver / "src"),
            str(source),
            "-o",
            str(binary),
            *shlex.split(flags),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, timeout=15)


class _SyntheticGlove:
    def __init__(self, **kwargs):
        self.sequence = 0

    def get_target(self):
        from types import SimpleNamespace

        from rlinf_dexhand.wuji_spec import to_radians, wuji_spec

        self.sequence += 1
        spec = wuji_spec("left")
        return SimpleNamespace(
            spec=spec,
            timestamp=time.time(),
            sequence=self.sequence,
            values=to_radians(spec, np.full(20, 0.4)),
        )

    def close(self):
        pass


class _SyntheticMouse:
    def __init__(self):
        self.started = time.monotonic()

    def get_action(self):
        return np.zeros(6), [False, time.monotonic() - self.started > 0.5]

    def close(self):
        pass


def _synthetic_inputs_worker(connection, shared, config):
    import rlinf_dexhand.glove

    from rlinf.envs.realworld.common.spacemouse import spacemouse_expert
    from rlinf.envs.realworld.common.wrappers.dexhand_intervention import _teleop_worker

    rlinf_dexhand.glove.GloveExpert = _SyntheticGlove
    spacemouse_expert.SpaceMouseExpert = _SyntheticMouse
    _teleop_worker(connection, shared, config)


def test_real_teleop_process_with_simulated_inputs_and_driver(ros):
    from rlinf_dexhand.wuji_spec import to_radians

    from rlinf.envs.realworld.common.hand.wuji_hand import WujiHand
    from rlinf.envs.realworld.common.wrappers.dexhand_intervention import TeleopProcess

    params = {
        "serial_number": "fake-" + uuid.uuid4().hex,
        "namespace": "/wuji_process_test/n" + uuid.uuid4().hex,
        "fake_hardware": True,
        "reset_duration": 0.15,
    }
    hand = WujiHand(ros, **params)
    child = None
    try:
        hand.initialize()
        child = TeleopProcess(
            {
                "hand": params,
                "hand_type": "wuji_hand",
                "hand_dim": 20,
                "right_button_labels_only": True,
                "frequency": 60,
                "glove_frequency": 60,
                "pipeline_config": "synthetic",
                "scale_file": None,
                "mode": "absolute",
                "timeout": 0.5,
                "max_delta": float("inf"),
            },
            worker=_synthetic_inputs_worker,
        )
        child.request("resume")
        time.sleep(0.2)
        first = child.snapshot()[2]
        time.sleep(0.2)
        assert child.snapshot()[2] - first >= 5
        time.sleep(0.5)
        np.testing.assert_allclose(
            hand.get_state(), to_radians(hand.spec, np.full(20, 0.4)), atol=0.04
        )
        # Normal env commands must be ignored while the child owns output.
        hand.command(np.full(20, 0.7))
        time.sleep(0.1)
        np.testing.assert_allclose(
            hand.get_state(), to_radians(hand.spec, np.full(20, 0.4)), atol=0.04
        )
        child.request("pause")
        hand.reset(np.full(20, 0.3))
        np.testing.assert_allclose(
            hand.get_state(), to_radians(hand.spec, np.full(20, 0.3)), atol=0.05
        )
        child.request("resume")
        time.sleep(0.2)
        # Button remained held across reset: require release before reacquiring.
        np.testing.assert_allclose(
            hand.get_state(), to_radians(hand.spec, np.full(20, 0.3)), atol=0.05
        )
    finally:
        if child is not None:
            child.close()
        hand.shutdown()
