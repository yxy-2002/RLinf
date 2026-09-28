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
