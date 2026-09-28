#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""System-Python ROS1 CI smoke test. Always uses fake_hardware, never USB."""

import os
import signal
import socket
import subprocess
import time

import rosgraph
import rospy
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger


def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("ROS smoke test timed out")
        time.sleep(0.02)


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % port
    core = subprocess.Popen(["roscore", "-p", str(port)], start_new_session=True)
    driver = None
    try:
        until(rosgraph.is_master_online)
        rospy.init_node("wuji_ci", disable_signals=True)
        ns = "/wuji_ci_hand"
        for key, value in {
            "fake_hardware": True,
            "side": "left",
            "lower": [-1.0] * 20,
            "upper": [1.0] * 20,
        }.items():
            rospy.set_param(ns + "/" + key, value)
        driver = subprocess.Popen(
            [
                "rosrun",
                "wuji_hand_driver",
                "wuji_hand_driver_node",
                "__name:=wuji_ci_hand",
            ],
            start_new_session=True,
        )
        rospy.wait_for_service(ns + "/set_enabled", timeout=10)
        assert rospy.ServiceProxy(ns + "/set_enabled", SetBool)(True).success
        states, diagnostics = [], []
        state_sub = rospy.Subscriber(
            ns + "/joint_states", JointState, states.append, queue_size=1
        )
        diag_sub = rospy.Subscriber(
            ns + "/diagnostics", DiagnosticArray, diagnostics.append, queue_size=1
        )
        pub = rospy.Publisher(ns + "/joint_commands", JointState, queue_size=1)
        until(lambda: pub.get_num_connections())
        msg = JointState()
        msg.name = [
            "left_finger%d_joint%d" % (f, j) for f in range(1, 6) for j in range(1, 5)
        ]
        msg.position = [0.3] * 20
        for _ in range(20):
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)
            time.sleep(0.02)
        until(lambda: states and all(abs(q - 0.3) < 1e-6 for q in states[-1].position))

        def held():
            return (
                diagnostics
                and {p.key: p.value for p in diagnostics[-1].status[0].values}["held"]
                == "true"
            )

        until(held)
        assert rospy.ServiceProxy(ns + "/resume", Trigger)().success
        until(lambda: not held())
        # Partial joint messages must latch holding rather than zeroing other joints.
        msg.position = [0.2]
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        until(held)
        assert rospy.ServiceProxy(ns + "/set_enabled", SetBool)(False).success
        state_sub.unregister()
        diag_sub.unregister()
        assert core.poll() is None
    finally:
        for process in (driver, core):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    main()
