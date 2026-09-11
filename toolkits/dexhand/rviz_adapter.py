# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""ROS2 process boundary. Only this module imports rclpy."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from rlinf_dexhand.retargeting.kinematics import ASSETS
from rlinf_dexhand.types import HandSpec

from .transport import TargetReceiver


def wuji_spec(side):
    root = ET.parse(ASSETS / f"wuji/urdf/{side}-ros.urdf").getroot()
    joints = {
        j.get("name"): j for j in root.findall("joint") if j.get("type") == "revolute"
    }
    names = tuple(
        f"{side}_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)
    )
    # Models have their own exact naming convention; use shipped algorithm config.
    import yaml

    names = tuple(
        yaml.safe_load((ASSETS / f"wuji_{side}.yaml").read_text())["retargeting"][
            "target_joint_names"
        ]
    )
    return HandSpec(
        "wuji1hand",
        side,
        names,
        "rad",
        tuple(float(joints[n].find("limit").get("lower")) for n in names),
        tuple(float(joints[n].find("limit").get("upper")) for n in names),
    )


def robot_description(side):
    path = ASSETS / f"wuji/urdf/{side}-ros.urdf"
    root = ET.parse(path).getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith("package://"):
            filename = filename.split("/", 3)[3]
            mesh.set("filename", (ASSETS / "wuji" / filename).resolve().as_uri())
        elif not filename.startswith(("file://", "/")):
            mesh.set("filename", (path.parent / filename).resolve().as_uri())
    return ET.tostring(root, encoding="unicode")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=["left", "right"], default="left")
    p.add_argument("--bind", default="tcp://127.0.0.1:5557")
    p.add_argument("--no-rviz", action="store_true")
    args = p.parse_args()
    import os
    import signal
    import subprocess
    import tempfile

    import rclpy
    import zmq
    from sensor_msgs.msg import JointState

    spec = wuji_spec(args.side)
    rclpy.init()
    node = rclpy.create_node(f"wuji_teleop_adapter_{args.side}")
    publisher = node.create_publisher(JointState, f"/wuji/{args.side}/joint_states", 10)

    def publish(values):
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = list(spec.joint_names)
        msg.position = list(values)
        publisher.publish(msg)

    receiver = TargetReceiver(spec, publish)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    children = []
    try:
        sock.bind(args.bind)
        import yaml

        with tempfile.TemporaryDirectory(prefix="wuji-rviz-") as tmp:
            params = Path(tmp) / "rsp.yaml"
            params.write_text(
                yaml.safe_dump(
                    {
                        "/**": {
                            "ros__parameters": {
                                "robot_description": robot_description(args.side)
                            }
                        }
                    }
                )
            )
            children.append(
                subprocess.Popen(
                    [
                        "ros2",
                        "run",
                        "robot_state_publisher",
                        "robot_state_publisher",
                        "--ros-args",
                        "--params-file",
                        str(params),
                        "-r",
                        f"__ns:=/wuji/{args.side}",
                        "-r",
                        f"joint_states:=/wuji/{args.side}/joint_states",
                    ],
                    start_new_session=True,
                )
            )
            if not args.no_rviz:
                config = Path(tmp) / "display.rviz"
                config.write_text(
                    yaml.safe_dump(
                        {
                            "Visualization Manager": {
                                "Global Options": {
                                    "Fixed Frame": f"{args.side}_palm_link"
                                },
                                "Displays": [
                                    {
                                        "Class": "rviz_default_plugins/RobotModel",
                                        "Name": "Wuji",
                                        "Enabled": True,
                                        "Description Source": "Topic",
                                        "Description Topic": {
                                            "Value": f"/wuji/{args.side}/robot_description"
                                        },
                                    }
                                ],
                            }
                        }
                    )
                )
                children.append(
                    subprocess.Popen(
                        ["rviz2", "-d", str(config)], start_new_session=True
                    )
                )
            while rclpy.ok():
                if any(c.poll() is not None for c in children):
                    raise RuntimeError("Display child exited")
                rclpy.spin_once(node, timeout_sec=0)
                if sock.poll(10):
                    try:
                        reply = receiver.receive(sock.recv_json())
                    except ValueError as exc:
                        reply = {"ok": False, "error": str(exc)}
                    sock.send_json(reply)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(0)
        ctx.term()
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
