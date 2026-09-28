# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""ROS1 display of SDK input targets and measured Wuji joint positions."""

import argparse
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from rlinf_dexhand.wuji_spec import robot_description, wuji_spec


def display_config(side: str) -> dict:
    """Generate two RobotModels with distinct TF prefixes and descriptions."""
    return {
        "Visualization Manager": {
            "Global Options": {
                "Fixed Frame": "world",
                "Background Color": "48; 48; 48",
            },
            "Displays": [
                {
                    "Class": "rviz/RobotModel",
                    "Name": label,
                    "Enabled": True,
                    "Robot Description": f"/wuji_display/{side}/{kind}/robot_description",
                }
                for kind, label in (
                    ("target", "SDK input target"),
                    ("actual", "Measured position"),
                )
            ],
            "Views": {
                "Current": {
                    "Class": "rviz/Orbit",
                    "Distance": 0.7,
                    "Pitch": 0.5,
                    "Yaw": 0.7,
                }
            },
        }
    }


def main():
    import rospy
    import yaml
    from sensor_msgs.msg import JointState

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--namespace", default="/wuji_hand/left")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display retargeting only; no hardware feedback",
    )
    parser.add_argument("--no-rviz", action="store_true")
    args = parser.parse_args()
    spec = wuji_spec(args.side)
    rospy.init_node("wuji_display", anonymous=True, disable_signals=True)
    children, subscribers = [], []
    try:
        with tempfile.TemporaryDirectory(prefix="wuji-display-") as tmp:
            for kind in ("target",) if args.preview else ("target", "actual"):
                ns = f"/wuji_display/{args.side}/{kind}"
                prefix = f"{kind}_"
                description = robot_description(args.side, prefix)
                rospy.set_param(ns + "/robot_description", description)
                root = ET.fromstring(description)
                child_links = {c.get("link") for c in root.iter("child")}
                base = next(
                    n.get("name")
                    for n in root.findall("link")
                    if n.get("name") not in child_links
                )
                children.append(
                    subprocess.Popen(
                        [
                            "rosrun",
                            "robot_state_publisher",
                            "robot_state_publisher",
                            f"__ns:={ns}",
                        ]
                    )
                )
                offset = "-0.15" if kind == "target" else "0.15"
                children.append(
                    subprocess.Popen(
                        [
                            "rosrun",
                            "tf2_ros",
                            "static_transform_publisher",
                            offset,
                            "0",
                            "0",
                            "0",
                            "0",
                            "0",
                            "world",
                            base,
                        ]
                    )
                )
                pub = rospy.Publisher(ns + "/joint_states", JointState, queue_size=1)

                def relay(msg, publisher=pub, joint_prefix=prefix):
                    if tuple(msg.name) != spec.joint_names or len(msg.position) != 20:
                        return
                    result = JointState()
                    result.header = msg.header
                    result.name = [joint_prefix + n for n in msg.name]
                    result.position = msg.position
                    publisher.publish(result)

                topic = (
                    f"/wuji_preview/{args.side}/joint_targets"
                    if args.preview
                    else args.namespace.rstrip("/")
                    + ("/joint_targets" if kind == "target" else "/joint_states")
                )
                subscribers.append(
                    rospy.Subscriber(topic, JointState, relay, queue_size=1)
                )
            if not args.no_rviz:
                config = display_config(args.side)
                if args.preview:
                    config["Visualization Manager"]["Displays"] = config[
                        "Visualization Manager"
                    ]["Displays"][:1]
                    config["Visualization Manager"]["Displays"][0]["Name"] = (
                        "Retargeting preview (no hardware)"
                    )
                path = Path(tmp) / "wuji.rviz"
                path.write_text(yaml.safe_dump(config))
                children.append(
                    subprocess.Popen(["rosrun", "rviz", "rviz", "-d", str(path)])
                )
            while not rospy.is_shutdown():
                if any(child.poll() is not None for child in children):
                    break
                rospy.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for sub in subscribers:
            sub.unregister()
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    main()
