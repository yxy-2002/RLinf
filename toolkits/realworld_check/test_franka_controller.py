# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Read Franka poses with an attached gripper, Ruiyan hand, or Wuji hand.

Wuji example (run in the configured Franka/ROS environment)::

    python -m toolkits.realworld_check.test_franka_controller \
        --robot-ip 172.17.0.2 --end-effector-type wuji_hand \
        --hand-serial-number 365C356C3333 --hand-side left

At the prompt, use ``getpos`` for [x, y, z, qx, qy, qz, qw],
``getpos_euler`` for [x, y, z, roll, pitch, yaw] (xyz Euler angles),
``getstate`` for the full state, or ``q`` to exit. Positions are meters;
angles are radians. The pose is Franka's O_T_EE, not a fingertip pose.

This starts the hand driver and arm impedance controller. Stop collection
or other controllers using the same hardware before running this script.
"""

import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Franka controller state.")
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("FRANKA_ROBOT_IP", None),
        help="Franka robot IP. Defaults to FRANKA_ROBOT_IP.",
    )
    parser.add_argument(
        "--end-effector-type",
        default="franka_gripper",
        choices=["franka_gripper", "robotiq_gripper", "ruiyan_hand", "wuji_hand"],
        help="Mounted end-effector type.",
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help="Serial port for Ruiyan hand, e.g. /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--hand-baudrate",
        type=int,
        default=460800,
        help="Serial baudrate for Ruiyan hand.",
    )
    parser.add_argument(
        "--hand-motor-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Motor IDs for Ruiyan hand.",
    )
    parser.add_argument(
        "--hand-serial-number",
        help="Wuji hand USB serial number (required for wuji_hand).",
    )
    parser.add_argument(
        "--hand-side",
        choices=["left", "right"],
        default="left",
        help="Wuji hand side (default: left).",
    )
    parser.add_argument(
        "--hand-namespace",
        help="Wuji ROS namespace (default: /wuji_hand/<hand-side>).",
    )
    args = parser.parse_args()
    if args.end_effector_type == "wuji_hand" and not args.hand_serial_number:
        parser.error("--hand-serial-number is required when using wuji_hand.")
    return args


def main() -> None:
    args = _parse_args()
    robot_ip = args.robot_ip
    assert robot_ip is not None, "Please set the FRANKA_ROBOT_IP environment variable."

    end_effector_config = {}
    if args.end_effector_type == "ruiyan_hand":
        if args.hand_port is None:
            raise ValueError("--hand-port is required when using ruiyan_hand.")
        end_effector_config = {
            "port": args.hand_port,
            "baudrate": args.hand_baudrate,
            "motor_ids": tuple(args.hand_motor_ids),
        }

    elif args.end_effector_type == "wuji_hand":
        end_effector_config = {
            "serial_number": args.hand_serial_number,
            "side": args.hand_side,
            "namespace": args.hand_namespace or f"/wuji_hand/{args.hand_side}",
        }

    from rlinf.envs.realworld.franka.franka_controller import FrankaController

    controller = FrankaController.launch_controller(
        robot_ip=robot_ip,
        end_effector_type=args.end_effector_type,
        end_effector_config=end_effector_config,
    )

    try:
        start_time = time.time()
        while not controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                print(
                    f"Waited {time.time() - start_time} seconds for Franka robot to be ready."
                )
        while True:
            try:
                cmd_str = input("Please input cmd:")
                if cmd_str == "q":
                    break
                elif cmd_str == "getpos":
                    print(controller.get_state().wait()[0].tcp_pose)
                elif cmd_str == "getpos_euler":
                    tcp_pose = controller.get_state().wait()[0].tcp_pose
                    r = R.from_quat(tcp_pose[3:].copy())
                    euler = r.as_euler("xyz")
                    print(np.concatenate([tcp_pose[:3], euler]))
                elif cmd_str == "getstate":
                    state = controller.get_state().wait()[0]
                    print(state.to_dict())
                elif cmd_str == "gethand":
                    print(controller.get_hand_detailed_state().wait()[0])
                else:
                    print(f"Unknown cmd: {cmd_str}")
            except (EOFError, KeyboardInterrupt):
                break
            time.sleep(1.0)

    finally:
        controller.shutdown().wait()


if __name__ == "__main__":
    main()
