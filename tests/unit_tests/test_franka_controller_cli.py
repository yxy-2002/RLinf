# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free checks for the interactive Franka controller tool."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from toolkits.realworld_check import test_franka_controller as cli


@pytest.mark.parametrize(
    "options, expected",
    [
        (
            ["--end-effector-type", "wuji_hand", "--hand-serial-number", "TEST"],
            {"serial_number": "TEST", "side": "left", "namespace": "/wuji_hand/left"},
        ),
        (
            [
                "--end-effector-type",
                "wuji_hand",
                "--hand-serial-number",
                "TEST",
                "--hand-side",
                "right",
            ],
            {"serial_number": "TEST", "side": "right", "namespace": "/wuji_hand/right"},
        ),
        (
            [
                "--end-effector-type",
                "wuji_hand",
                "--hand-serial-number",
                "TEST",
                "--hand-namespace",
                "/custom/hand",
            ],
            {"serial_number": "TEST", "side": "left", "namespace": "/custom/hand"},
        ),
        (
            ["--end-effector-type", "ruiyan_hand", "--hand-port", "/dev/ttyUSB0"],
            {
                "port": "/dev/ttyUSB0",
                "baudrate": 460800,
                "motor_ids": (1, 2, 3, 4, 5, 6),
            },
        ),
        ([], {}),
    ],
)
def test_pose_commands(monkeypatch, capsys, options, expected):
    """Forward connection options, print both pose formats, and shut down."""
    monkeypatch.setattr(sys, "argv", ["test", "--robot-ip", "test-ip", *options])
    controller = MagicMock()
    controller.is_robot_up.return_value.wait.return_value = [True]
    controller.get_state.return_value.wait.return_value = [
        SimpleNamespace(tcp_pose=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
    ]
    factory = MagicMock()
    factory.launch_controller.return_value = controller
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.realworld.franka.franka_controller",
        SimpleNamespace(FrankaController=factory),
    )
    commands = iter(["getpos", "getpos_euler", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    cli.main()

    assert factory.launch_controller.call_args.kwargs["end_effector_config"] == expected
    assert factory.launch_controller.call_args.kwargs["robot_ip"] == "test-ip"
    output = capsys.readouterr().out
    assert "[1. 2. 3. 0. 0. 0. 1.]" in output
    assert "[1. 2. 3. 0. 0. 0.]" in output
    controller.shutdown.assert_called_once_with()
    controller.shutdown.return_value.wait.assert_called_once_with()


def test_wuji_requires_serial_number(monkeypatch, capsys):
    """Reject a missing hardware identity before launching any controller."""
    monkeypatch.setattr(sys, "argv", ["test", "--end-effector-type", "wuji_hand"])
    with pytest.raises(SystemExit, match="2"):
        cli._parse_args()
    assert "--hand-serial-number is required" in capsys.readouterr().err
