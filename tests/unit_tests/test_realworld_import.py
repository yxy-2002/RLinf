# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""RealWorld imports must preserve shared ROS services in fresh interpreters."""

import importlib
import multiprocessing
from unittest.mock import Mock, patch

import pytest


def _import_in_child(sender) -> None:
    """Import the real HID entrypoint with harmless stand-ins for ROS processes."""
    processes = [Mock() for _ in range(3)]
    for process, name in zip(processes, ("roscore", "rosmaster", "rosout")):
        process.name.return_value = name
    try:
        with patch("psutil.process_iter", return_value=processes):
            importlib.import_module(
                "rlinf.envs.realworld.common.spacemouse.spacemouse_expert"
            )
            # A later import/reload must also preserve an already running master.
            importlib.reload(importlib.import_module("rlinf.envs.realworld"))
        sender.send([process.kill.call_count for process in processes])
    finally:
        sender.close()


@pytest.mark.parametrize(
    "process_name",
    ["MainProcess", "dexhand-teleop", "spacemouse-reader", "other-reader"],
)
def test_fresh_import_preserves_ros_processes(process_name: str) -> None:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_import_in_child, args=(sender,), name=process_name)
    try:
        child.start()
        sender.close()
        assert receiver.poll(60), "RealWorld import timed out"
        assert receiver.recv() == [0, 0, 0], "Import terminated shared ROS processes"
        child.join(timeout=5)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        receiver.close()
        sender.close()
