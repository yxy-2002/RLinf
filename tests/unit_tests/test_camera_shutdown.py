# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Test camera shutdown without importing robot setup or accessing hardware."""

import importlib.util
import sys
import threading
from pathlib import Path
from unittest.mock import patch

path = (
    Path(__file__).resolve().parents[2]
    / "rlinf/envs/realworld/common/camera/base_camera.py"
)
spec = importlib.util.spec_from_file_location("camera_shutdown_base", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Camera(module.BaseCamera):
    def __init__(self, fps=100):
        super().__init__(module.CameraInfo("fake", "fake", fps=fps))
        self.reading = threading.Event()
        self.release = threading.Event()
        self.closes = 0
        self.reads = 0

    def _read_frame(self):
        self.reads += 1
        self.reading.set()
        self.release.wait(4)
        raise RuntimeError("device read interrupted")

    def _close_device(self):
        self.closes += 1
        self.release.set()


def test_close_interrupts_pacing_without_read():
    c = Camera(fps=0.1)
    c.open()
    c.close()
    c.close()
    assert c.reads == 0 and c.closes == 1
    assert not c._frame_capturing_thread.is_alive()


def test_close_unblocks_read_without_false_error():
    c = Camera()
    with patch.object(module._logger, "error") as error:
        c.open()
        assert c.reading.wait(1)
        c.close()
        assert not c._frame_capturing_thread.is_alive()
        error.assert_not_called()
    assert c.closes == 1


def test_runtime_failure_still_reported():
    c = Camera()
    c.release.set()
    with patch.object(module._logger, "error") as error:
        c.open()
        c._frame_capturing_thread.join(1)
        assert not c._frame_capturing_thread.is_alive()
        error.assert_called_once()
    c.close()


def test_close_before_open():
    c = Camera()
    c.close()
    c.close()
    assert c.closes == 1
