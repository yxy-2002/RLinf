# Copyright 2025 The RLinf Authors.
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

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rlinf.envs.realworld.common.camera import CameraInfo
from rlinf.envs.realworld.franka import franka_env


@pytest.mark.parametrize("failed_name", ["wrist_1", "global"])
def test_reconnect_releases_all_devices_before_reopening(monkeypatch, failed_name):
    """A timeout on either camera must not reopen an occupied device."""
    occupied = set()
    created = []
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    class Camera:
        def __init__(self, info):
            if info.name in occupied:
                raise RuntimeError("Device or resource busy")
            occupied.add(info.name)
            self._camera_info = info
            self.fail = False
            self.closed = False
            created.append(self)

        def open(self):
            pass

        def close(self):
            occupied.remove(self._camera_info.name)
            self.closed = True

        def get_frame(self):
            if self.fail:
                raise queue.Empty
            return image

    monkeypatch.setattr(franka_env, "create_camera", Camera)
    monkeypatch.setattr(franka_env.time, "sleep", lambda _: None)
    env = object.__new__(franka_env.FrankaEnv)
    env.config = SimpleNamespace(is_dummy=False)
    env._camera_infos = [CameraInfo(name, name) for name in ("wrist_1", "global")]
    env._logger = Mock()
    env.camera_player = Mock()
    env.observation_space = {
        "frames": {
            info.name: SimpleNamespace(shape=image.shape) for info in env._camera_infos
        }
    }
    env._crop_frame = Mock(return_value=(image, image))
    env._open_cameras()
    originals = list(env._cameras)
    for camera in originals:
        camera.fail = camera._camera_info.name == failed_name
    try:
        frames = env._get_camera_frames()
        assert set(frames) == {"wrist_1", "global"}
        assert all(camera.closed for camera in originals)
        assert len(created) == 4
        env.camera_player.put_frame.assert_called_once()
    finally:
        env._close_cameras()
    assert not occupied
