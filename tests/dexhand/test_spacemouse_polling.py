# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""SpaceMouse polling remains bounded without physical devices."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.realworld.common.spacemouse import spacemouse_expert as mouse_module


class Clock:
    def __init__(self):
        self.wall = 0.0

    def monotonic(self):
        return self.wall


def test_spacemouse_waits_only_remaining_period(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(mouse_module, "time", clock)
    waits = []
    durations = iter([0.001, 0.006])

    class Stop:
        def is_set(self):
            return len(waits) == 2

        def wait(self, timeout):
            waits.append(timeout)
            clock.wall += timeout

    mouse = mouse_module.SpaceMouseExpert.__new__(mouse_module.SpaceMouseExpert)
    mouse._stop = Stop()
    mouse.state_lock = threading.Lock()
    mouse.latest_data = {}

    def read():
        clock.wall += next(durations)
        return SimpleNamespace(x=1, y=2, z=3, roll=4, pitch=5, yaw=6, buttons=[0, 1])

    mouse._device = SimpleNamespace(read=read)
    mouse._read_spacemouse()
    assert waits == pytest.approx([0.003, 0.0])
    np.testing.assert_array_equal(mouse.get_action()[0], [-2, 1, 3, -4, -5, -6])
    assert mouse.get_action()[1] == [0, 1]
