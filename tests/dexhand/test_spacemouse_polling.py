# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Polling, native reader failures and reconnection without physical devices."""

import logging
import multiprocessing as mp
import os
import threading
import time
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rlinf.envs.realworld.common.spacemouse import spacemouse_expert as mouse_module


def _sample(buttons):
    return SimpleNamespace(x=1, y=2, z=3, roll=4, pitch=5, yaw=6, buttons=buttons)


def test_spacemouse_waits_only_remaining_period(monkeypatch):
    clock = SimpleNamespace(wall=0.0)
    monkeypatch.setattr(
        mouse_module, "time", SimpleNamespace(monotonic=lambda: clock.wall)
    )
    waits = []
    durations = iter([0.001, 0.006])

    def wait(timeout):
        waits.append(timeout)
        clock.wall += timeout

    def read():
        clock.wall += next(durations)
        return _sample([0, 0])

    connection = Mock()
    stop = SimpleNamespace(is_set=lambda: len(waits) == 2, wait=wait)
    mouse_module._poll_device(SimpleNamespace(read=read), connection, stop, 0.004)
    assert waits == pytest.approx([0.003, 0.0])
    np.testing.assert_array_equal(
        connection.send.call_args.args[0][1], [-2, 1, 3, -4, -5, -6]
    )


def test_button_counts_retain_taps_and_ignore_buttons_held_across_reconnect():
    device, connection, stop = Mock(), Mock(), Mock()
    buttons = [[1, 1], [0, 0], [1, 0], [1, 0], [0, 0], [1, 1], [0, 1]]
    device.read.side_effect = [_sample(b) for b in buttons]
    stop.is_set.side_effect = [False] * len(buttons) + [True]
    mouse_module._poll_device(device, connection, stop, 0.004)
    messages = [call.args[0] for call in connection.send.call_args_list]
    assert messages[0][2:] == ([0, 0], (0, 0))
    assert messages[-1][2:] == ([0, 1], (2, 1))


@pytest.mark.parametrize("open_fails", [False, True])
def test_reader_reports_open_and_read_errors_and_closes_device(monkeypatch, open_fails):
    import sys

    device = Mock()
    device.read.side_effect = OSError("HID disconnected")
    library = Mock()
    if open_fails:
        library.open.side_effect = OSError("HID disconnected")
    else:
        library.open.return_value = device
    monkeypatch.setitem(sys.modules, "pyspacemouse", library)
    connection = Mock()
    mouse_module._reader_worker(connection, threading.Event(), 0, 0.004)
    assert connection.send.call_args.args[0] == ("error", "OSError: HID disconnected")
    assert device.close.call_count == int(not open_fails)
    connection.close.assert_called_once()


def _controlled_reader(control, connection, stop, device_index, interval):
    with control.get_lock():
        control[1] += 1
    while not stop.is_set():
        mode = control[0]
        if mode == 1:
            connection.send(("error", "HID unplugged"))
            return
        if mode == 2:
            os._exit(23)  # Native failure cannot be caught by Python.
        if mode == 3:
            time.sleep(30)  # A stuck native call does not observe the stop event.
        connection.send((time.monotonic(), [0.5] * 6, [0, 0], (1, 0)))
        stop.wait(interval)


def _until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("SpaceMouse did not reach expected state")


@pytest.mark.parametrize("failure", [1, 2, 3])
def test_reader_recovers_from_exception_native_exit_and_hang(
    monkeypatch, caplog, failure
):
    monkeypatch.setattr(
        "rlinf.utils.logging.get_logger", lambda: logging.getLogger("spacemouse-test")
    )
    caplog.set_level(logging.INFO, logger="spacemouse-test")
    monkeypatch.setattr(mouse_module.SpaceMouseExpert, "RESTART_DELAY_S", 0.1)
    control = mp.get_context("spawn").Array("i", [0, 0])
    mouse = mouse_module.SpaceMouseExpert(worker=partial(_controlled_reader, control))
    try:
        _until(lambda: mouse.get_state().connected)
        old = mouse.get_state()
        assert old.generation == 1 and old.press_counts == (1, 0)
        control[0] = failure
        _until(lambda: not mouse.get_state().connected)
        state = mouse.get_state()
        np.testing.assert_array_equal(state.action, np.zeros(6))
        assert state.buttons == [0, 0]
        assert state.press_counts == (1, 0)
        # Keep unavailable long enough to exercise the watchdog and retries.
        time.sleep(0.35)
        control[0] = 0
        _until(lambda: mouse.get_state().connected and mouse.get_state().generation > 1)
        restored = mouse.get_state()
        assert restored.press_counts == (2, 0)
        assert control[1] >= 2
        messages = [record.getMessage() for record in caplog.records]
        assert sum("Connection interrupted" in msg for msg in messages) == 1
        assert sum("Connection restored" in msg for msg in messages) == 1
    finally:
        mouse.close()
        mouse.close()
    assert not mouse.thread.is_alive()
    assert not mouse.get_state().connected
    assert not any(p.name == "spacemouse-reader" for p in mp.active_children())


def test_cached_motion_expires_even_if_supervisor_stalls():
    mouse = mouse_module.SpaceMouseExpert.__new__(mouse_module.SpaceMouseExpert)
    mouse.state_lock = threading.Lock()
    mouse.thread = SimpleNamespace(is_alive=lambda: True)
    mouse._stop = threading.Event()
    mouse._timestamp = time.monotonic() - 1
    mouse._state = mouse_module.SpaceMouseState(np.ones(6), [1, 1], (2, 3), True, 1)
    state = mouse.get_state()
    assert not state.connected
    np.testing.assert_array_equal(state.action, np.zeros(6))
    assert state.buttons == [0, 0] and state.press_counts == (2, 3)


def test_close_terminates_a_blocked_native_reader(monkeypatch):
    monkeypatch.setattr(
        "rlinf.utils.logging.get_logger", lambda: logging.getLogger("spacemouse-test")
    )
    control = mp.get_context("spawn").Array("i", [0, 0])
    mouse = mouse_module.SpaceMouseExpert(worker=partial(_controlled_reader, control))
    try:
        _until(lambda: mouse.get_state().connected)
        control[0] = 3
        time.sleep(0.05)
        mouse.close()
        assert not mouse.thread.is_alive()
        assert not any(p.name == "spacemouse-reader" for p in mp.active_children())
    finally:
        mouse.close()
