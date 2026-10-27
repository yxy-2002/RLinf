# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Disconnects gate policy execution and recover without operator input."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers import dexhand_intervention as module


def _client():
    client = object.__new__(module.TeleopProcess)
    client.process = SimpleNamespace(is_alive=lambda: True)
    client.connection = SimpleNamespace(poll=lambda: False)
    client.shared = None
    client.mouse_generation = 1
    return client


def _snapshot(connected, generation, failure=0):
    return np.array(
        [time.monotonic(), time.time(), 1, 0, 0, 0]
        + [0] * 6
        + [0.4] * 6
        + [failure, connected, generation]
    )


def _wrapper(client):
    env = gym.Env()
    env.step = Mock(
        side_effect=lambda action: (
            {},
            0.0,
            False,
            False,
            {"executed_action": action.copy()},
        )
    )
    wrapper = object.__new__(module.DexHandIntervention)
    gym.Wrapper.__init__(wrapper, env)
    wrapper._teleop = client
    wrapper._release_behavior = "policy"
    wrapper._mouse_generation = 1
    wrapper._right_button_ends_episode = True
    wrapper._failure_sequence = 0
    return wrapper


def test_disconnected_snapshot_blocks_commands_and_recovers_automatically(monkeypatch):
    client = _client()
    wrapper = _wrapper(client)
    frames = iter([_snapshot(False, 1), _snapshot(False, 1), _snapshot(True, 2)])
    sleeps = []

    def wait(interval):
        wrapper.env.step.assert_not_called()
        sleeps.append(interval)

    monkeypatch.setattr(module, "_read_snapshot", lambda shared: next(frames))
    monkeypatch.setattr(module.time, "sleep", wait)
    assert wrapper.prepare_lamp_chunk_step(fresh_chunk=True)
    assert len(sleeps) == 2
    monkeypatch.setattr(module, "_read_snapshot", lambda shared: _snapshot(True, 2))
    *_, info = wrapper.step(np.ones(12))
    # A plan produced before the outage is replaced by a stationary step.
    np.testing.assert_array_equal(wrapper.env.step.call_args.args[0][:6], 0)
    np.testing.assert_array_equal(wrapper.env.step.call_args.args[0][6:], 0.4)
    assert info["lamp_control_released"] and not info["manual_failure"]
    # No confirmation/reset is required: the subsequent plan executes normally.
    *_, info = wrapper.step(np.full(12, 0.2))
    np.testing.assert_array_equal(wrapper.env.step.call_args.args[0], 0.2)
    assert not info["lamp_control_released"]


def test_reconnect_between_steps_discards_chunk_suffix(monkeypatch):
    client = _client()
    wrapper = _wrapper(client)
    monkeypatch.setattr(module, "_read_snapshot", lambda shared: _snapshot(True, 2))
    assert not wrapper.prepare_lamp_chunk_step(fresh_chunk=False)
    wrapper.env.step.assert_not_called()
    assert wrapper._lamp_prepared_input is None


def test_outage_during_execution_cancels_next_action_and_preserves_failure(monkeypatch):
    wrapper = _wrapper(_client())
    frames = iter(
        [_snapshot(True, 1), _snapshot(False, 1), _snapshot(True, 2, failure=1)]
    )
    monkeypatch.setattr(module, "_read_snapshot", lambda shared: next(frames))
    *_, info = wrapper.step(np.full(12, 0.2))
    wrapper.env.step.assert_called_once()
    assert info["lamp_control_released"] and info["manual_failure"]


@pytest.mark.parametrize("failure", ["process", "heartbeat"])
def test_unrelated_teleop_failure_still_raises(monkeypatch, failure):
    client = _client()
    snapshot = _snapshot(False, 1)
    if failure == "process":
        client.process.is_alive = lambda: False
    else:
        snapshot[0] -= 1
    monkeypatch.setattr(module, "_read_snapshot", lambda shared: snapshot)
    with pytest.raises(RuntimeError, match="exited|stale"):
        client.snapshot()


@pytest.mark.parametrize("device", ["mouse", "glove"])
def test_input_worker_resets_glove_baseline_and_preserves_failure_sequence(
    monkeypatch, device
):
    import os
    import sys
    from types import ModuleType

    spec = SimpleNamespace(
        action_dim=6, lower=np.zeros(6), upper=np.ones(6), hand_type="ruiyanhand"
    )
    hand = Mock()
    hand.spec = spec
    hand.get_state.return_value = np.full(6, 0.4)
    glove = Mock()
    glove.get_target.return_value = SimpleNamespace(
        spec=spec, values=np.full(6, 0.5), timestamp=time.time(), sequence=1
    )
    mouse = Mock()
    mouse.get_button_press_counts.return_value = (0, 0)
    states = [
        # Take over, disconnect while holding, reconnect, release and take over.
        (True, 1, 0, False),
        (True, 1, 1, True),
        (False, 1, 2, True),  # A short right tap arrived just before disconnect.
        (True, 2, 2, True),
        (True, 2, 2, False),
        (True, 2, 3, True),
    ]
    tick = [0]

    def read():
        connected, generation, count, left = states[tick[0]]
        tick[0] += 1
        return SimpleNamespace(
            connected=connected if device == "mouse" else True,
            generation=generation if device == "mouse" else 1,
            press_counts=(count, 0),
            action=np.ones(6),
            buttons=[0, left],
        )

    mouse.get_state.side_effect = read

    def glove_state():
        connected, generation, _, _ = states[tick[0] - 1]
        return SimpleNamespace(
            target=glove.get_target.return_value,
            connected=connected if device == "glove" else True,
            generation=generation if device == "glove" else 1,
        )

    glove.get_state.side_effect = glove_state

    imports = {
        "rlinf_dexhand.glove": {"GloveExpert": lambda **kw: glove},
        "rlinf_dexhand.wuji_spec": {
            "to_normalized": lambda spec, value: value,
            "to_radians": lambda spec, value: value,
        },
        "rlinf.envs.realworld.common.hand.wuji_hand": {"WujiHand": Mock()},
        "rlinf.envs.realworld.common.ros": {"ROSController": Mock()},
        "rlinf.envs.realworld.common.spacemouse.spacemouse_expert": {
            "SpaceMouseExpert": lambda: mouse
        },
    }
    for name, values in imports.items():
        stub = ModuleType(name)
        stub.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(module, "RuiyanControllerClient", lambda config: hand)
    connection = Mock()
    resumed = [False]
    connection.poll.side_effect = lambda *a: not resumed[0] or tick[0] == len(states)

    def receive():
        if resumed[0]:
            return "stop"
        resumed[0] = True
        return "resume"

    connection.recv.side_effect = receive
    snapshots = []
    monkeypatch.setattr(
        module, "_write_snapshot", lambda _, values: snapshots.append(values)
    )
    module._teleop_worker(
        connection,
        [0.0] * 21,
        {
            "release_behavior": "policy",
            "hand_type": "ruiyan_hand",
            "hand": {},
            "glove_frequency": 60,
            "frequency": 60,
            "pipeline_config": {},
            "scale_file": None,
            "parent_pid": os.getppid(),
            "mode": "relative",
            "max_delta": 0.05,
            "right_button_labels_only": True,
            "right_button_ends_episode": True,
            "timeout": 0.5,
        },
    )
    assert not any(
        call.args[0][0] == "error" for call in connection.send.call_args_list
    )
    frames = snapshots[1:]  # The first write clears state on resume.
    assert [frame[18] for frame in frames] == [0, 1, 2, 2, 2, 3]
    assert [bool(frame[5]) for frame in frames] == [
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    np.testing.assert_array_equal(frames[2][6:12], 0)
    np.testing.assert_array_equal(frames[-1][12:18], 0.4)
    hand.command.assert_not_called()  # Policy and human actions share the env executor.
    mouse.close.assert_called_once()
