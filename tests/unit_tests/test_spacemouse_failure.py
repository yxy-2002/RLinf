# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Manual failure preserves executed actions and terminal transitions."""

from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
    DexHandIntervention,
)


@pytest.mark.parametrize("during_step", [False, True])
def test_right_button_failure_overrides_success_and_is_consumed_once(during_step):
    env = gym.Env()
    env.step = Mock(
        side_effect=lambda a: (
            {},
            1.0,
            True,
            True,
            {"executed_action": a.copy(), "success": True},
        )
    )
    wrapper = object.__new__(DexHandIntervention)
    gym.Wrapper.__init__(wrapper, env)
    wrapper._release_behavior = "policy"
    wrapper._right_button_ends_episode = True
    wrapper._failure_sequence = 0
    wrapper._teleop = Mock()
    # Button has already been released. The trailing counter retains the tap.
    before = np.array([1, 1, 1, 0, 0, 0] + [0] * 6 + [0.7] * 6)
    before = np.r_[before, 0 if during_step else 1]
    after = before.copy()
    after[-1] = 1
    wrapper._teleop.snapshot.side_effect = [before, after]
    action = np.full(12, 0.2)
    _, reward, term, trunc, info = wrapper.step(action)
    assert (reward, term, trunc) == (0.0, True, False)
    assert info["manual_failure"] and info["right_click"]
    assert not info["success"] and not info["succeed"]
    assert not info["right"]  # The level alone would have lost this short press.
    assert info["manual_failure_source"] == "right_click"
    assert "intervene_action" not in info
    np.testing.assert_array_equal(info["executed_action"], action)
    np.testing.assert_array_equal(env.step.call_args.args[0], action)
    wrapper._teleop.snapshot.side_effect = [after, after]
    *_, info = wrapper.step(action)
    assert not info["manual_failure"]
    # Resume resets the event counter; a button still held must not fail again.
    resumed = after.copy()
    resumed[-1] = 0
    resumed[4] = 1
    env.reset = Mock(return_value=({}, {}))
    wrapper._teleop.snapshot.side_effect = [resumed]
    wrapper.reset()
    wrapper._teleop.snapshot.side_effect = [resumed, resumed]
    *_, info = wrapper.step(action)
    assert info["right"] and not info["manual_failure"]
    wrapper._teleop.snapshot.side_effect = [after, after]
    *_, info = wrapper.step(action)
    assert info["manual_failure"]


def test_right_failure_ends_chunk_and_preserves_terminal_observation():
    import torch

    from rlinf.envs.lamp_realworld_adapter import RealWorldLampAdapter

    env = gym.Env()
    env.step = Mock(
        side_effect=lambda a: (
            {"terminal": True},
            0.0,
            False,
            False,
            {"executed_action": a.copy()},
        )
    )
    wrapper = object.__new__(DexHandIntervention)
    gym.Wrapper.__init__(wrapper, env)
    wrapper._release_behavior = "policy"
    wrapper._right_button_ends_episode = True
    wrapper._failure_sequence = 0
    wrapper._teleop = Mock()
    base = np.array([1, 1, 1, 0, 0, 0] + [0] * 6 + [0.7] * 6 + [0])
    failed = base.copy()
    failed[-1] = 1
    wrapper._teleop.snapshot.side_effect = [base, base, base, failed]
    adapter = object.__new__(RealWorldLampAdapter)
    adapter.env = wrapper
    adapter.robot_spec = SimpleNamespace(action_dim=12)
    adapter.continue_intervention = True
    adapter.auto_reset = True
    adapter.reset = Mock(return_value=({"reset": True}, {}))

    def step(action):
        obs, reward, term, trunc, info = wrapper.step(action[0].numpy())
        info["executed_action"] = torch.tensor(info["executed_action"])[None]
        return (
            obs,
            torch.tensor([reward]),
            torch.tensor([term]),
            torch.tensor([trunc]),
            info,
        )

    adapter.step = step
    obs, rewards, terms, truncs, infos = adapter.chunk_step(torch.zeros(1, 8, 12))
    assert env.step.call_count == 2
    assert terms.tolist() == [[False, True, False, False, False, False, False, False]]
    assert not rewards.any() and not truncs.any()
    assert infos[-1]["primitive_valid"].sum() == 2
    assert infos[-1]["final_info"]["manual_failure"]
    assert infos[-1]["final_observation"] == {"terminal": True}
    assert obs[-1] == {"reset": True}
    adapter.reset.assert_called_once_with()


def test_pending_failure_survives_release_without_sending_stale_actions():
    wrapper = object.__new__(DexHandIntervention)
    gym.Wrapper.__init__(wrapper, gym.Env())
    wrapper._release_behavior = "policy"
    wrapper._right_button_ends_episode = True
    wrapper._failure_sequence = 0
    wrapper._policy_control_state = (False, True)
    wrapper._teleop = Mock()
    wrapper._teleop.snapshot.return_value = np.array(
        [1, 1, 1, 0, 0, 0] + [0] * 6 + [0.7] * 6 + [1]
    )
    assert not wrapper.prepare_lamp_chunk_step(fresh_chunk=False)
    assert wrapper._failure_sequence == 0
    assert wrapper.prepare_lamp_chunk_step(fresh_chunk=True)
    assert wrapper._failure_pending(wrapper._lamp_prepared_input[0])
