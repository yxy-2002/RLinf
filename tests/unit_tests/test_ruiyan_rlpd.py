# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ruiyan RLPD conventions and sparse reward behavior, without robot devices."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.utils.ruiyan_rlpd import (
    OnlineReward,
    close_on_error,
    decode_action,
    encode_action,
)


def test_action_roundtrip():
    original = np.array(
        [[1.14, -1.1, 0, 0.2, -0.1, 0, 0, 0.2, 0.4, 0.6, 0.8, 1]], dtype=np.float32
    )
    normalized = encode_action(original)
    assert np.max(np.abs(normalized)) <= 1
    np.testing.assert_allclose(decode_action(normalized), original, atol=1e-6)
    assert original[0, 0] == np.float32(1.14)


@pytest.mark.parametrize(
    "action,scale",
    [
        (np.zeros(11), 2),
        (np.full(12, np.nan), 2),
        (np.zeros(12), 0),
        (np.zeros(12), float("nan")),
    ],
)
def test_invalid_actions(action, scale):
    with pytest.raises(ValueError):
        encode_action(action, scale)
    with pytest.raises(ValueError):
        decode_action(action, scale)


def test_no_silent_clipping():
    a = np.zeros(12)
    a[0] = 2.1
    with pytest.raises(ValueError):
        encode_action(a)
    a = np.zeros(12)
    a[6] = -0.1
    with pytest.raises(ValueError):
        encode_action(a)


def test_online_reward_and_manual_reset():
    cfg = OmegaConf.create(
        {
            "reward_url": "unused",
            "threshold": 0.9,
            "hold_steps": 1,
            "min_steps": 2,
            "manual_reset": True,
        }
    )
    with patch("rlinf.utils.ruiyan_reward_protocol.RewardClient") as factory:
        client = factory.return_value
        client.request.return_value = {"image_keys": ["global", "wrist_1"]}
        client.status.return_value = {"command": "start"}
        reward = OnlineReward(cfg)
        reward.before_reset()
        client.status.return_value = {"command": None}
        obs = {
            "main_images": np.zeros((1, 128, 128, 3), dtype=np.uint8),
            "extra_view_images": np.ones((1, 1, 128, 128, 3), dtype=np.uint8),
        }
        client.predict.return_value = 0.95
        assert reward.evaluate(obs) == (0.95, False, False)
        assert reward.evaluate(obs) == (
            0.95,
            True,
            False,
        )  # one eligible frame suffices
        client.predict.return_value = 0.9
        assert reward.evaluate(obs) == (0.9, False, False)  # strict threshold
        client.predict.return_value = 0.95
        client.status.return_value = {"command": "discard"}
        assert reward.evaluate(obs) == (0.95, False, True)
        client.status.return_value = {"command": "quit"}
        with pytest.raises(RuntimeError):
            reward.evaluate(obs)


def test_failure_cleanup():
    calls = []
    env = SimpleNamespace(_ruiyan_reward=object(), close=lambda: calls.append("closed"))

    @close_on_error
    def fail(self):
        raise TimeoutError("reward server down")

    with pytest.raises(TimeoutError):
        fail(env)
    assert calls == ["closed"]


def test_policy_hand_passthrough_and_takeover():
    # Package import otherwise performs ROS process cleanup; never do that in tests.
    with patch("psutil.process_iter", return_value=[]):
        from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
            DexHandIntervention,
        )
    wrapper = DexHandIntervention.__new__(DexHandIntervention)
    wrapper._policy_passthrough = True
    wrapper._spacemouse = SimpleNamespace(
        get_action=lambda: (np.zeros(6), [False, False])
    )
    wrapper._glove = SimpleNamespace(
        get_target=lambda: SimpleNamespace(values=np.full(6, 0.2))
    )
    wrapper._last_intervene = 0
    wrapper._timeout = 0.5
    wrapper._prev_left = False
    wrapper._hand_current = np.zeros(6)
    action = np.r_[np.zeros(6), np.full(6, 0.7)]
    result, intervened = wrapper.action(action)
    np.testing.assert_allclose(result, action)
    assert not intervened
    wrapper._spacemouse.get_action = lambda: (np.zeros(6), [False, True])
    result, intervened = wrapper.action(action)
    assert intervened
    np.testing.assert_allclose(result[6:], 0.7)  # first press starts at policy target
    wrapper._glove.get_target = lambda: SimpleNamespace(values=np.full(6, 0.3))
    result, _ = wrapper.action(action)
    np.testing.assert_allclose(result[6:], 0.8)
    # With policy passthrough disabled, idle hand control holds its last target.
    wrapper._policy_passthrough = False
    wrapper._spacemouse.get_action = lambda: (np.zeros(6), [False, False])
    wrapper._last_intervene = 0
    result, intervened = wrapper.action(action)
    assert not intervened
    np.testing.assert_allclose(result[6:], 0.8)


@pytest.mark.parametrize(
    "success,aborted", [(True, False), (False, False), (False, True)]
)
def test_reward_terminal_observation_and_intervention(success, aborted):
    import torch

    with patch("psutil.process_iter", return_value=[]):
        from rlinf.envs.realworld.realworld_env import RealWorldEnv
    env = RealWorldEnv.__new__(RealWorldEnv)
    env._ruiyan_cfg = {"arm_scale": 2}
    env._ruiyan_reward = SimpleNamespace(
        check_stopped=lambda: None,
        before_reset=lambda: None,
        evaluate=lambda obs: (0.95, success, aborted),
    )
    env.cfg = SimpleNamespace(max_episode_steps=1)
    env.num_envs = 1
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env.manual_episode_control_only = False
    env.ignore_terminations = False
    env.auto_reset = True
    env.use_fixed_reset_state_ids = False
    env._init_metrics()
    physical = np.r_[np.full(6, 0.3), np.full(6, 0.7)].reshape(1, 12)
    received = []

    def step(a):
        received.append(a)
        return (
            {"states": torch.ones(1, 24)},
            np.array([1.0]),
            np.array([True]),
            np.array([False]),
            {"intervene_action": physical.copy()},
        )

    env.env = SimpleNamespace(
        step=step, reset=lambda **kwargs: ({"states": torch.zeros(1, 24)}, {})
    )
    env._wrap_obs = lambda obs: obs
    obs, reward, term, trunc, info = env.step(np.zeros((1, 12)))
    assert bool(term[0]) == success
    assert bool(trunc[0]) == (not success)
    assert reward.item() == float(success)
    assert obs["states"].sum() == 0
    assert info["final_observation"]["states"].sum() == 24
    np.testing.assert_allclose(
        info["final_info"]["intervene_action"], encode_action(physical)
    )
    np.testing.assert_allclose(received[0][:, 6:], 0.5)


def test_wait_can_be_stopped():
    cfg = OmegaConf.create({"reward_url": "unused"})
    with patch("rlinf.utils.ruiyan_reward_protocol.RewardClient") as factory:
        factory.return_value.request.return_value = {
            "image_keys": ["global", "wrist_1"]
        }
        reward = OnlineReward(cfg)
        reward.stop_requested.set()
        with pytest.raises(RuntimeError, match="stopped"):
            reward.before_reset()


def test_async_stop_joins_waiting_step():
    import asyncio
    import threading

    from rlinf.workers.env.async_env_worker import AsyncEnvWorker

    started = threading.Event()
    stop = threading.Event()
    completed = []

    def step(actions, stage):
        started.set()
        assert stop.wait(timeout=3)
        completed.append(True)
        raise RuntimeError("stopped")

    worker = SimpleNamespace(
        env_interact_step=step, env_list=[SimpleNamespace(request_stop=stop.set)]
    )

    async def run():
        task = asyncio.create_task(
            AsyncEnvWorker._ruiyan_interact_step(worker, None, 0)
        )
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed == [True]

    asyncio.run(run())
