# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from examples.embodiment.franka_ruiyan import collect_demos as module
from rlinf.utils.ruiyan_reward_protocol import SuccessGate, executed_action


def test_gate_and_actual_hold():
    gate = SuccessGate(0.9, 3)
    assert [gate.update(p) for p in [0.95, 0.8, 0.95, 0.96, 0.99]] == [
        False,
        False,
        False,
        False,
        True,
    ]
    with pytest.raises(ValueError):
        gate.update(float("nan"))
    action = executed_action(
        {"intervene_flag": [False], "teleop_hand_target": np.full((1, 6), 0.4)}
    )
    torch.testing.assert_close(action[0, 6:], torch.full((6,), 0.4))
    assert action[0, :6].sum() == 0
    with pytest.raises(ValueError):
        executed_action({"intervene_flag": [False]})


@pytest.mark.parametrize("decision,expected", [("accept", 1), ("discard", 0)])
def test_complete_demo_and_rejection(monkeypatch, tmp_path, decision, expected):
    saved = []

    class Buffer:
        def __init__(self, **kwargs):
            pass

        def add_trajectories(self, items):
            saved.extend(items)

        def close(self):
            pass

    class Client:
        def __init__(self, *args):
            self.started = False

        def request(self, *args):
            return {"image_keys": ["global", "wrist_1"]}

        def status(self, state, **kwargs):
            if state == "waiting":
                if self.started:
                    return {"command": "quit"}
                self.started = True
                return {"command": "start"}
            return {"command": decision if state == "candidate" else None}

        def predict(self, images):
            return 0.99

    class Env:
        resets = 0
        closed = False
        action_space = SimpleNamespace(shape=(12,))

        def __init__(self, *args, **kwargs):
            self.env = self
            self.count = 0

        def obs(self):
            return {
                "main_images": torch.full(
                    (1, 128, 128, 3), self.count, dtype=torch.uint8
                ),
                "extra_view_images": torch.ones(1, 1, 128, 128, 3, dtype=torch.uint8),
                "states": torch.zeros(1, 24),
            }

        def reset(self):
            Env.resets += 1
            return self.obs(), {}

        def step(self, a):
            self.count += 1
            return (
                self.obs(),
                torch.ones(1),
                torch.ones(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {"intervene_flag": [False], "teleop_hand_target": np.full((1, 6), 0.4)},
            )

        def close(self):
            Env.closed = True

    monkeypatch.setattr(module, "RewardClient", Client)
    monkeypatch.setattr(module, "RealWorldEnv", Env)
    monkeypatch.setattr(module, "TrajectoryReplayBuffer", Buffer)
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {"log_path": str(tmp_path)},
                "reward_url": "http://localhost",
                "rpc_timeout": 1,
                "num_data_episodes": 1,
                "success_threshold": 0.9,
                "success_hold_steps": 3,
                "min_episode_steps": 1,
                "require_confirmation": True,
            },
            "env": {"eval": {"max_episode_steps": 10}},
        }
    )
    collector = SimpleNamespace(cfg=cfg, worker_info=None, log_info=lambda x: None)
    module.DemoCollector.run(collector)
    assert Env.closed and Env.resets == 1
    assert len(saved) == expected
    record = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert record["steps"] == 3 and record["saved"] == bool(expected)
    if saved:
        t = saved[0]
        assert t.actions.shape == (3, 1, 12)
        assert t.rewards.flatten().tolist() == [0.0, 0.0, 1.0]
        assert t.terminations.flatten().tolist() == [False, False, True]
        assert t.curr_obs["main_images"][:, 0, 0, 0, 0].tolist() == [0, 1, 2]
        assert t.next_obs["main_images"][:, 0, 0, 0, 0].tolist() == [1, 2, 3]


def test_gym_vector_object_hand_target():
    target = np.empty(1, dtype=object)
    target[0] = np.array([0.4, 0.1, 0.2, 0.3, 0.5, 0.6])
    info = {
        "intervene_flag": [False],
        "teleop_hand_target": target,
        "_teleop_hand_target": np.array([True]),
    }
    result = executed_action(info)
    torch.testing.assert_close(
        result[0, 6:], torch.tensor(target[0], dtype=torch.float32)
    )
    info["_teleop_hand_target"][0] = False
    with pytest.raises(ValueError):
        executed_action(info)


def test_disabled_video_player_stop():
    from rlinf.envs.realworld.common.video_player.video_player import VideoPlayer

    player = VideoPlayer(enable=False)
    player.stop()
    player.stop()
    assert not player.is_running
