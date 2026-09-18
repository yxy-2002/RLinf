# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import pickle

import numpy as np
import pytest
import torch

from rlinf.envs.wrappers.collect_episode import CollectEpisode


class FakeEnv:
    def reset(self, **kwargs):
        return np.zeros((2, 24)), {}

    def step(self, action):
        self.input_action = action.copy()
        return (
            np.ones((2, 24)),
            np.zeros(2),
            np.ones(2, bool),
            np.zeros(2, bool),
            self.info,
        )

    def close(self):
        pass


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("final", [False, True])
def test_saved_pickle_records_intervention(tmp_path, tensor, final):
    env = FakeEnv()
    expert = np.arange(24, dtype=np.float32).reshape(2, 12)
    info = {
        "intervene_action": expert,
        "intervene_flag": np.array([True, False]),
        "_intervene_action": np.array([True, True]),
    }
    if tensor:
        info = {k: torch.from_numpy(v.copy()) for k, v in info.items()}
    env.info = (
        {"final_observation": np.ones((2, 24)), "final_info": info} if final else info
    )
    wrapper = CollectEpisode(env, str(tmp_path), num_envs=2, show_goal_site=False)
    submitted = np.zeros((2, 12))
    try:
        wrapper.reset()
        wrapper.step(submitted)
        expert[:] = -1  # Writing must not retain mutable environment buffers.
    finally:
        wrapper.close()
    files = sorted(tmp_path.glob("*.pkl"))
    assert len(files) == 2
    with files[0].open("rb") as f:
        episode = pickle.load(f)
    np.testing.assert_array_equal(episode["actions"][0], np.arange(12))
    assert len(episode["observations"]) == len(episode["actions"]) + 1
    with files[1].open("rb") as f:
        episode = pickle.load(f)
    np.testing.assert_array_equal(episode["actions"][0], submitted[1])
    np.testing.assert_array_equal(env.input_action, submitted)


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"intervene_action": None},
        {"intervene_action": np.ones(12), "_intervene_action": False},
        {"intervene_action": np.ones(12), "intervene_flag": False},
    ],
)
def test_no_valid_intervention_preserves_policy(tmp_path, info):
    wrapper = CollectEpisode(FakeEnv(), str(tmp_path), show_goal_site=False)
    try:
        action = np.arange(12)
        saved = wrapper._recorded_action(action, info)
        action[:] = -1
        np.testing.assert_array_equal(saved, np.arange(12))
    finally:
        wrapper.close()


@pytest.mark.parametrize("flattened", [False, True])
def test_chunk_terminal_action(tmp_path, flattened):
    env = FakeEnv()
    history = np.arange(48).reshape(2, 2, 12)
    env.info = {
        "final_observation": np.ones((2, 24)),
        "final_info": {
            "intervene_action": history.reshape(2, -1) if flattened else history,
            "intervene_flag": np.array([[False, True], [True, False]]),
        },
    }
    wrapper = CollectEpisode(env, str(tmp_path), num_envs=2, show_goal_site=False)
    try:
        wrapper.reset()
        wrapper.step(np.zeros((2, 12)))
    finally:
        wrapper.close()
    with sorted(tmp_path.glob("*.pkl"))[0].open("rb") as f:
        episode = pickle.load(f)
    np.testing.assert_array_equal(episode["actions"][0], history[0, -1])
