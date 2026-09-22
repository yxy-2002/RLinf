# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Legacy 12-D button/hold behavior is unchanged by structured targets."""

import ast
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import gymnasium as gym
import numpy as np
from rlinf_dexhand import debug_trace as trace
from rlinf_dexhand.retargeting.channel_linear import ChannelLinear
from rlinf_dexhand.types import HandTarget

ROOT = Path(__file__).resolve().parents[3]


class Glove:
    def __init__(self, **kwargs):
        self.values = np.zeros(6)

    def get_target(self):
        return HandTarget(
            ChannelLinear("left").spec, tuple(self.values), 1, time.time()
        )

    def close(self):
        pass


class Mouse:
    def __init__(self):
        self.buttons = [0, 0]

    def get_action(self):
        return np.zeros(6), self.buttons


class Env(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float64)
    observation_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float64)
    config = SimpleNamespace(hand_reset_state=[0.2] * 6)

    def reset(self, **kwargs):
        return np.zeros(12), {}

    def step(self, action):
        return action, 0, False, False, {}


def test_relative_press_release_and_policy_fallback():
    source = (
        ROOT / "rlinf/envs/realworld/common/wrappers/dexhand_intervention.py"
    ).read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    ns = {
        "gym": gym,
        "trace": trace,
        "uuid": uuid,
        "np": np,
        "time": time,
        "Optional": Optional,
        "GloveExpert": Glove,
        "SpaceMouseExpert": Mouse,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            "<wrapper>",
            "exec",
        ),
        ns,
    )
    w = ns["DexHandIntervention"](Env())
    w.reset()
    w._spacemouse.buttons = [0, 1]
    first, replaced = w.action(np.ones(12))
    assert replaced
    np.testing.assert_allclose(first[6:], [0.2] * 6)
    w._glove.values[:] = 0.3
    moved, _ = w.action(np.ones(12))
    np.testing.assert_allclose(moved[6:], [0.5] * 6)
    # A retained glove sample must not accumulate hand motion across steps.
    for _ in range(20):
        held_sample, _ = w.action(np.ones(12))
        np.testing.assert_allclose(held_sample[6:], [0.5] * 6)
    # Fresh data resumes the existing press-relative mapping without rebasing.
    w._glove.values[:] = 0.4
    recovered, _ = w.action(np.ones(12))
    np.testing.assert_allclose(recovered[6:], [0.6] * 6)
    w._glove.values[:] = 0.3
    w.action(np.ones(12))
    w._spacemouse.buttons = [0, 0]
    w._glove.values[:] = 0.9
    w._last_intervene = 0
    held, replaced = w.action(np.ones(12))
    assert not replaced
    np.testing.assert_allclose(held[:6], np.ones(6))
    np.testing.assert_allclose(held[6:], [0.5] * 6)
    assert "intervene_action" not in w.step(np.ones(12))[-1]
    w.close()
