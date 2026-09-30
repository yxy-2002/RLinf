# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""The unified wrapper validates hand dimensions before starting a child."""

import ast
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest


def wrapper_class():
    path = (
        Path(__file__).resolve().parents[3]
        / "rlinf/envs/realworld/common/wrappers/dexhand_intervention.py"
    )
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DexHandIntervention"
    )
    namespace = {"gym": gym, "np": np}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["DexHandIntervention"]


@pytest.mark.parametrize(
    "hand_type,wrong_dimension", [("wuji_hand", 12), ("ruiyan_hand", 26)]
)
def test_wrong_dimension_fails_before_process_or_device_open(
    hand_type, wrong_dimension
):
    env = gym.Env()
    env.action_space = gym.spaces.Box(-1, 1, (wrong_dimension,))
    env.config = SimpleNamespace(end_effector_type=hand_type)
    with pytest.raises(ValueError, match="dimension"):
        wrapper_class()(env)


@pytest.mark.parametrize("hand_type", ["wuji_hand", "ruiyan_hand"])
def test_policy_release_is_explicitly_rejected(hand_type):
    env = gym.Env()
    env.config = SimpleNamespace(end_effector_type=hand_type)
    with pytest.raises(ValueError, match="release_behavior=hold"):
        wrapper_class()(env, release_behavior="policy")
