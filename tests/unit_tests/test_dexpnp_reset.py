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

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.realworld.franka.tasks.dex_pnp import DexpnpConfig, DexpnpEnv


@pytest.mark.parametrize(
    "region", [[0] * 6, [0.05] * 3 + [0] * 3, [0, 0, 0, 0.1, 0.1, 0.1]]
)
@pytest.mark.parametrize("absolute", [False, True])
def test_reset_moves_directly_to_target_plus_offset(monkeypatch, region, absolute):
    """Reset sends no intermediate lift target before the configured rest pose."""
    target = np.array([0.6, 0.08, 0.41, 2.9, -0.16, 0.5])
    offset = np.array([0, 0, 0.05, 0, 0, 0])
    cfg = DexpnpConfig(
        target_ee_pose=target,
        reset_ee_pose_offset=offset.tolist(),
        ee_pose_limit_min_offset=[-0.2] * 6,
        ee_pose_limit_max_offset=[0.2] * 6,
        enable_random_reset=False,
        random_reset_ee_pose_region=region,
        reset_ee_pose=target.tolist() if absolute else None,
        end_effector_type="ruiyan_hand",
    )
    nominal = target if absolute else target + offset
    np.testing.assert_allclose(cfg.reset_ee_pose, nominal)
    region = np.array(region)
    draws = Mock(side_effect=[region * 0.6, -region * 0.4])
    monkeypatch.setattr(np.random, "uniform", draws)
    env = object.__new__(DexpnpEnv)
    env.config = cfg
    env._reset_pose = np.concatenate(
        [
            cfg.reset_ee_pose[:3],
            Rotation.from_euler("xyz", cfg.reset_ee_pose[3:]).as_quat(),
        ]
    )
    initial = env._reset_pose.copy()
    if not np.any(region[3:]):
        initial[2] += 0.1
    state = SimpleNamespace(tcp_pose=initial)
    env._controller = Mock()
    env._controller.get_state.return_value.wait.return_value = [state]
    env._end_effector_action = Mock()
    env._move_action = Mock()

    def arrive(pose):
        state.tcp_pose = pose.copy()

    env._interpolate_move = Mock(side_effect=arrive)
    for scale in (0.6, -0.4):
        env._interpolate_move.reset_mock()
        state.tcp_pose = initial.copy()
        env.go_to_rest()
        env._interpolate_move.assert_called_once()
        expected = nominal + region * scale
        actual = env._interpolate_move.call_args.args[0]
        np.testing.assert_allclose(actual[:3], expected[:3])
        np.testing.assert_allclose(
            Rotation.from_quat(actual[3:]).as_matrix(),
            Rotation.from_euler("xyz", expected[3:]).as_matrix(),
        )
        np.testing.assert_allclose(cfg.reset_ee_pose, nominal)
        np.testing.assert_allclose(env._reset_pose[:3], nominal[:3])
    if np.any(region):
        assert draws.call_count == 2
        np.testing.assert_array_equal(draws.call_args.args[0], -region)
        np.testing.assert_array_equal(draws.call_args.args[1], region)
    else:
        draws.assert_not_called()
    env._controller.reset_joint.assert_not_called()
    assert env._controller.reset_end_effector.call_count == 2
