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
from scipy.spatial.transform import Rotation

from rlinf.envs.realworld.franka.tasks.dex_pnp import DexpnpConfig, DexpnpEnv


def test_reset_moves_directly_to_target_plus_offset():
    """Reset sends no intermediate lift target before the configured rest pose."""
    target = np.array([0.6, 0.08, 0.41, 2.9, -0.16, 0.5])
    offset = np.array([0, 0, 0.05, 0, 0, 0])
    cfg = DexpnpConfig(
        target_ee_pose=target,
        reset_ee_pose_offset=offset.tolist(),
        ee_pose_limit_min_offset=[-0.2] * 6,
        ee_pose_limit_max_offset=[0.2] * 6,
        enable_random_reset=False,
        end_effector_type="ruiyan_hand",
    )
    np.testing.assert_allclose(cfg.reset_ee_pose, target + offset)
    env = object.__new__(DexpnpEnv)
    env.config = cfg
    env._reset_pose = np.concatenate(
        [
            cfg.reset_ee_pose[:3],
            Rotation.from_euler("xyz", cfg.reset_ee_pose[3:]).as_quat(),
        ]
    )
    initial = env._reset_pose.copy()
    initial[2] += 0.1
    state = SimpleNamespace(tcp_pose=initial)
    env._controller = Mock()
    env._controller.get_state.return_value.wait.return_value = [state]
    env._end_effector_action = Mock()
    env._move_action = Mock()

    def arrive(pose):
        state.tcp_pose = pose.copy()

    env._interpolate_move = Mock(side_effect=arrive)
    env.go_to_rest()
    env._interpolate_move.assert_called_once()
    np.testing.assert_allclose(env._interpolate_move.call_args.args[0], env._reset_pose)
    env._controller.reset_joint.assert_not_called()
    env._controller.reset_end_effector.assert_called_once()
