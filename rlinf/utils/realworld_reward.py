# Copyright 2026 The RLinf Authors.
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

"""Shared placement injection for environment-owned real-world reward workers."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from rlinf.scheduler import Cluster, ComponentPlacement


def inject_realworld_reward_cfg(
    cfg: DictConfig,
    env_cfg: DictConfig,
    component_placement: "ComponentPlacement",
    cluster: "Cluster",
) -> DictConfig:
    """Inject placement into a config copy, or leave disabled configs untouched."""
    reward = cfg.get("reward", {})
    if not (
        reward.get("use_reward_model", False)
        and reward.get("standalone_realworld", False)
    ):
        return env_cfg
    placements = component_placement.get_strategy("reward").get_placement(cluster)
    ranks = component_placement.get_hardware_ranks("reward")
    if not placements or not ranks:
        raise ValueError("Standalone reward inference requires reward placement")
    if env_cfg.get("override_cfg", {}).get("reward_success_confirmation", False):
        if not reward.model.get("model_path"):
            raise ValueError("Set reward.model.model_path to a trained checkpoint")
        if reward.model.get("reward_threshold") is not None:
            raise ValueError("Use reward.reward_threshold, not model.reward_threshold")
    placement = placements[0]
    if env_cfg.get("override_cfg", {}).get("reward_success_confirmation", False):
        # The collector receives this subtree separately from the root config.
        result = OmegaConf.create(OmegaConf.to_container(env_cfg, resolve=True))
    else:
        result = OmegaConf.create(env_cfg)
    override = OmegaConf.create(
        OmegaConf.to_container(
            env_cfg.get("override_cfg", OmegaConf.create({})), resolve=True
        )
    )
    result.override_cfg = override
    override = result.override_cfg
    override.use_reward_model = True
    override.reward_worker_cfg = OmegaConf.to_container(reward, resolve=True)
    override.reward_worker_hardware_rank = ranks[0]
    override.reward_worker_node_rank = placement.cluster_node_rank
    override.reward_worker_node_group = placement.node_group_label
    override.reward_image_key = env_cfg.main_image_key
    if reward.model.get("camera_keys"):
        override.reward_camera_keys = reward.model.camera_keys
    return result
