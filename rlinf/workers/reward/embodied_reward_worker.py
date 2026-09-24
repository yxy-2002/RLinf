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

import asyncio
import os
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.scheduler import (
    Channel,
    Cluster,
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    Worker,
)
from rlinf.utils.placement import HybridComponentPlacement


class EmbodiedRewardWorker(Worker):
    """Reward Worker for inference during embodied RL training."""

    @staticmethod
    def launch_for_realworld(
        reward_cfg: dict,
        node_rank: int,
        node_group_label: Optional[str] = None,
        hardware_rank: Optional[int] = None,
        env_idx: int = 0,
        worker_rank: int = 0,
    ):
        """Launch a single-rank reward worker for real-world env inference."""
        cluster = Cluster()
        if hardware_rank is not None:
            placement = FlexiblePlacementStrategy(
                [[hardware_rank]],
                node_group_label=node_group_label,
            )
        elif node_group_label is not None:
            node_group = cluster.get_node_group(node_group_label)
            assert node_group is not None, (
                f"Node group {node_group_label} not found in cluster."
            )
            assert node_rank in node_group.node_ranks, (
                f"Node rank {node_rank} is not in node group {node_group_label}: "
                f"{node_group.node_ranks}"
            )
            placement_node_rank = node_group.node_ranks.index(node_rank)
            placement = NodePlacementStrategy(
                [placement_node_rank], node_group_label=node_group_label
            )
        else:
            placement = NodePlacementStrategy([node_rank])

        standalone_reward_cfg = dict(reward_cfg)
        standalone_reward_cfg["standalone_realworld"] = True
        standalone_cfg = OmegaConf.create({"reward": standalone_reward_cfg})
        return EmbodiedRewardWorker.create_group(standalone_cfg).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"EmbodiedRewardWorker-{worker_rank}-{env_idx}",
        )

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg

        self._standalone_realworld = self.cfg.reward.get("standalone_realworld", False)
        self.placement = (
            HybridComponentPlacement(cfg, Cluster())
            if not self._standalone_realworld
            else None
        )

        # Device setup
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        if not self._standalone_realworld:
            self.total_num_train_envs = cfg.env.train.total_num_envs
            self.total_num_eval_envs = (
                cfg.env.eval.total_num_envs
                if cfg.env.get("eval", None) is not None
                else 0
            )
            self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
            self.train_batch_size = (
                self.total_num_train_envs // self.num_pipeline_stages
            )
            self.eval_batch_size = self.total_num_eval_envs // self.num_pipeline_stages
        else:
            self.total_num_train_envs = 1
            self.total_num_eval_envs = 1
            self.num_pipeline_stages = 1
            self.train_batch_size = 1
            self.eval_batch_size = 1

        self.enable_offload = self.cfg.reward.get("enable_offload", False)
        self._interact_task = None

        self.reward_threshold = self.cfg.reward.get("reward_threshold", 0.6)
        self._use_reward_prob = self.cfg.reward.get("use_reward_prob", False)

        self.env_decoupled_mode = self.cfg.get("runner", {}).get(
            "enable_decoupled_mode", False
        )

        if self.env_decoupled_mode:
            # save the run-time imformation in communicate channel for decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {
                "train_reward_obs": [],
            }

    def model_provider_func(self):
        from rlinf.models.embodiment.reward import get_reward_model_class

        reward_cls = get_reward_model_class(self.cfg.reward.model.model_type)

        model_cfg = self.cfg.reward.model
        with open_dict(model_cfg):
            model_cfg.num_envs = self.local_num_train_envs
        model = reward_cls(model_cfg)

        return model

    def init_worker(self):
        """Initialize the reward worker for inference."""
        if self._standalone_realworld:
            self.local_num_train_envs = self.total_num_train_envs
        else:
            assert self.train_batch_size % self._world_size == 0, (
                f"train_batch_size ({self.train_batch_size}) must be divisible by "
                f"world_size ({self._world_size})."
            )
            self.local_num_train_envs = self.train_batch_size // self._world_size

        self.model = self.model_provider_func()

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        if self._standalone_realworld:
            return

    @Worker.timer("compute_rewards")
    async def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.model.to(self.device)

        total_last_run_count = 0
        while True:
            merged_data = await self.recv_from(
                group_name=self.cfg.env.group_name,
                channel=input_channel,
                tag="train_reward_obs",
                async_op=True,
                batch_size=self.train_batch_size,
            ).async_wait()
            last_run = merged_data.get("last_run", None)
            last_run_count = int(last_run.sum().item()) if last_run is not None else 0
            rewards = self.compute_image_rewards(observations=merged_data)
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.contiguous()
            self.send_to(
                group_name=self.cfg.env.group_name,
                channel=output_channel,
                data=rewards,
                tag="train_reward_obs",
                async_op=True,
            )
            total_last_run_count += last_run_count
            if total_last_run_count >= self.local_num_train_envs:
                break

        if self.enable_offload:
            self.model.to("cpu")

    @Worker.timer("compute_image_rewards")
    def compute_image_rewards(
        self, observations: dict[str, Any]
    ) -> torch.Tensor | np.ndarray:
        """Compute reward scores from observation input.

        Interface:
            - Input: ``observations`` (batched observation payload passed to
              ``self.model.compute_reward``).
            - Output: ``torch.Tensor`` or ``np.ndarray`` reward results. Tensor
              outputs are detached to CPU, and 1-D tensors are reshaped to ``(N, 1)``.

        Called from:
            - ``RewardWorker.compute_rewards`` (in-process)
            - ``RewardWorker._compute_rewards`` (in-process)
            - ``FrankaEnv._compute_reward_model`` via worker RPC
            - ``RealworldTeleopEvaluator._teleop_loop`` via worker RPC
        """
        rewards = self.model.compute_reward(observations)
        if rewards is not None and rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if isinstance(rewards, torch.Tensor):
            return rewards.detach().cpu()
        return rewards

    async def compute_rewards_async(
        self, input_channel: Channel, output_channel: Channel
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        self._interact_task = asyncio.create_task(
            self._compute_rewards(input_channel, output_channel)
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Continuously compute image rewards for embodied env batches.

        This private coroutine is used by ``compute_rewards_async`` in embodied RL. It
        receives image observations from Env Workers through routed worker communication,
        runs the embodied reward model with ``compute_image_rewards``, and sends the
        resulting rewards back to the Env Worker group.

        Unlike ``RewardWorker.compute_rewards``, this path operates on image data from
        ``train_reward_obs`` messages, can use ``env_decoupled_mode`` routing, and runs
        as a long-lived async loop until the task is stopped.

        Args:
            input_channel: Channel used to receive reward inputs from Env Workers.
            output_channel: Channel used to return computed rewards to Env Workers.
        """
        while True:
            merged_data = await self.recv_from(
                group_name=self.cfg.env.group_name,
                channel=input_channel,
                tag="train_reward_obs",
                async_op=True,
                batch_size=self.train_batch_size,
                decoupled_mode=self.env_decoupled_mode,
            ).async_wait()
            rewards = self.compute_image_rewards(observations=merged_data)
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.contiguous()
            self.send_to(
                group_name=self.cfg.env.group_name,
                channel=output_channel,
                data=rewards,
                tag="train_reward_obs",
                async_op=True,
                decoupled_mode=self.env_decoupled_mode,
            )

    async def stop(self):
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()
