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

"""Asynchronous replay ingestion and SAC learner execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
from typing import Any

import torch

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class AsyncSACExecutionMixin:
    """Add collector-round-aware asynchronous execution to a SAC worker.

    LAMP residual SAC grants optimizer work from newly ingested online macro
    transitions. One call to :meth:`run_training` consumes exactly one complete
    collector round; before ``learning_starts_macro_transitions`` it only grows
    replay, and afterwards it consumes the accumulated ``utd_ratio`` budget.

    The legacy fixed-round mode remains available to non-v3 async SAC configs
    that still define ``algorithm.async.max_learner_rounds_per_collector``.
    """

    def init_worker(self) -> None:
        super().init_worker()
        self._ensure_async_sac_state()

    def _ensure_async_sac_state(self) -> None:
        if getattr(self, "_async_sac_state_initialized", False):
            return

        async_cfg = self.cfg.algorithm.get("async", {}) or {}
        configured_rounds = async_cfg.get("max_learner_rounds_per_collector", None)
        configured_utd = self.cfg.algorithm.get("utd_ratio", None)
        configured_learning_starts = self.cfg.algorithm.get(
            "learning_starts_macro_transitions", None
        )
        self._utd_control_enabled = (
            configured_utd is not None and configured_learning_starts is not None
        )
        if self._utd_control_enabled and configured_rounds is not None:
            raise ValueError(
                "UTD-controlled async SAC cannot also configure "
                "max_learner_rounds_per_collector"
            )
        self._collector_control_enabled = (
            self._utd_control_enabled or configured_rounds is not None
        )
        self._max_learner_rounds_per_collector = int(configured_rounds or 1)
        self._utd_ratio = float(configured_utd or 0.0)
        self._learning_starts_macro_transitions = int(configured_learning_starts or 0)
        self._progressive_exploration_macro_steps = int(
            self.cfg.algorithm.get("progressive_exploration_macro_steps", 1)
        )
        max_pending = int(async_cfg.get("max_pending_collector_rounds", 2))
        if (
            not self._utd_control_enabled
            and self._max_learner_rounds_per_collector <= 0
        ):
            raise ValueError(
                "max_learner_rounds_per_collector must be greater than zero"
            )
        if self._utd_control_enabled and self._utd_ratio <= 0.0:
            raise ValueError("utd_ratio must be greater than zero")
        if self._utd_control_enabled and not math.isfinite(self._utd_ratio):
            raise ValueError("utd_ratio must be finite")
        if self._learning_starts_macro_transitions < 0:
            raise ValueError("learning_starts_macro_transitions must be non-negative")
        if self._progressive_exploration_macro_steps <= 0:
            raise ValueError(
                "progressive_exploration_macro_steps must be greater than zero"
            )
        if max_pending <= 0:
            raise ValueError("max_pending_collector_rounds must be greater than zero")

        self._received_collector_rounds: asyncio.Queue[list[Any]] = asyncio.Queue(
            maxsize=max_pending
        )
        self._recv_rollout_task: asyncio.Task | None = None
        self._recv_stop_event = asyncio.Event()
        self._collector_rounds_received = 0
        self._collector_rounds_committed = 0
        self._collector_rounds_consumed = 0
        self._learner_rounds_completed = 0
        self._online_macro_transitions = 0
        self._primitive_env_steps = 0
        self._optimizer_update_budget = 0.0
        self._optimizer_updates_completed = 0
        self._latest_behavior_version_mean = 0.0
        self._latest_behavior_version_min = 0.0
        self._async_sac_state_initialized = True

    async def recv_rollout_trajectories(self, input_channel) -> None:
        """Start a cancellable coroutine that receives complete collector rounds."""

        self._ensure_async_sac_state()
        if self._recv_rollout_task is None or self._recv_rollout_task.done():
            self._recv_stop_event.clear()
            self._recv_rollout_task = asyncio.create_task(
                self._recv_rollout_loop(input_channel)
            )

    async def _recv_rollout_loop(self, input_channel) -> None:
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        while not self._recv_stop_event.is_set():
            collector_round = []
            for _ in range(split_num):
                trajectory = await input_channel.get(async_op=True).async_wait()
                collector_round.append(trajectory)
            await self._received_collector_rounds.put(collector_round)
            self._collector_rounds_received += 1

    @staticmethod
    def _count_macro_transitions(trajectories: list[Any]) -> int:
        count = 0
        for trajectory in trajectories:
            rewards = getattr(trajectory, "rewards", None)
            if rewards is not None and rewards.ndim >= 2:
                count += int(rewards.shape[0] * rewards.shape[1])
        return count

    @staticmethod
    def _count_primitive_env_steps(trajectories: list[Any]) -> int:
        count = 0
        for trajectory in trajectories:
            forward_inputs = getattr(trajectory, "forward_inputs", None) or {}
            primitive_valid = forward_inputs.get("primitive_valid")
            if isinstance(primitive_valid, torch.Tensor):
                count += int(primitive_valid.to(torch.int64).sum().item())
            else:
                rewards = getattr(trajectory, "rewards", None)
                if isinstance(rewards, torch.Tensor):
                    count += int(rewards.numel())
        return count

    def _commit_collector_round(self, trajectories: list[Any]) -> None:
        """Atomically add one complete collector round to online/demo replay."""

        previous_transition_count = self._online_macro_transitions
        custom_ingest = getattr(self, "_ingest_rollout_trajectories", None)
        custom_counters = getattr(self, "_update_rollout_ingest_counters", None)
        if callable(custom_ingest) and callable(custom_counters):
            added, completed = custom_ingest(trajectories)
            custom_counters(added, completed)
            transition_count = int(added)
        else:
            self.replay_buffer.add_trajectories(trajectories)
            transition_count = self._count_macro_transitions(trajectories)
        if transition_count < 0:
            raise ValueError("Collector transition count must be non-negative")
        self._online_macro_transitions += transition_count
        self._primitive_env_steps += self._count_primitive_env_steps(trajectories)
        if self._utd_control_enabled:
            previous_eligible = max(
                0,
                previous_transition_count - self._learning_starts_macro_transitions,
            )
            current_eligible = max(
                0,
                self._online_macro_transitions
                - self._learning_starts_macro_transitions,
            )
            self._optimizer_update_budget += (
                current_eligible - previous_eligible
            ) * self._utd_ratio
        self._collector_rounds_committed += 1

        behavior_versions = []
        add_interventions = self.demo_buffer is not None and not callable(custom_ingest)
        intervene_traj_list = []
        for trajectory in trajectories:
            versions = getattr(trajectory, "versions", None)
            if versions is not None and versions.numel() > 0:
                behavior_versions.append(versions.detach().float().reshape(-1).cpu())
            if add_interventions:
                intervene_trajs = trajectory.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)
        if add_interventions and intervene_traj_list:
            self.demo_buffer.add_trajectories(intervene_traj_list)

        if behavior_versions:
            values = torch.cat(behavior_versions)
            self._latest_behavior_version_mean = float(values.mean().item())
            self._latest_behavior_version_min = float(values.min().item())

    def _drain_available_collector_rounds(self) -> int:
        """Drain queued collector rounds for legacy, unbounded async SAC."""

        committed = 0
        while True:
            try:
                trajectories = self._received_collector_rounds.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._commit_collector_round(trajectories)
            committed += 1
        return committed

    async def _wait_for_training_data(self, min_buffer_size: int) -> None:
        while True:
            if self._collector_control_enabled:
                has_budget = (
                    self._collector_rounds_committed > self._collector_rounds_consumed
                )
                if not has_budget:
                    trajectories = await self._received_collector_rounds.get()
                    self._commit_collector_round(trajectories)
                if self._utd_control_enabled:
                    return
            else:
                self._drain_available_collector_rounds()

            if await self.replay_buffer.is_ready_async(min_buffer_size):
                if not self._collector_control_enabled or (
                    self._collector_rounds_committed > self._collector_rounds_consumed
                ):
                    return
            await asyncio.sleep(0.1)

    def _available_optimizer_updates(self) -> int:
        """Return whole optimizer updates currently granted by the UTD budget."""

        if not self._utd_control_enabled:
            return 0
        granted_updates = math.floor(self._optimizer_update_budget + 1.0e-9)
        return max(0, granted_updates - self._optimizer_updates_completed)

    def _async_progress_metrics(self) -> dict[str, float]:
        eligible_transitions = max(
            0,
            self._online_macro_transitions - self._learning_starts_macro_transitions,
        )
        if self._utd_control_enabled:
            current_version = float(self._optimizer_updates_completed)
            optimizer_utd = self._optimizer_updates_completed / max(
                1, eligible_transitions
            )
        else:
            current_version = float(self._learner_rounds_completed)
            optimizer_utd = 0.0
        exploration_probability = min(
            self._online_macro_transitions
            / float(self._progressive_exploration_macro_steps),
            1.0,
        )
        return {
            "async/collector_step": float(self._collector_rounds_consumed),
            "async/learner_round": current_version,
            "async/critic_update_step": float(self.update_step),
            "async/pending_collector_rounds": float(
                self._collector_rounds_received - self._collector_rounds_consumed
            ),
            "async/online_macro_transitions": float(self._online_macro_transitions),
            "progress/env_step": float(self._online_macro_transitions),
            "progress/primitive_env_step": float(self._primitive_env_steps),
            "progress/collector_step": float(self._collector_rounds_consumed),
            "async/optimizer_update_budget": float(self._optimizer_update_budget),
            "async/optimizer_updates_completed": float(
                self._optimizer_updates_completed
            ),
            "async/optimizer_updates_pending": float(
                self._available_optimizer_updates()
            ),
            "async/optimizer_utd": float(optimizer_utd),
            "async/progressive_exploration_probability": float(exploration_probability),
            "async/policy_lag_mean": max(
                0.0, current_version - self._latest_behavior_version_mean
            ),
            "async/policy_lag_max": max(
                0.0, current_version - self._latest_behavior_version_min
            ),
        }

    def get_async_progress(self) -> dict[str, float]:
        """Return collector and learner progress without mutating worker state."""

        self._ensure_async_sac_state()
        return self._async_progress_metrics()

    def get_rollout_sync_version(self) -> int:
        """Version rollout weights by completed optimizer updates."""

        self._ensure_async_sac_state()
        if self._utd_control_enabled:
            return int(self._optimizer_updates_completed)
        return int(self._learner_rounds_completed)

    @Worker.timer("run_training")
    async def run_training(self):
        """Consume one collector round and its available SAC update budget."""

        self._ensure_async_sac_state()
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        await self._wait_for_training_data(min_buffer_size)

        metrics: dict[str, list[Any]] = {}
        if self._utd_control_enabled:
            optimizer_updates = self._available_optimizer_updates()
        else:
            learner_rounds = (
                self._max_learner_rounds_per_collector
                if self._collector_control_enabled
                else 1
            )
            optimizer_updates = learner_rounds * int(
                self.cfg.algorithm.get("update_epoch", 1)
            )

        if optimizer_updates > 0:
            torch.distributed.barrier()
            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            )
            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )
            if self._utd_control_enabled:
                # Replay readiness counts trajectories, whereas the v3 warmup
                # contract counts macro transitions.
                train_actor = (
                    self._online_macro_transitions
                    >= self._learning_starts_macro_transitions
                    and self.replay_buffer.is_ready(min_buffer_size)
                )
            else:
                train_actor_steps = max(
                    min_buffer_size,
                    int(self.cfg.algorithm.get("train_actor_steps", 0)),
                )
                train_actor = self.replay_buffer.is_ready(train_actor_steps)
            self.model.train()
            for _ in range(optimizer_updates):
                await asyncio.sleep(0)
                metrics_data = self.update_one_epoch(train_actor=train_actor)
                append_to_dict(metrics, metrics_data)
                self.update_step += 1
                if self._utd_control_enabled:
                    self._optimizer_updates_completed += 1
                    self.version = self._optimizer_updates_completed

            if not self._utd_control_enabled:
                self._learner_rounds_completed += learner_rounds
                self.version = self._learner_rounds_completed

        if self._collector_control_enabled:
            self._collector_rounds_consumed += 1
        append_to_dict(metrics, self._async_progress_metrics())
        mean_metric_dict = self.process_train_metrics(metrics)

        if optimizer_updates > 0:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()
        return mean_metric_dict

    def _async_state_path(self, base_path: str) -> str:
        return os.path.join(
            base_path,
            "sac_components",
            f"async_state_rank_{self._rank}.json",
        )

    def save_checkpoint(self, save_base_path, step) -> None:
        super().save_checkpoint(save_base_path, step)
        self._ensure_async_sac_state()
        state_path = self._async_state_path(save_base_path)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        state = {
            "contract_version": int(self.cfg.actor.model.get("contract_version", 0)),
            "collector_step": self._collector_rounds_consumed,
            "learner_round": self._learner_rounds_completed,
            "update_step": self.update_step,
            "policy_version": int(self.version),
            "online_macro_transitions": self._online_macro_transitions,
            "primitive_env_steps": self._primitive_env_steps,
            "optimizer_update_budget": self._optimizer_update_budget,
            "optimizer_updates_completed": self._optimizer_updates_completed,
            "utd_ratio": self._utd_ratio,
            "learning_starts_macro_transitions": (
                self._learning_starts_macro_transitions
            ),
            "progressive_exploration_macro_steps": (
                self._progressive_exploration_macro_steps
            ),
        }
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)

    def load_checkpoint(self, load_base_path) -> None:
        super().load_checkpoint(load_base_path)
        self._ensure_async_sac_state()
        state_path = self._async_state_path(load_base_path)
        if os.path.isfile(state_path):
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
            collector_step = int(state["collector_step"])
            self._collector_rounds_received = collector_step
            self._collector_rounds_committed = collector_step
            self._collector_rounds_consumed = collector_step
            self._learner_rounds_completed = int(state.get("learner_round", 0))
            self.update_step = int(state["update_step"])
            self.version = int(state["policy_version"])
            self._online_macro_transitions = int(
                state.get("online_macro_transitions", 0)
            )
            self._primitive_env_steps = int(state.get("primitive_env_steps", 0))
            self._optimizer_update_budget = float(
                state.get("optimizer_update_budget", 0.0)
            )
            self._optimizer_updates_completed = int(
                state.get("optimizer_updates_completed", 0)
            )
            invalid_counters = (
                collector_step < 0
                or self._online_macro_transitions < 0
                or self._primitive_env_steps < 0
                or self._optimizer_updates_completed < 0
                or self._optimizer_update_budget < 0.0
                or not math.isfinite(self._optimizer_update_budget)
                or self._optimizer_updates_completed
                > math.floor(self._optimizer_update_budget + 1.0e-9)
            )
            if invalid_counters:
                raise ValueError(
                    "Async SAC checkpoint contains invalid progress counters"
                )
            if self._utd_control_enabled:
                saved_contract = int(state.get("contract_version", 0))
                expected_contract = int(self.cfg.actor.model.get("contract_version", 0))
                if saved_contract != expected_contract:
                    raise ValueError(
                        "Async SAC checkpoint contract mismatch: "
                        f"expected v{expected_contract}, got v{saved_contract}"
                    )
                for key, expected in (
                    ("utd_ratio", self._utd_ratio),
                    (
                        "learning_starts_macro_transitions",
                        self._learning_starts_macro_transitions,
                    ),
                    (
                        "progressive_exploration_macro_steps",
                        self._progressive_exploration_macro_steps,
                    ),
                ):
                    if float(state.get(key, -1)) != float(expected):
                        raise ValueError(
                            f"Async SAC checkpoint {key} does not match config"
                        )
        elif self._utd_control_enabled:
            raise ValueError(
                "LAMP residual SAC cannot resume a checkpoint without "
                "sac_components/async_state_rank_*.json"
            )

    async def stop(self) -> None:
        self._ensure_async_sac_state()
        self._recv_stop_event.set()
        if self._recv_rollout_task is not None:
            self._recv_rollout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_rollout_task
        if getattr(self, "buffer_dataset", None) is not None:
            self.buffer_dataset.close()


class AsyncEmbodiedSACFSDPPolicy(
    AsyncSACExecutionMixin,
    EmbodiedSACFSDPPolicy,
):
    """Generic embodied SAC worker with asynchronous replay ingestion."""


__all__ = ["AsyncEmbodiedSACFSDPPolicy", "AsyncSACExecutionMixin"]
