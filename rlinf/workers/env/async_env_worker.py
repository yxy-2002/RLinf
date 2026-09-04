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

import torch
from omegaconf.omegaconf import DictConfig

from rlinf.scheduler import Channel, Worker
from rlinf.workers.env.env_worker import EnvWorker


class AsyncCollectorGate:
    """Bound collector progress without interrupting an in-flight round."""

    def __init__(self) -> None:
        self.collector_step = 0
        self.limit_step: int | None = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    def configure(self, collector_step: int, limit_step: int | None) -> None:
        if collector_step < 0:
            raise ValueError("collector_step must be non-negative")
        if limit_step is not None and limit_step < collector_step:
            raise ValueError("collector limit cannot precede collector progress")
        self.collector_step = collector_step
        self.limit_step = limit_step
        self._refresh_event()

    def advance(self, limit_step: int | None) -> None:
        if limit_step is not None and limit_step < self.collector_step:
            raise ValueError("collector limit cannot precede collector progress")
        if (
            self.limit_step is not None
            and limit_step is not None
            and limit_step < self.limit_step
        ):
            raise ValueError("collector limit must be monotonic")
        self.limit_step = limit_step
        self._refresh_event()

    def mark_collected(self) -> None:
        self.collector_step += 1
        self._refresh_event()

    async def wait_for_permit(self) -> None:
        while self.limit_step is not None and self.collector_step >= self.limit_step:
            self._resume_event.clear()
            if self.limit_step is None or self.collector_step < self.limit_step:
                self._resume_event.set()
                return
            await self._resume_event.wait()

    async def wait_until_collected(self, collector_step: int) -> None:
        while self.collector_step < collector_step:
            await asyncio.sleep(0.05)

    def _refresh_event(self) -> None:
        if self.limit_step is None or self.collector_step < self.limit_step:
            self._resume_event.set()


class AsyncEnvWorker(EnvWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._interact_task: asyncio.Task = None
        self._collector_gate = AsyncCollectorGate()
        self._online_macro_transitions = 0
        # Every env rank tracks the same global counter; v3 fixes pipeline
        # stages to one, so one synchronized dispatch executes every train env.
        self._macro_transitions_per_dispatch = int(cfg.env.train.total_num_envs)
        assert not (self.train_enable_offload or self.eval_enable_offload), (
            "Offload not supported in AsyncEnvWorker"
        )

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        metric_channel: Channel,
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        self._interact_task = asyncio.create_task(
            self._interact(
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
                metric_channel,
            )
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        metric_channel: Channel,
    ):
        while True:
            await self._collector_gate.wait_for_permit()
            env_metrics = await self._run_interact_once(
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
                cooperative_yield=True,
            )

            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            env_interact_time_metrics = self.pop_execution_times()
            env_interact_time_metrics = {
                f"time/env/{k}": v for k, v in env_interact_time_metrics.items()
            }
            metrics = {
                "rank": self._rank,
                "env": env_metrics,
                "time": env_interact_time_metrics,
            }
            metric_channel.put(metrics, async_op=True)
            self._collector_gate.mark_collected()

    def _build_rollout_input_data(self, env_batch):
        """Attach the exact global count before the requested action executes."""

        data = super()._build_rollout_input_data(env_batch)
        obs = data["obs"]
        reference = next(
            value
            for value in obs.values()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        )
        obs["online_macro_transitions"] = torch.full(
            (reference.shape[0],),
            self._online_macro_transitions,
            dtype=torch.long,
            device=reference.device,
        )
        return data

    def env_interact_step(self, chunk_actions, stage_id):
        """Count macro transitions only after a chunk is dispatched to the env."""

        result = super().env_interact_step(chunk_actions, stage_id)
        self._online_macro_transitions += self._macro_transitions_per_dispatch
        return result

    async def configure_collector_window(
        self,
        collector_step: int,
        limit_step: int,
        online_macro_transitions: int = 0,
    ) -> None:
        """Initialize collector progress and the maximum permitted round."""

        if online_macro_transitions < 0:
            raise ValueError("Online macro transition count must be non-negative")
        self._online_macro_transitions = int(online_macro_transitions)
        self._collector_gate.configure(collector_step, limit_step)

    def get_online_macro_transitions(self) -> int:
        """Return the exact transition count used by progressive exploration."""

        return self._online_macro_transitions

    async def advance_collector_limit(self, limit_step: int) -> None:
        """Permit collection through ``limit_step`` (exclusive next round)."""

        self._collector_gate.advance(limit_step)

    async def wait_for_collector_step(self, collector_step: int) -> None:
        """Wait until the current in-flight collector round reaches a boundary."""

        await self._collector_gate.wait_until_collected(collector_step)

    async def stop(self):
        self._collector_gate.advance(None)
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()


__all__ = ["AsyncCollectorGate", "AsyncEnvWorker"]
