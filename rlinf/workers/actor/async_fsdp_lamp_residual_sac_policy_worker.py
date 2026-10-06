# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LAMP macro accounting on the stable RealWorld asynchronous SAC worker."""

import asyncio
import json
import math
import queue
from pathlib import Path

from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.workers.actor.async_fsdp_sac_policy_worker import AsyncEmbodiedSACFSDPPolicy
from rlinf.workers.actor.fsdp_lamp_residual_sac_policy_worker import (
    LampResidualSACFSDPPolicy,
)


class AsyncLampResidualSACFSDPPolicy(
    AsyncEmbodiedSACFSDPPolicy, LampResidualSACFSDPPolicy
):
    """Reuse RealWorld receipt, mixed replay and optimizer updates for LAMP."""

    def _ensure_lamp_progress(self):
        if hasattr(self, "_online_macro_transitions"):
            return
        self._collector_step = 0
        self._online_macro_transitions = 0
        self._primitive_env_steps = 0
        self._optimizer_update_budget = 0.0
        self._optimizer_updates_completed = 0
        split = 1
        if hasattr(self, "_component_placement"):
            split = compute_split_num(
                self._component_placement.get_world_size("env") * self.stage_num,
                self._world_size,
            )
        self._recv_queue = queue.Queue(
            maxsize=split
            * int(
                self.cfg.algorithm.get("async", {}).get(
                    "max_pending_collector_rounds", 2
                )
            )
        )

    def init_worker(self):
        super().init_worker()
        if self._world_size != 1:
            raise ValueError(
                "LAMP transition-budget training currently requires one actor rank"
            )
        self._ensure_lamp_progress()

    def _drain_received_trajectories(self, max_trajectories=None):
        self._ensure_lamp_progress()
        trajectories = []
        while max_trajectories is None or len(trajectories) < max_trajectories:
            try:
                trajectories.append(self._recv_queue.get_nowait())
            except queue.Empty:
                break
        if not trajectories:
            return
        previous = self._online_macro_transitions
        added, _ = self._ingest_rollout_trajectories(trajectories)
        self._online_macro_transitions += added
        self._primitive_env_steps += sum(
            int(t.forward_inputs["primitive_valid"].sum()) for t in trajectories
        )
        start = int(self.cfg.algorithm.learning_starts_macro_transitions)
        newly_eligible = max(0, self._online_macro_transitions - start) - max(
            0, previous - start
        )
        self._optimizer_update_budget += newly_eligible * float(
            self.cfg.algorithm.utd_ratio
        )

    def get_lamp_progress(self):
        self._ensure_lamp_progress()
        return {
            "collector_step": self._collector_step,
            "online_macro_transitions": self._online_macro_transitions,
            "primitive_env_steps": self._primitive_env_steps,
            "optimizer_update_budget": self._optimizer_update_budget,
            "optimizer_updates_completed": self._optimizer_updates_completed,
        }

    async def run_training(self):
        self._ensure_lamp_progress()
        lockstep = bool(
            self.cfg.algorithm.get("async", {}).get("lockstep_updates", False)
        )
        if lockstep:
            count = compute_split_num(
                self._component_placement.get_world_size("env") * self.stage_num,
                self._world_size,
            )
            while self._recv_queue.qsize() < count and not self.should_stop:
                await asyncio.sleep(0.05)
            self._drain_received_trajectories(count)
            updates = (
                math.floor(self._optimizer_update_budget + 1e-9)
                - self._optimizer_updates_completed
            )
        else:
            while not self.should_stop:
                self._drain_received_trajectories(
                    self.cfg.actor.get("recv_drain_max_trajectories", 256)
                )
                updates = (
                    math.floor(self._optimizer_update_budget + 1e-9)
                    - self._optimizer_updates_completed
                )
                if updates > 0 and self.replay_buffer.is_ready(
                    int(self.cfg.algorithm.replay_buffer.min_buffer_size)
                ):
                    break
                await asyncio.sleep(0.05)
        if self.should_stop:
            return {}
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = self.cfg.actor.global_batch_size // (
            self.cfg.actor.micro_batch_size * self._world_size
        )
        self.model.train()
        metrics = {}
        for _ in range(updates):
            await asyncio.sleep(0)
            # This is the unmodified realenv-lamp SAC optimizer/mixing path.
            append_to_dict(metrics, self.update_one_epoch())
            self.update_step += 1
            self._optimizer_updates_completed += 1
        self._collector_step += 1
        for key, value in self.get_lamp_progress().items():
            metrics[f"async/{key}"] = [float(value)]
        return self.process_train_metrics(metrics)

    def save_checkpoint(self, save_base_path, step):
        super().save_checkpoint(save_base_path, step)
        path = Path(save_base_path) / f"lamp_async_rank_{self._rank}.json"
        path.write_text(json.dumps(self.get_lamp_progress(), indent=2) + "\n")

    def load_checkpoint(self, load_base_path):
        self._ensure_lamp_progress()
        path = Path(load_base_path) / f"lamp_async_rank_{self._rank}.json"
        legacy = (
            Path(load_base_path)
            / "sac_components"
            / f"async_state_rank_{self._rank}.json"
        )
        legacy_state = None
        if not path.exists() and legacy.exists():
            legacy_state = json.loads(legacy.read_text())
            if int(legacy_state["contract_version"]) != int(
                self.cfg.actor.model.contract_version
            ):
                raise ValueError("Legacy async LAMP contract differs")
            for key in (
                "utd_ratio",
                "learning_starts_macro_transitions",
                "progressive_exploration_macro_steps",
            ):
                if legacy_state[key] != self.cfg.algorithm[key]:
                    raise ValueError(f"Legacy async LAMP {key} differs")
            state = {key: legacy_state[key] for key in self.get_lamp_progress()}
        elif path.exists():
            state = json.loads(path.read_text())
        else:
            state = None
        if state is not None:
            if any(not math.isfinite(v) or v < 0 for v in state.values()):
                raise ValueError("Invalid LAMP async progress counters")
            if state["optimizer_updates_completed"] > math.floor(
                state["optimizer_update_budget"] + 1e-9
            ):
                raise ValueError("LAMP checkpoint exceeded its optimizer budget")
        super().load_checkpoint(load_base_path)
        if legacy_state is not None:
            self.update_step = int(legacy_state["update_step"])
            self.version = int(legacy_state["policy_version"])
        if state is not None:
            for key, value in state.items():
                setattr(self, "_" + key, value)


__all__ = ["AsyncLampResidualSACFSDPPolicy"]
