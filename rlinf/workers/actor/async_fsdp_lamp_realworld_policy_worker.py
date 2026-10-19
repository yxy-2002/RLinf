# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native RealWorld RLPD scheduling with the shared LAMP networks and losses."""

import json
import queue
from pathlib import Path

from omegaconf import ListConfig

from rlinf.data.datasets.lamp.realworld import (
    RealWorldTrajectorySource,
    realworld_robot_spec,
)
from rlinf.data.datasets.lamp.realworld_replay import convert_demo_trajectories
from rlinf.workers.actor.async_fsdp_sac_policy_worker import AsyncEmbodiedSACFSDPPolicy
from rlinf.workers.actor.fsdp_lamp_residual_sac_policy_worker import (
    LampResidualSACFSDPPolicy,
)


class AsyncLampRealWorldRLPDPolicy(
    AsyncEmbodiedSACFSDPPolicy, LampResidualSACFSDPPolicy
):
    """Use native receipt/readiness/update_epoch; counters never grant updates.

    The MRO keeps the native async run_training and stop methods, and LAMP's
    model, loss, executed-trajectory validation and replay checkpoint methods.
    """

    def _load_demo_buffer(self, load_path: str | list[str] | ListConfig) -> None:
        """Adapt raw recordings directly into the native demo replay buffer."""
        self._raw_demo_identity = None
        if (
            isinstance(load_path, str)
            and (Path(load_path) / "lamp_demo_manifest.json").is_file()
        ):
            # Existing preconverted buffers remain loadable.
            super()._load_demo_buffer(load_path)
            return
        roots = (
            list(load_path) if isinstance(load_path, (list, ListConfig)) else load_path
        )
        policy = self._unwrap_policy(self.model)
        robot_spec = policy.base_policy.spec.robot_spec
        robot_type = next(
            (
                name
                for name in ("wuji", "ruiyan")
                if realworld_robot_spec(name) == robot_spec
            ),
            None,
        )
        if robot_type is None:
            raise ValueError("Raw RealWorld demos require a Wuji or Ruiyan robot spec")
        source = RealWorldTrajectorySource(roots, robot_type=robot_type)
        history_length = int(self.cfg.env.train.lamp_history_length)
        # Match native replay loading's contiguous per-rank episode partition.
        count, remainder = divmod(len(source.paths), self._world_size)
        start = self._rank * count + min(self._rank, remainder)
        stop = start + count + int(self._rank < remainder)
        if start == stop:
            raise ValueError(
                "Raw demo data must contain at least one episode per actor rank"
            )
        for trajectory in convert_demo_trajectories(
            source,
            policy,
            history_length=history_length,
            episodes=list(range(start, stop)),
        ):
            self.demo_buffer.add_trajectories([trajectory])
        # Source identity is independent of the frozen base weights. The normal
        # training checkpoint contract separately checks model compatibility.
        self._raw_demo_identity = (
            f"raw_realworld_v1:{source.metadata.data_sha256}:history={history_length}"
        )

    def _demo_training_identity(self, policy) -> str | None:
        """Use raw data identity without requiring an external conversion manifest."""
        identity = getattr(self, "_raw_demo_identity", None)
        if identity is not None:
            return identity
        return super()._demo_training_identity(policy)

    def _ensure_realworld_progress(self) -> None:
        if not hasattr(self, "_realworld_progress"):
            self._realworld_progress = {
                "training_rounds": 0,
                "online_macro_transitions": 0,
                "primitive_env_steps": 0,
            }

    def _drain_received_trajectories(self, max_trajectories: int | None = None) -> None:
        self._ensure_realworld_progress()
        if getattr(self, "_recv_queue", None) is None:
            return
        trajectories = []
        while max_trajectories is None or len(trajectories) < max_trajectories:
            try:
                trajectories.append(self._recv_queue.get_nowait())
            except queue.Empty:
                break
        if not trajectories:
            return
        added, _ = self._ingest_rollout_trajectories(trajectories)
        self._realworld_progress["online_macro_transitions"] += added
        self._realworld_progress["primitive_env_steps"] += sum(
            int(t.forward_inputs["primitive_valid"].sum()) for t in trajectories
        )

    def process_train_metrics(self, metrics: dict) -> dict:
        """Report native training rounds and updates separately from collection."""
        self._ensure_realworld_progress()
        self._realworld_progress["training_rounds"] += 1
        for key, value in self._realworld_progress.items():
            metrics[f"realworld/{key}"] = [float(value)]
        metrics["realworld/critic_updates"] = [float(self.update_step)]
        metrics["realworld/actor_updates"] = [
            float(
                (self.update_step + self.critic_actor_ratio - 1)
                // self.critic_actor_ratio
            )
        ]
        metrics["realworld/demo_sample_fraction"] = [
            0.5 if self.demo_buffer is not None else 0.0
        ]
        return super().process_train_metrics(metrics)

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        """Persist observational counters alongside the shared replay state."""
        self._ensure_realworld_progress()
        super().save_checkpoint(save_base_path, step)
        path = Path(save_base_path) / f"lamp_realworld_rank_{self._rank}.json"
        path.write_text(json.dumps(self._realworld_progress, indent=2) + "\n")

    def load_checkpoint(self, load_base_path: str) -> None:
        """Restore only native-recipe progress, never a macro update budget."""
        self._ensure_realworld_progress()
        path = Path(load_base_path) / f"lamp_realworld_rank_{self._rank}.json"
        if not path.is_file():
            raise ValueError(
                "RealWorld RLPD requires a native-recipe checkpoint; "
                "start a new run from the base policy artifact"
            )
        state = json.loads(path.read_text())
        if set(state) != set(self._realworld_progress) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in state.values()
        ):
            raise ValueError("Invalid RealWorld RLPD progress counters")
        super().load_checkpoint(load_base_path)
        self._realworld_progress = state
