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

"""LAMP environment boundary, independent of simulator and hardware SDKs."""

from abc import ABC, abstractmethod
from importlib import import_module

import torch

from rlinf.models.embodiment.lamp.robot_spec import (
    dexjoco_robot_spec,
    resolve_robot_spec,
)


class LampEnvAdapter(ABC):
    """Wrap an RLinf environment in a declared LAMP command coordinate system.

    Implement reset/step/chunk_step without moving history updates to policy
    decision time. Return canonical measured observations, final observations
    before reset, and chunk infos containing executed_action [B,K,D] and
    primitive_valid [B,K]. Executed actions use the same coordinates as the
    input commands, including any clipping or intervention by the environment.
    """

    env_type: str

    @classmethod
    def validate_config(cls, env_cfg, robot_spec) -> None:
        if str(env_cfg.env_type) != cls.env_type:
            raise ValueError("LAMP adapter does not support this environment type")
        declared = env_cfg.get("lamp_robot_spec")
        if declared is None or resolve_robot_spec(declared) != robot_spec:
            raise ValueError("Environment lamp_robot_spec differs from model")

    def __init__(self, env, cfg, model_cfg) -> None:
        self.env = env
        self.robot_spec = resolve_robot_spec(model_cfg.get("robot_spec"))
        self.validate_config(cfg, self.robot_spec)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @abstractmethod
    def reset(self, **kwargs):
        """Reset and return canonical observations plus info."""

    @abstractmethod
    def step(self, actions, **kwargs):
        """Execute one primitive step and append its measured state history."""

    @abstractmethod
    def chunk_step(self, actions):
        """Execute a prefix and return primitive rewards, dones and feedback."""


class LampObservationHistory:
    """Maintain measured state pairs and left-padded per-environment history.

    Call update after reset and after every executed primitive step. Do not
    update rows that have already terminated inside the current chunk.
    """

    def __init__(self, robot_spec, num_envs: int, history_length: int) -> None:
        self.spec = resolve_robot_spec(robot_spec)
        if num_envs < 1 or history_length < 2:
            raise ValueError("History requires num_envs >= 1 and history_length >= 2")
        self.arm_pair = torch.zeros(num_envs, 2, self.spec.arm_state_dim)
        self.hand_history = torch.zeros(
            num_envs, history_length, self.spec.hand_state_dim
        )
        self.valid = torch.zeros(num_envs, dtype=torch.long)

    def update(self, arm_state, hand_state, indices, *, reset: bool = False) -> None:
        indices = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
        arm = torch.as_tensor(arm_state, dtype=torch.float32).cpu()
        hand = torch.as_tensor(hand_state, dtype=torch.float32).cpu()
        if arm.shape != (len(indices), self.spec.arm_state_dim) or hand.shape != (
            len(indices),
            self.spec.hand_state_dim,
        ):
            raise ValueError("Measured state shape differs from robot spec")
        if not torch.isfinite(arm).all() or not torch.isfinite(hand).all():
            raise ValueError("Measured states must be finite")
        if (
            indices.unique().numel() != len(indices)
            or (indices < 0).any()
            or (indices >= len(self.valid)).any()
        ):
            raise ValueError("Invalid history environment indices")
        if reset:
            self.arm_pair[indices] = arm[:, None]
            self.hand_history[indices] = hand[:, None]
            self.valid[indices] = 1
        else:
            if (self.valid[indices] == 0).any():
                raise ValueError("Reset history before appending a primitive step")
            self.arm_pair[indices, 0] = self.arm_pair[indices, 1]
            self.arm_pair[indices, 1] = arm
            self.hand_history[indices] = self.hand_history[indices].roll(-1, dims=1)
            self.hand_history[indices, -1] = hand
            self.valid[indices] = (self.valid[indices] + 1).clamp_max(
                self.hand_history.shape[1]
            )

    def observation(self) -> dict[str, torch.Tensor]:
        length = self.hand_history.shape[1]
        return {
            "arm_state_pair": self.arm_pair.clone(),
            "hand_state_pair": self.hand_history[:, -2:].clone(),
            "hand_history": self.hand_history.clone(),
            "hand_history_mask": (
                torch.arange(length)[None] >= length - self.valid[:, None]
            ).float(),
        }


def lamp_adapter_class(env_cfg):
    """Load only the explicitly configured adapter implementation."""
    target = env_cfg.get("lamp_adapter")
    if not target or ":" not in target:
        raise ValueError("LAMP requires an environment adapter (module:Class)")
    module, name = target.split(":", 1)
    adapter = getattr(import_module(module), name)
    if not isinstance(adapter, type) or not issubclass(adapter, LampEnvAdapter):
        raise ValueError("lamp_adapter must subclass LampEnvAdapter")
    return adapter


def validate_lamp_environment(env_cfg, model_cfg) -> None:
    """Reject unadapted environments before allocating workers."""
    spec = resolve_robot_spec(model_cfg.get("robot_spec"))
    if str(env_cfg.env_type) == "dexjoco":
        if spec != dexjoco_robot_spec():
            raise ValueError("Dexjoco requires its native Allegro robot spec")
        declared = env_cfg.get("lamp_robot_spec")
        if declared is not None and resolve_robot_spec(declared) != spec:
            raise ValueError("Environment lamp_robot_spec differs from model")
        return
    if model_cfg.get("robot_spec") is None:
        raise ValueError("Non-Dexjoco LAMP requires an explicit model robot_spec")
    lamp_adapter_class(env_cfg).validate_config(env_cfg, spec)


def apply_lamp_execution_feedback(forward_inputs, infos) -> None:
    """Update the retained replay command tensor in place after execution."""
    if (
        infos is None
        or "executed_action" not in infos
        or "primitive_valid" not in infos
    ):
        raise ValueError(
            "LAMP v5 requires executed_action and primitive_valid feedback"
        )
    proposed = forward_inputs["action"]
    executed = torch.as_tensor(
        infos["executed_action"], device=proposed.device, dtype=proposed.dtype
    )
    if executed.ndim == 3:
        executed = executed.flatten(start_dim=1)
    if executed.shape != proposed.shape or not torch.isfinite(executed).all():
        raise ValueError("Invalid LAMP executed_action feedback")
    valid = torch.as_tensor(infos["primitive_valid"], dtype=torch.bool)
    if valid.shape != forward_inputs["primitive_valid"].shape:
        raise ValueError("Invalid LAMP primitive_valid feedback")
    # ChunkStepResult already retains this tensor, so replacing the dict value
    # would leave the old policy proposal in replay.
    proposed.copy_(executed)
    forward_inputs["primitive_valid"] = valid.cpu().contiguous()
