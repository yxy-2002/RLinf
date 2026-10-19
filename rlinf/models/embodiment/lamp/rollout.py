# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""LAMP adaptation to the existing generic rollout worker contract."""

from typing import Literal, TypeVar

import torch

from rlinf.models.embodiment.base_policy import BasePolicy

Policy = TypeVar("Policy", bound=BasePolicy)


def configure_rollout_policy(policy: Policy) -> Policy:
    """Partition evaluation noise streams for a rollout copy.

    Actor and standalone model construction are unchanged. Resolve the current
    worker only at factory invocation, after worker initialization; never mutate
    its config or sampling state. Both sync and async rollouts use this path.
    """
    from rlinf.scheduler import Worker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    worker = Worker.current_worker
    if not isinstance(worker, MultiStepRolloutWorker):
        return policy
    if worker.enable_eval:
        offset = int(policy.eval_base_noise_seed_offset)
        rank = int(worker._rank)
        batch_size = int(worker.per_node_eval_batch_size)
        if min(offset, rank, batch_size) < 0:
            raise ValueError("LAMP eval noise offset inputs must be non-negative")
        policy.set_eval_base_noise_seed_offset(offset + rank * batch_size)
    return policy


def resolve_rollout_mode(
    mode: str | None, do_sample: bool | None, *, default: str
) -> str:
    """Translate generic worker sampling flags while preserving direct calls."""
    if mode is None:
        mode = default if do_sample is None else "train" if do_sample else "eval"
    if mode not in ("train", "eval"):
        raise ValueError(f"Unsupported LAMP rollout mode: {mode!r}")
    return mode


def residual_exploration(
    *,
    recipe: str,
    mode: str,
    count: int | float | torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    learning_starts: int,
    progressive_steps: int,
) -> tuple[torch.Tensor | None, float, Literal["actor", "uniform"]]:
    """Choose rollout exploration without changing the actor or critic network.

    RealWorld always samples the residual actor in training. Policy Decorator
    retains its global macro-count gate and uniform warmup. Evaluation uses
    the deterministic actor for both recipes.
    """
    from .residual_sac import LampResidualSACPolicy
    from .training_recipe import training_recipe

    recipe = training_recipe({"training_recipe": recipe})
    if mode != "train" or recipe == "realworld_rlpd":
        return None, 1.0, "actor"
    count = LampResidualSACPolicy._counter_value(count)
    probability = min(count / float(progressive_steps), 1.0)
    enabled = torch.rand(batch_size, device=device) < probability
    return enabled, probability, "uniform" if count < learning_starts else "actor"
