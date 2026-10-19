# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Select training behavior independently of the shared LAMP architecture."""

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

POLICY_DECORATOR = "policy_decorator"
REALWORLD_RLPD = "realworld_rlpd"


def training_recipe(algorithm: Mapping[str, Any]) -> str:
    """Resolve a recipe, preserving legacy Policy Decorator configurations."""
    recipe = algorithm.get("training_recipe", POLICY_DECORATOR)
    if recipe not in (POLICY_DECORATOR, REALWORLD_RLPD):
        raise ValueError(f"Unknown LAMP training_recipe: {recipe!r}")
    return recipe


def uses_policy_decorator(cfg: DictConfig) -> bool:
    """Whether this run needs the legacy macro budget and exploration clock."""
    return (
        cfg.actor.model.model_type == "lamp_residual_sac"
        and training_recipe(cfg.algorithm) == POLICY_DECORATOR
    )
