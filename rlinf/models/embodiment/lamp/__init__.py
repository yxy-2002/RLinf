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

"""LAMP single-arm diffusion and residual policies for DexJoCo."""

from __future__ import annotations

import copy
from dataclasses import fields

import torch
from omegaconf import DictConfig, open_dict

from rlinf.models.embodiment.lamp.artifact_io import load_artifact
from rlinf.models.embodiment.lamp.policy_wrapper import LampPolicy, LampPolicySpec
from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import LAMPDiffusionPolicy


def get_model(cfg: DictConfig, torch_dtype: torch.dtype = torch.float32) -> LampPolicy:
    """Build a deployment policy from a native self-contained artifact."""

    del torch_dtype
    artifact_path = cfg.get("model_path", None)
    if not artifact_path:
        raise ValueError(
            "LAMP rollout.model.model_path must point to an artifact directory"
        )
    metadata, state, statistics = load_artifact(artifact_path)
    if str(cfg.model_type) != "lamp_dp":
        raise ValueError("Only lamp_dp deployment is supported")
    if (
        metadata.get("model_type") not in ("lamp_dp", "lamp_dp_v2")
        or metadata.get("policy_version") != 2
    ):
        raise ValueError("LAMP requires a version-2 DP deployment artifact")
    if metadata.get("spec", {}).get("embodiment") != "single":
        raise ValueError("LAMP supports single-arm deployment only")
    architecture = metadata.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("LAMP artifact is missing architecture")
    if architecture.get("hand_prior_source") == "lamplstm" and (
        architecture.get("decoder_history_contract") != "primitive_v1"
        or architecture.get("decoder_history_length") is None
    ):
        raise ValueError(
            "LSTM artifact requires decoder_history_contract=primitive_v1 "
            "and an explicit decoder_history_length"
        )
    core = LAMPDiffusionPolicy(**architecture)
    spec_payload = metadata.get("spec")
    required = {item.name for item in fields(LampPolicySpec)}
    if isinstance(spec_payload, dict) and "policy_version" not in spec_payload:
        spec_payload = dict(spec_payload)
        spec_payload["policy_version"] = 1
    if not isinstance(spec_payload, dict) or set(spec_payload) != required:
        raise ValueError("LAMP artifact has an invalid policy spec")
    spec_payload = dict(spec_payload)
    spec_payload["image_keys"] = tuple(spec_payload["image_keys"])
    spec = LampPolicySpec(**spec_payload)
    configured_prior = cfg.get("hand_prior", None)
    if configured_prior is not None:
        configured_type = str(configured_prior.get("type", spec.hand_prior_type))
        if configured_type != spec.hand_prior_type:
            raise ValueError("Configured LAMP hand prior differs from the artifact")
        if (
            spec.embodiment == "single"
            and configured_prior.get("latent_dim") is not None
        ):
            expected_dim = spec.latent_dims.get("single", 0)
            if int(configured_prior.latent_dim) != expected_dim:
                raise ValueError("Configured LAMP latent_dim differs from the artifact")
    policy = LampPolicy(
        core,
        spec,
        statistics,
        temporal_ensemble_decay=float(cfg.get("temporal_ensemble_decay", 0.25)),
        use_temporal_ensemble=bool(cfg.get("use_temporal_ensemble", True)),
        execution_horizon_override=cfg.get("execution_horizon_override", None),
        num_inference_steps_override=cfg.get("num_inference_steps_override", None),
        eval_base_noise_seed=cfg.get("eval_base_noise_seed", None),
        eval_base_noise_seeds=cfg.get("eval_base_noise_seeds", None),
        eval_base_noise_seed_offset=int(cfg.get("eval_base_noise_seed_offset", 0)),
    )
    policy.load_state_dict(state, strict=True)
    return policy.float()


def get_residual_model(
    cfg: DictConfig, torch_dtype: torch.dtype = torch.float32
) -> LampResidualSACPolicy:
    """Build online residual SAC around a frozen native LAMP DP artifact."""

    contract_version = int(cfg.get("contract_version", 4))
    if contract_version != 4:
        raise ValueError("LAMP residual model supports only contract_version=4")
    residual_application = str(cfg.get("residual_application", "corrected_plan_crop"))
    base_use_temporal_ensemble = cfg.get("base_use_temporal_ensemble", False)
    if not isinstance(base_use_temporal_ensemble, bool):
        raise ValueError("base_use_temporal_ensemble must be a boolean")

    base_cfg = copy.deepcopy(cfg)
    with open_dict(base_cfg):
        base_cfg.model_type = "lamp_dp"
        base_cfg.use_temporal_ensemble = base_use_temporal_ensemble
        base_cfg.execution_horizon_override = int(cfg.num_action_chunks)
    base_policy = get_model(base_cfg, torch_dtype=torch_dtype)
    residual_scale = float(cfg.get("residual_scale", 0.05))
    return LampResidualSACPolicy(
        base_policy,
        num_q_heads=int(cfg.get("num_q_heads", 2)),
        residual_scale=residual_scale,
        wrist_residual_scale=float(cfg.get("wrist_residual_scale", residual_scale)),
        hand_residual_scale=float(cfg.get("hand_residual_scale", residual_scale)),
        actor_hidden_dims=tuple(cfg.get("actor_hidden_dims", (256, 256, 256))),
        log_std_min=float(cfg.get("log_std_min", -20.0)),
        log_std_max=float(cfg.get("log_std_max", 2.0)),
        init_log_std=float(cfg.get("init_log_std", -9.0)),
        eval_base_noise_seed=cfg.get("eval_base_noise_seed", None),
        eval_base_noise_seeds=cfg.get("eval_base_noise_seeds", None),
        eval_base_noise_seed_offset=int(cfg.get("eval_base_noise_seed_offset", 0)),
        contract_version=contract_version,
        residual_application=residual_application,
        base_use_temporal_ensemble=base_use_temporal_ensemble,
        actor_input=str(cfg.get("actor_input", "condition")),
        critic_observation_input=str(cfg.get("critic_observation_input", "condition")),
        entropy_scope=str(cfg.get("entropy_scope", "decoder_causal")),
        target_entropy=cfg.get("target_entropy", None),
        learning_starts_macro_transitions=int(
            cfg.get("learning_starts_macro_transitions", 8000)
        ),
        progressive_exploration_macro_steps=int(
            cfg.get("progressive_exploration_macro_steps", 30000)
        ),
    ).float()


__all__ = [
    "LampPolicy",
    "LampPolicySpec",
    "LampResidualSACPolicy",
    "get_model",
    "get_residual_model",
]
