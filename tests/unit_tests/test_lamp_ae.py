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

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn
from transformers import ResNetConfig

from rlinf.models.embodiment.lamp.artifact_io import save_artifact
from rlinf.models.embodiment.lamp.hand_ae import DexJoCoHandAE
from rlinf.models.embodiment.lamp.hand_prior_artifact import load_prior_artifact
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
    LAMPDiffusionPolicy,
)
from rlinf.workers.actor.lamp_il_worker import _prior_architecture

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
_TASK_STEPS = {
    "click_mouse": 20_000,
    "fold_glasses": 20_000,
    "hammer_nail": 30_000,
    "pick_bucket": 30_000,
    "pinch_tongs": 30_000,
    "water_plant": 30_000,
}


class _ConstantBackbone(nn.Module):
    def forward(self, image: torch.Tensor, *, train: bool | None = None):
        del train
        return image.new_zeros((image.shape[0], 512))


def _compose(config_name: str):
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(config_name=config_name)


def _tiny_ae_policy() -> LAMPDiffusionPolicy:
    backbone_config = ResNetConfig(
        depths=[1, 1, 1, 1],
        hidden_sizes=[64, 128, 256, 512],
    ).to_dict()
    policy = LAMPDiffusionPolicy(
        backbone_config=backbone_config,
        hand_prior_source="ae",
        ae_model_config={"hidden_dim": 16, "latent_dim": 2},
        core_action_mean=np.zeros(9, dtype=np.float32),
        core_action_std=np.ones(9, dtype=np.float32),
        hand_action_mean=np.zeros(16, dtype=np.float32),
        hand_action_std=np.ones(16, dtype=np.float32),
    )
    policy.front_backbone = _ConstantBackbone()
    policy.wrist_backbone = _ConstantBackbone()
    return policy


def test_hand_ae_reconstructs_future_chunks_with_masked_mse() -> None:
    torch.manual_seed(7)
    model = DexJoCoHandAE(hidden_dim=16, latent_dim=2)
    future = torch.randn(3, 16, 16)
    mask = torch.ones(3, 16)
    mask[1, 11:] = 0
    mask[2, 5:] = 0

    output = model(future, target_mask=mask)

    assert output.latent.shape == (3, 16, 2)
    assert output.prediction.shape == future.shape
    expected = ((output.prediction - future).square() * mask[..., None]).sum() / (
        mask.sum() * 16
    )
    torch.testing.assert_close(output.reconstruction_loss, expected)
    torch.testing.assert_close(output.total_loss, output.reconstruction_loss)
    output.total_loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_hand_ae_validates_future_latent_and_mask_shapes() -> None:
    model = DexJoCoHandAE(hidden_dim=8, latent_dim=2)
    with pytest.raises(ValueError, match="future must have shape"):
        model.encode(torch.zeros(2, 8, 16))
    with pytest.raises(ValueError, match="target_mask must have shape"):
        model.encode(torch.zeros(2, 16, 16), torch.ones(2, 8))
    with pytest.raises(ValueError, match="latent must have shape"):
        model.decode(torch.zeros(2, 8, 2))


def test_ae_prior_artifact_round_trip(tmp_path) -> None:
    model = DexJoCoHandAE(hidden_dim=8, latent_dim=2).eval()
    metadata = {
        "kind": "prior",
        "prior_type": "ae",
        "task": "pick_bucket",
        "dataset_fingerprint": "fingerprint",
        "hand_side": "single",
        "latent_dim": 2,
        "architecture": {"hidden_dim": 8, "latent_dim": 2},
    }
    save_artifact(tmp_path, model=model, metadata=metadata, statistics={})

    restored, restored_metadata, statistics = load_prior_artifact(
        tmp_path,
        expected_type="ae",
        expected_task="pick_bucket",
        expected_dataset_fingerprint="fingerprint",
        expected_hand_side="single",
    )

    assert isinstance(restored, DexJoCoHandAE)
    assert restored_metadata["latent_dim"] == 2
    assert statistics == {}
    future = torch.randn(2, 16, 16)
    torch.testing.assert_close(restored.encode(future), model.encode(future))


def test_ae_dp_condition_does_not_call_prior_encoder() -> None:
    policy = _tiny_ae_policy().train()
    assert not policy.ae.training
    assert not any(parameter.requires_grad for parameter in policy.ae.parameters())
    policy.ae.encode = mock.Mock(side_effect=AssertionError("encoder was called"))
    batch = 2

    condition, auxiliary = policy._encode_observation(
        torch.zeros(batch, 3, 8, 8),
        torch.zeros(batch, 3, 8, 8),
        torch.zeros(batch, 7),
        torch.randn(batch, 8, 16),
        train=False,
    )

    assert condition.shape == (batch, 256)
    assert auxiliary["hand_prior_feat"].shape == (batch, 128)
    assert auxiliary["mu_prior"].shape == (batch, 16, 2)
    assert policy.ae.encode.call_count == 0


def test_ae_dp_action_path_uses_only_decoder() -> None:
    policy = _tiny_ae_policy().eval()
    policy.ae.encode = mock.Mock(side_effect=AssertionError("encoder was called"))
    decode = mock.Mock(wraps=policy.ae.decode)
    policy.ae.decode = decode

    physical, auxiliary = policy._decode_core(torch.zeros(2, 16, 9))

    assert physical.shape == (2, 16, 23)
    assert auxiliary["latent_action"].shape == (2, 16, 2)
    assert decode.call_count == 1
    assert policy.ae.encode.call_count == 0


@pytest.mark.parametrize(("task", "max_steps"), _TASK_STEPS.items())
def test_ae_six_task_configs_match_reference_hyperparameters(
    task: str, max_steps: int
) -> None:
    prior_cfg = _compose(f"dexjoco_lamp_prior_ae_dim2_{task}")
    dp_cfg = _compose(f"dexjoco_lamp_dp_il_ae_{task}")

    assert prior_cfg.data.task_name == task
    assert prior_cfg.runner.max_steps == max_steps
    assert prior_cfg.actor.validation_interval == max_steps
    assert prior_cfg.actor.global_batch_size == 256
    assert prior_cfg.actor.micro_batch_size == 256
    assert prior_cfg.actor.eval_batch_size == 512
    assert prior_cfg.actor.optim.lr == 3e-4
    assert prior_cfg.actor.optim.min_lr == 1e-5
    assert prior_cfg.actor.optim.warmup_steps == 500
    assert prior_cfg.actor.model.hand_prior.type == "ae"
    assert _prior_architecture("ae", prior_cfg.actor.model.hand_prior) == {
        "hidden_dim": 1024,
        "latent_dim": 2,
    }

    assert dp_cfg.data.task_name == task
    assert dp_cfg.runner.max_steps == 30_000
    assert dp_cfg.actor.global_batch_size == 512
    assert dp_cfg.actor.micro_batch_size == 512
    assert dp_cfg.actor.eval_batch_size == 512
    assert dp_cfg.actor.optim.lr == 3e-5
    assert dp_cfg.actor.optim.warmup_steps == 1_000
    assert dp_cfg.actor.optim.backbone_lr_ratio == 0.1
    assert dp_cfg.actor.model.hand_prior.type == "ae"
    assert dp_cfg.actor.model.hand_prior.latent_dim == 2
    assert f"{task}_prior_ae_z2/artifact" in dp_cfg.actor.model.hand_prior.artifact_path
