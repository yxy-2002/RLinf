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

import pytest
import torch
from torch import nn
from transformers import ResNetConfig

from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
    LAMPDiffusionPolicy,
)
from rlinf.workers.actor.lamp_il_worker import LampILWorker


def _policy(dropout_prob: float) -> LAMPDiffusionPolicy:
    return LAMPDiffusionPolicy(
        ResNetConfig(depths=[1, 1, 1, 1], hidden_sizes=[64, 128, 256, 512]).to_dict(),
        hand_prior_source="mlp",
        core_action_mean=torch.zeros(23),
        core_action_std=torch.ones(23),
        hand_action_mean=torch.zeros(16),
        hand_action_std=torch.ones(16),
        dropout_prob=dropout_prob,
    )


def test_dp_dropout_is_disabled_in_eval_and_enabled_in_train():
    model = _policy(0.1)
    dropout_modules = [
        module
        for module in model.condition_encoder.modules()
        if isinstance(module, nn.Dropout)
    ]
    assert len(dropout_modules) == 2
    assert all(module.p == 0.1 for module in dropout_modules)

    torch.manual_seed(7)
    model.train()
    image = torch.rand(2, 3, 8, 8)
    arm_pair = torch.zeros(2, 2, 7)
    hand_pair = torch.zeros(2, 2, 16)
    with torch.no_grad():
        first_train = model._encode_observation(
            image, image, arm_pair, hand_pair, train=True
        )[0]
        second_train = model._encode_observation(
            image, image, arm_pair, hand_pair, train=True
        )[0]
    assert not torch.equal(first_train, second_train)

    model.eval()
    with torch.no_grad():
        first_eval = model._encode_observation(
            image, image, arm_pair, hand_pair, train=False
        )[0]
        second_eval = model._encode_observation(
            image, image, arm_pair, hand_pair, train=False
        )[0]
    torch.testing.assert_close(first_eval, second_eval, rtol=0, atol=0)


def test_zero_dropout_preserves_existing_dp_architecture():
    model = _policy(0.0)
    assert model.dropout_prob == 0.0
    assert not [
        module
        for module in model.condition_encoder.modules()
        if isinstance(module, nn.Dropout)
    ]


def test_ema_update_uses_shadow_parameters_only():
    worker = object.__new__(LampILWorker)
    raw = nn.Linear(2, 2)
    ema = nn.Linear(2, 2)
    with torch.no_grad():
        raw.weight.zero_()
        raw.bias.zero_()
        ema.weight.fill_(1.0)
        ema.bias.fill_(1.0)
    worker.model = raw
    worker._ema_model = ema
    worker._ema_settings = {"enabled": True, "decay": 0.75, "start_step": 0}
    worker._global_step = 1
    worker._update_ema()
    torch.testing.assert_close(ema.weight, torch.full_like(ema.weight, 0.75))
    torch.testing.assert_close(ema.bias, torch.full_like(ema.bias, 0.75))
    assert raw.weight.eq(0).all() and raw.bias.eq(0).all()


def test_invalid_dp_dropout_is_rejected():
    with pytest.raises(ValueError, match="dropout_prob"):
        _policy(1.0)
