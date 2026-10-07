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

"""Check VQ export against forward reconstruction for both robot contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.lamp.hand_prior_artifact import sorted_vq_codebook
from rlinf.models.embodiment.lamp.hand_vq_vae import (
    HandVQVAE,
    decode_all_code_combinations,
    enumerate_code_indices,
)
from rlinf.models.embodiment.lamp.robot_spec import (
    dexjoco_robot_spec,
    resolve_robot_spec,
)
from rlinf.models.embodiment.lamp.vq_action_normalization import (
    denormalize_vq_hand_action,
)


@pytest.mark.parametrize("robot", ["dexjoco", "wuji"])
@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.35, 0.65)])
def test_export_preserves_forward_reconstruction(robot, weights):
    if robot == "dexjoco":
        spec = dexjoco_robot_spec()
    else:
        path = (
            Path(__file__).resolve().parents[2]
            / "examples/embodiment/config/robot/wuji_lamp.yaml"
        )
        spec = resolve_robot_spec(OmegaConf.to_container(OmegaConf.load(path)))

    model = HandVQVAE(
        action_dim=spec.hand_action_dim,
        latent_dim=spec.hand_action_dim,
        hidden_dim=16,
        layer_num=0,
    )
    # Orthogonal code directions make all 16 residual-VQ pairs reachable,
    # without relying on a randomly initialized encoder's code utilization.
    model.encoder = torch.nn.Identity()
    model.decoder = torch.nn.Identity()
    with torch.no_grad():
        model.quantizer.layer_weights.copy_(torch.tensor(weights).log())
        model.quantizer.codebooks.zero_()
        model.quantizer.codebooks[0, :, 0] = torch.linspace(-0.8, 0.8, 4)
        model.quantizer.codebooks[1, :, 1] = torch.linspace(-0.8, 0.8, 4)
    indices = enumerate_code_indices().long()
    actions = (
        model.quantizer.codebooks[0, indices[:, 0]]
        + model.quantizer.codebooks[1, indices[:, 1]]
    )
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.train()
    output = model(actions, training=True, update_ema=False)
    torch.testing.assert_close(output["indices"], indices)

    # Export must preserve what the decoder saw during training, including
    # nonuniform learned weights, rather than substituting equal weights.
    torch.testing.assert_close(
        model.decode_indices(output["indices"]), output["reconstruction"]
    )
    torch.testing.assert_close(
        decode_all_code_combinations(model), output["reconstruction"]
    )
    expected = denormalize_vq_hand_action(
        output["reconstruction"].detach().numpy(), spec
    )
    exported = sorted_vq_codebook(model, spec)
    # PCA sorting may reorder codes but must preserve every physical action.
    distances = np.square(exported[:, None] - expected[None]).sum(axis=-1)
    matched = distances.argmin(axis=1)
    assert len(np.unique(matched)) == 16
    np.testing.assert_allclose(exported, expected[matched], atol=1e-6)
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
