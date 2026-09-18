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
from omegaconf import OmegaConf

from rlinf.workers.actor.lamp_il_worker import LampILWorker


class FakeDPV2Model:
    def __init__(self) -> None:
        self.history = None
        self.history_mask = None

    def set_decoder_history(self, history, history_mask):
        self.history = history
        self.history_mask = history_mask

    def __call__(self, *args, **kwargs):
        predicted = torch.empty(2, 16, 23)
        predicted[..., :7] = 1.0
        predicted[..., 7:] = 2.0
        return predicted


def test_ddim_validation_metrics_are_masked_and_split_by_action_group():
    worker = object.__new__(LampILWorker)
    worker.stage = "dp"
    worker.device = torch.device("cpu")
    worker.model = FakeDPV2Model()
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "hand_prior": {"type": "lamplstm"},
                    "execution_horizon": 8,
                }
            }
        }
    )
    batch = {
        "front": torch.zeros(2, 3, 8, 8),
        "wrist": torch.zeros(2, 3, 8, 8),
        "arm_state_pair_norm": torch.zeros(2, 2, 7),
        "hand_state_pair_norm": torch.zeros(2, 2, 16),
        "target_action23": torch.zeros(2, 16, 23),
        "mask": torch.cat(
            (torch.ones(2, 8), torch.tensor([[1.0] * 8, [0.0] * 8])), dim=1
        ),
        "lamplstm_decoder_history_norm": torch.zeros(2, 8, 16),
        "lamplstm_decoder_history_mask": torch.ones(2, 8),
    }

    metrics = worker._ddim_validation_metrics(batch)

    assert set(metrics) == {
        "ddim_action_mse",
        "ddim_first_execution_horizon_mse",
        "ddim_arm_mse",
        "ddim_hand_mse",
    }
    assert metrics["ddim_arm_mse"] == pytest.approx(1.0)
    assert metrics["ddim_hand_mse"] == pytest.approx(4.0)
    assert metrics["ddim_action_mse"] == pytest.approx((7 * 1 + 16 * 4) / 23)
    assert metrics["ddim_first_execution_horizon_mse"] == pytest.approx(
        (7 * 1 + 16 * 4) / 23
    )
    assert worker.model.history is batch["lamplstm_decoder_history_norm"]
    assert worker.model.history_mask is batch["lamplstm_decoder_history_mask"]
