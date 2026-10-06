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

"""Regression checks for Wuji training parity and read-only cache access."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.data.datasets.lamp.offline_dataset import LampMMapDataset

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
PAIRS = {
    "prior_lamplstm_film": "dexjoco_lamp_prior_lamplstm_water_plant",
    "prior_lamplstm_none": "dexjoco_lamp_prior_lamplstm_water_plant",
    "prior_pca": "dexjoco_lamp_prior_pca",
    "prior_vq": "dexjoco_lamp_prior_vq_water_plant",
    "dp_lamplstm_film": "dexjoco_lamp_dp_lamplstm",
    "dp_lamplstm_none": "dexjoco_lamp_dp_lamplstm",
    "dp_pca": "dexjoco_lamp_dp_il_pca_water_plant",
    "dp_vq": "dexjoco_lamp_dp_il_vq_water_plant",
    "dp_mlp": "dexjoco_lamp_dp_il_mlp_water_plant",
}


@pytest.mark.parametrize("variant,reference", PAIRS.items())
def test_wuji_training_matches_simulation(variant, reference):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        actual = compose(config_name=f"realworld_lamp_{variant}_stack_cube")
        expected = compose(config_name=reference)
    for key in (
        "seed",
        "global_batch_size",
        "micro_batch_size",
        "eval_batch_size",
        "torch_compile",
        "compile_mode",
        "validation_interval",
        "validation_batches",
        "validate_at_end",
        "optim",
    ):
        assert actual.actor.get(key) == expected.actor.get(key), key
    for key in (
        "max_steps",
        "max_epochs",
        "local_update_steps",
        "save_interval",
        "log_interval",
        "val_check_interval",
        "save_every_epochs",
    ):
        assert actual.runner.get(key) == expected.runner.get(key), key
    for key in ("num_workers", "persistent_workers", "pin_memory", "image_size"):
        assert actual.data[key] == expected.data[key], key
    for key in ("execution_horizon", "num_action_chunks"):
        assert actual.actor.model[key] == expected.actor.model[key], key
    prior = OmegaConf.to_container(expected.actor.model.hand_prior, resolve=True)
    for key, value in prior.items():
        if key in ("action_dim", "history_dim"):
            value = 20
        elif key == "artifact_path":
            continue
        elif variant.endswith("_none") and key in (
            "encoder_condition_mode",
            "decoder_condition_mode",
        ):
            value = "none"
        assert actual.actor.model.hand_prior[key] == value, key
    assert actual.data.history_length == 8
    assert actual.actor.model.hand_prior.history_length == 8
    assert actual.actor.model.robot_spec.arm_state_dim == 6
    assert actual.actor.model.robot_spec.hand_state_dim == 20
    assert actual.cluster.num_nodes == 2


@pytest.mark.parametrize("fortran", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.bool_])
def test_readonly_mmap_samples_are_independent(tmp_path, fortran, dtype):
    split = tmp_path / "train"
    split.mkdir()
    values = (np.arange(96).reshape(8, 3, 4) % 2).astype(dtype)
    if fortran:
        values = np.asfortranarray(values)
    path = split / "state.npy"
    np.save(path, values)
    original = path.read_bytes()
    dataset = LampMMapDataset(tmp_path, "train", ["state"])
    for index in range(len(dataset)):
        sample = dataset[index]["state"]
        np.testing.assert_array_equal(sample.numpy(), values[index])
        assert sample.numpy().flags.writeable
        assert not np.shares_memory(sample.numpy(), dataset._open_arrays()["state"])
        sample.zero_()
        np.testing.assert_array_equal(dataset[index]["state"].numpy(), values[index])
    restored = pickle.loads(pickle.dumps(dataset))
    loader = torch.utils.data.DataLoader(
        restored, batch_size=2, num_workers=1, multiprocessing_context="spawn"
    )
    actual = torch.cat([batch["state"] for batch in loader]).numpy()
    np.testing.assert_array_equal(actual, values)
    assert path.read_bytes() == original
    assert not dataset._open_arrays()["state"].flags.writeable
