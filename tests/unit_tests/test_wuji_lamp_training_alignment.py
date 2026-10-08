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
    "prior_lamplstm_stack_cube": "dexjoco_lamp_prior_lamplstm_water_plant",
    "prior_lamplstm_none": "dexjoco_lamp_prior_lamplstm_water_plant",
    "prior_lamplstm_concat": "dexjoco_lamp_prior_lamplstm_water_plant",
    "prior_pca_stack_cube": "dexjoco_lamp_prior_pca_dim2_water_plant",
    "prior_vq_stack_cube": "dexjoco_lamp_prior_vq_water_plant",
    **{
        f"dp_{variant}": "dexjoco_lamp_dp_lamplstm"
        for variant in (
            "lamplstm",
            "lamplstm_none",
            "lamplstm_concat",
            "pca",
            "vq",
            "mlp",
        )
    },
}


@pytest.mark.parametrize("variant,reference", PAIRS.items())
def test_wuji_training_matches_simulation(variant, reference):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        actual = compose(config_name=f"realworld_lamp_{variant}")
        expected = compose(config_name=reference)
    for key in (
        "seed",
        "global_batch_size",
        "micro_batch_size",
        "eval_batch_size",
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
    # Keep the real-world data loader and robot contract across all recipes.
    assert not actual.actor.torch_compile
    assert actual.data.num_workers == 0
    assert not actual.data.persistent_workers
    assert (
        actual.data.source_factory == "rlinf.data.datasets.lamp.realworld:create_source"
    )
    assert actual.data.task_name == "wuji_stack_cube"
    assert actual.data.history_length == 8
    assert actual.data.horizon == 16
    assert actual.actor.model.action_dim == 26
    assert actual.actor.model.execution_horizon == 8
    assert actual.actor.model.num_action_chunks == 8
    assert actual.actor.model.robot_spec.arm_state_dim == 6
    assert actual.actor.model.robot_spec.hand_state_dim == 20
    assert actual.cluster.num_nodes == 2
    assert actual.cluster.component_placement.actor.node_group == "training_gpu"
    prior_type = variant.split("_")[1]
    prior = actual.actor.model.hand_prior
    assert prior.type == prior_type
    assert prior.latent_dim == {"vq": 1, "mlp": 6}.get(prior_type, 2)
    if variant.startswith("prior_") or prior_type == "lamplstm":
        for key, value in expected.actor.model.hand_prior.items():
            if key in ("action_dim", "history_dim"):
                value = 20
            elif key == "artifact_path":
                continue
            elif variant.endswith(("lamplstm_none", "lamplstm_concat")) and key in (
                "encoder_condition_mode",
                "decoder_condition_mode",
            ):
                value = variant.rsplit("_", 1)[-1]
            assert prior[key] == value, key
    if variant.startswith("dp_") and variant != "dp_lamplstm":
        raw = OmegaConf.load(CONFIG_DIR / f"realworld_lamp_{variant}.yaml")
        assert raw.defaults[0] == "realworld_lamp_dp_lamplstm"
        assert "optim" not in raw.actor
        assert "data" not in raw


def test_training_launcher_routes_all_artifacts():
    import shlex
    import subprocess

    if not (CONFIG_DIR.parents[2] / "scripts/train_wuji_lamp.sh").is_file():
        pytest.skip("Local training launcher is not tracked in Git")
    result = subprocess.run(
        ["bash", "scripts/train_wuji_lamp.sh", "all", "both", "--dry-run"],
        cwd=CONFIG_DIR.parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    commands = [
        shlex.split(line) for line in result.stdout.splitlines() if line.startswith(" ")
    ]
    assert len(commands) == 11
    prior_artifacts = set()
    for command in commands:
        name = command[command.index("--config-name") + 1]
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
            cfg = compose(config_name=name, overrides=command[4:])
        assert cfg.runner.logger.experiment_name == name
        if cfg.algorithm.stage == "prior":
            prior_artifacts.add(f"{cfg.runner.logger.log_path}/{name}/artifact")
        elif cfg.actor.model.hand_prior.type == "mlp":
            assert cfg.actor.model.hand_prior.artifact_path is None
        else:
            assert cfg.actor.model.hand_prior.artifact_path in prior_artifacts


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
