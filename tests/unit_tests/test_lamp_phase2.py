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

import pickle
from dataclasses import asdict

import av
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import ResNetConfig

from rlinf.data.datasets.lamp.dexjoco_lerobot import decode_video_rows
from rlinf.data.datasets.lamp.offline_dataset import LampMMapDataset
from rlinf.models.embodiment.lamp import get_model as get_lamp_model
from rlinf.models.embodiment.lamp.artifact_io import save_artifact
from rlinf.models.embodiment.lamp.bc_policy import BCPolicy
from rlinf.models.embodiment.lamp.policy_wrapper import (
    LampPolicy,
    LampPolicySpec,
    LampTemporalEnsembleController,
)
from rlinf.models.embodiment.lamp.hand_prior_artifact import load_prior_artifact
from rlinf.models.embodiment.lamp.hand_vae import DexJoCoHandVAE
from rlinf.models.embodiment.lamp.hand_vq_vae import HandVQVAE
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.workers.actor.lamp_il_worker import DeterministicInfiniteBatchSampler


def test_offline_runner_preserves_standard_metric_namespaces():
    metrics = OfflineRunner._training_metric_namespaces(
        {
            "total_loss": 1.0,
            "validation/total_loss": 2.0,
            "data/samples_per_second": 3.0,
            "time/update_seconds_per_step": 4.0,
        }
    )
    assert metrics == {
        "train/total_loss": 1.0,
        "validation/total_loss": 2.0,
        "data/samples_per_second": 3.0,
        "time/update_seconds_per_step": 4.0,
    }


def test_video_cache_decode_is_nhwc_uint8(tmp_path):
    video_path = tmp_path / "videos" / "front" / "chunk-000" / "file-000.mp4"
    video_path.parent.mkdir(parents=True)
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 8
        stream.height = 8
        stream.pix_fmt = "yuv420p"
        for pixel_value in (32, 192):
            frame = av.VideoFrame.from_ndarray(
                np.full((8, 8, 3), pixel_value, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    images = decode_video_rows(tmp_path, "front", np.asarray([0, 1]), image_size=4)

    assert images.shape == (2, 4, 4, 3)
    assert images.dtype == np.uint8
    assert images[1].mean() > 100


def test_mmap_dataset_spawn_payload_does_not_copy_array_contents(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    values = np.arange(1_000_000, dtype=np.float32).reshape(1000, 1000)
    np.save(split_dir / "state.npy", values, allow_pickle=False)

    dataset = LampMMapDataset(tmp_path, "train", ("state",))
    np.testing.assert_array_equal(dataset[7]["state"].numpy(), values[7])
    payload = pickle.dumps(dataset)

    assert len(payload) < 10_000
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(restored[13]["state"].numpy(), values[13])


def test_configurable_vae_latent_dim_and_artifact_round_trip(tmp_path):
    model = DexJoCoHandVAE(latent_dim=5, hidden_dim=32)
    history = torch.randn(2, 8, 16)
    target = torch.randn(2, 16)
    output = model(history, target)
    assert output.mu.shape == (2, 5)
    output.total_loss.backward()

    statistics = {
        "hand_history_mean": np.zeros(16, np.float32),
        "hand_history_std": np.ones(16, np.float32),
    }
    metadata = {
        "kind": "prior",
        "prior_type": "vae",
        "task": "pick_bucket",
        "dataset_fingerprint": "fixture",
        "hand_side": "single",
        "latent_dim": 5,
        "architecture": {
            "backbone": "cnn",
            "hidden_dim": 32,
            "beta": 1e-4,
            "latent_dim": 5,
        },
        "statistics_keys": sorted(statistics),
    }
    save_artifact(tmp_path, model=model, metadata=metadata, statistics=statistics)
    restored, restored_metadata, restored_statistics = load_prior_artifact(
        tmp_path,
        expected_type="vae",
        expected_task="pick_bucket",
        expected_dataset_fingerprint="fixture",
        expected_hand_side="single",
    )
    assert restored.latent_dim == 5
    assert restored_metadata["latent_dim"] == 5
    assert set(restored_statistics) == set(statistics)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_vq_ema_can_be_applied_outside_compiled_forward():
    model = HandVQVAE(latent_dim=16, hidden_dim=32, layer_num=1)
    actions = torch.randn(8, 16)
    before = model.quantizer.codebooks.clone()
    output = model(actions, training=True, update_ema=False)
    torch.testing.assert_close(model.quantizer.codebooks, before)
    output["total_loss"].backward()
    model.quantizer.apply_ema_updates(output["ema_counts"], output["ema_sums"])
    assert torch.isfinite(model.quantizer.codebooks).all()
    assert not torch.equal(model.quantizer.codebooks, before)


def test_temporal_ensemble_aligns_quaternion_sign_and_resets_rows():
    controller = LampTemporalEnsembleController(execution_horizon=4, decay=0.25)
    first = torch.zeros(1, 16, 23)
    first[..., 3] = 1.0
    first[..., 7:] = 1.0
    action1 = controller.apply(first)
    torch.testing.assert_close(action1[..., 3], torch.ones(1, 4))

    second = torch.zeros_like(first)
    second[..., 3] = -1.0
    second[..., 7:] = 3.0
    action2 = controller.apply(second)
    assert torch.all(action2[..., 3].abs() > 0.999)
    assert torch.all((action2[..., 7:] > 1.0) & (action2[..., 7:] < 3.0))

    reset_action = controller.apply(second, reset_mask=torch.tensor([True]))
    torch.testing.assert_close(reset_action[..., 7:], torch.full((1, 4, 16), 3.0))

    zero_action = controller.apply(
        torch.zeros_like(first), reset_mask=torch.tensor([True])
    )
    torch.testing.assert_close(zero_action[..., 3], torch.ones(1, 4))
    torch.testing.assert_close(zero_action[..., 4:7], torch.zeros(1, 4, 3))


def test_deterministic_sampler_resumes_from_consumed_batch():
    sampler = DeterministicInfiniteBatchSampler(10, 2, seed=7)
    iterator = iter(sampler)
    batches = [next(iterator) for _ in range(7)]
    resumed = iter(DeterministicInfiniteBatchSampler(10, 2, seed=7, start_batch=6))
    assert next(resumed) == batches[6]


def test_bc_policy_artifact_round_trip_and_online_inference(tmp_path):
    backbone_config = ResNetConfig(
        depths=[1, 1, 1, 1],
        hidden_sizes=[64, 128, 256, 512],
    ).to_dict()
    architecture = {
        "backbone_config": backbone_config,
        "hand_prior_source": "mlp",
        "vae_model_config": None,
        "hidden_dims": [512, 512, 256],
        "state_hidden_dims": [128, 128],
        "hand_state_window_size": 8,
        "backbone_pooling": "avg",
        "dense_init": "torch_uniform",
    }
    spec = LampPolicySpec(
        task="pick_bucket",
        policy_family="bc",
        embodiment="single",
        hand_prior_type="mlp",
        action_horizon=1,
        execution_horizon=1,
        core_action_dim=23,
        physical_action_dim=23,
        image_size=32,
        image_keys=("front", "wrist"),
        latent_dims={"single": 0},
    )
    statistics = {
        "arm_state_mean": np.zeros(7, np.float32),
        "arm_state_std": np.ones(7, np.float32),
        "hand_history_mean": np.zeros(16, np.float32),
        "hand_history_std": np.ones(16, np.float32),
        "arm_action_mean": np.zeros(7, np.float32),
        "arm_action_std": np.ones(7, np.float32),
        "hand_action_mean": np.zeros(16, np.float32),
        "hand_action_std": np.ones(16, np.float32),
    }
    policy = LampPolicy(BCPolicy(**architecture), spec, statistics).eval()
    metadata = {
        "kind": "policy",
        "model_type": "lamp_bc",
        "architecture": architecture,
        "spec": asdict(spec),
        "statistics_keys": sorted(statistics),
    }
    save_artifact(tmp_path, model=policy, metadata=metadata, statistics=statistics)
    restored = get_lamp_model(
        OmegaConf.create({"model_type": "lamp_bc", "model_path": str(tmp_path)})
    ).eval()
    observation = {
        "main_images": torch.zeros(2, 32, 32, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 32, 32, 3, dtype=torch.uint8),
        "panda_qpos": torch.zeros(2, 7),
        "hand_history": torch.zeros(2, 8, 16),
        "reset_mask": torch.tensor([True, False]),
    }
    actions, extra = restored.predict_action_batch(env_obs=observation)
    assert actions.shape == (2, 1, 23)
    assert extra["core_action_norm"].shape == (2, 1, 23)
    torch.testing.assert_close(restored(env_obs=observation), actions)
    torch.testing.assert_close(
        torch.linalg.vector_norm(actions[..., 3:7], dim=-1), torch.ones(2, 1)
    )
