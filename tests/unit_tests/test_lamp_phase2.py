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

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import av
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import ResNetConfig

from rlinf.config import validate_offline_cfg
from rlinf.data.datasets.lamp.dexjoco_lerobot import decode_video_rows
from rlinf.data.datasets.lamp.offline_dataset import (
    LampMMapDataset,
    lamp_steps_per_epoch,
)
from rlinf.models.embodiment.lamp import get_model as get_lamp_model
from rlinf.models.embodiment.lamp.artifact_io import (
    load_training_state,
    save_artifact,
    save_training_state,
)
from rlinf.models.embodiment.lamp.bc_policy import BCPolicy
from rlinf.models.embodiment.lamp.hand_prior_artifact import (
    load_prior_artifact,
    sorted_vq_codebook,
)
from rlinf.models.embodiment.lamp.hand_vae import DexJoCoHandVAE
from rlinf.models.embodiment.lamp.hand_vq_vae import HandVQVAE
from rlinf.models.embodiment.lamp.policy_wrapper import (
    LampPolicy,
    LampPolicySpec,
    LampTemporalEnsembleController,
)
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
    LAMPDiffusionPolicy,
)
from rlinf.models.embodiment.lamp.vq_action_normalization import (
    ALLEGRO_HAND_ACTION_HIGH,
    ALLEGRO_HAND_ACTION_LOW,
    denormalize_vq_hand_action,
    normalize_vq_hand_action,
)
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.utils.runner_utils import resolve_save_interval, resolve_training_horizon
from rlinf.workers.actor.lamp_il_worker import (
    DeterministicInfiniteBatchSampler,
    _clip_gradients,
    _cvae_kl_weights,
    _nearest_vq_indices,
    _prior_architecture,
    _training_contract,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
_SINGLE_ARM_TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
_SELECTED_Z2_TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pinch_tongs",
)
_VQ_STEPS_PER_EPOCH = {
    "click_mouse": 114,
    "fold_glasses": 190,
    "hammer_nail": 75,
    "pick_bucket": 153,
    "pinch_tongs": 139,
    "water_plant": 97,
}
_CVAE_TASK_STEPS = {
    "click_mouse": 20_000,
    "fold_glasses": 20_000,
    "hammer_nail": 30_000,
    "pick_bucket": 30_000,
    "pinch_tongs": 30_000,
    "water_plant": 30_000,
}
_DP_CONFIG_NAMES = tuple(
    path.stem
    for path in sorted(_CONFIG_DIR.glob("dexjoco_lamp_dp*.yaml"))
    if "eval" not in path.stem
)


def _compose_lamp_config(config_name: str):
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(config_name=config_name)


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


def test_offline_runner_export_only_skips_training():
    runner = object.__new__(OfflineRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "export_only": True,
                "resume_dir": "/tmp/run/checkpoints/global_step_30000",
            }
        }
    )
    runner.logger = MagicMock()
    runner.actor = MagicMock()
    runner._finish_logging = MagicMock()

    runner.run()

    runner.actor.export_deployment_artifacts.assert_called_once_with(
        "/tmp/run/checkpoints/global_step_30000/actor"
    )
    runner.actor.export_deployment_artifacts.return_value.wait.assert_called_once_with()
    runner._finish_logging.assert_called_once_with()


def test_training_horizon_preserves_zero_step_cap_and_rejects_other_negatives():
    assert resolve_training_horizon(
        {"max_epochs": 10, "steps_per_epoch": 4, "max_steps": 0}
    ) == (4, 0)
    with pytest.raises(ValueError, match="max_steps"):
        resolve_training_horizon(
            {"max_epochs": 10, "steps_per_epoch": 4, "max_steps": -2}
        )


def test_lamp_epoch_schedule_is_derived_from_cache_at_runtime(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"train_rows": 35_812}), encoding="utf-8"
    )
    steps_per_epoch = lamp_steps_per_epoch(tmp_path, global_batch_size=256)
    runner_cfg = {
        "max_epochs": 1_500,
        "max_steps": -1,
        "save_every_epochs": 100,
        "save_interval": 10_000,
    }

    assert steps_per_epoch == 139
    assert resolve_training_horizon(runner_cfg, steps_per_epoch=steps_per_epoch) == (
        139,
        208_500,
    )
    assert resolve_save_interval(runner_cfg, steps_per_epoch=steps_per_epoch) == 13_900


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
    assert (
        int.from_bytes((tmp_path / "model.safetensors").read_bytes()[:8], "little") > 0
    )
    assert (tmp_path / "model.safetensors").stat().st_mode & 0o044 == 0o044
    restored, restored_metadata, restored_statistics = load_prior_artifact(
        tmp_path,
        expected_type="vae",
        expected_task="pick_bucket",
        expected_dataset_fingerprint="fixture",
        expected_hand_side="single",
    )
    assert restored.latent_dim == 5
    assert restored_metadata["latent_dim"] == 5
    assert len(restored_metadata["model_sha256"]) == 64
    assert len(restored_metadata["statistics_sha256"]) == 64
    assert set(restored_statistics) == set(statistics)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)

    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(model_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="model_sha256"):
        load_prior_artifact(tmp_path)


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


def test_vq_action_normalization_uses_fixed_allegro_ctrlrange():
    low = np.asarray(ALLEGRO_HAND_ACTION_LOW, dtype=np.float32)
    high = np.asarray(ALLEGRO_HAND_ACTION_HIGH, dtype=np.float32)
    midpoint = (low + high) * 0.5
    physical = np.stack((low - 1.0, midpoint, high + 1.0))

    normalized = normalize_vq_hand_action(physical)

    np.testing.assert_allclose(normalized[0], -1.0)
    np.testing.assert_allclose(normalized[1], 0.0, atol=2e-7)
    np.testing.assert_allclose(normalized[2], 1.0)
    np.testing.assert_allclose(
        denormalize_vq_hand_action(normalized),
        np.stack((low, midpoint, high)),
        atol=2e-7,
    )


def test_vq_codebook_is_exported_and_matched_in_physical_action_space():
    model = HandVQVAE(latent_dim=16, hidden_dim=32, layer_num=1)
    normalized_action = torch.linspace(-1.0, 1.0, 16)
    with torch.no_grad():
        for parameter in model.decoder.parameters():
            parameter.zero_()
        model.decoder.output.bias.copy_(normalized_action)

    codebook = sorted_vq_codebook(model)
    expected = denormalize_vq_hand_action(normalized_action.numpy())

    assert codebook.shape == (16, 16)
    np.testing.assert_allclose(codebook, np.broadcast_to(expected, codebook.shape))
    physical_codebook = np.zeros((16, 16), dtype=np.float32)
    physical_codebook[:, 0] = np.linspace(-0.4, 0.4, 16)
    assert _nearest_vq_indices(physical_codebook[[7]], physical_codebook).item() == 7


def test_vq_dp_decoder_consumes_a_physical_codebook_without_zscore():
    backbone_config = ResNetConfig(
        depths=[1, 1, 1, 1], hidden_sizes=[64, 128, 256, 512]
    ).to_dict()
    codebook = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) / 100.0
    model = LAMPDiffusionPolicy(
        backbone_config,
        hand_prior_source="vq_codebook",
        vq_codebook=codebook,
        core_action_mean=np.zeros(8, dtype=np.float32),
        core_action_std=np.ones(8, dtype=np.float32),
        hand_action_mean=np.full(16, 100.0, dtype=np.float32),
        hand_action_std=np.full(16, 10.0, dtype=np.float32),
    )
    core = torch.zeros(1, 1, 8)
    core[..., 7] = -1.0

    action, auxiliary = model._decode_core(core)

    torch.testing.assert_close(action[..., 7:], torch.from_numpy(codebook[None, 0:1]))
    assert auxiliary["vq_index"].item() == 0


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


def test_temporal_ensemble_preview_matches_commit_without_changing_state():
    controller = LampTemporalEnsembleController()
    committed_reference = LampTemporalEnsembleController()
    assert controller.execution_horizon == 4
    assert controller.decay == 0.25

    first = torch.zeros(1, 16, 23)
    first[..., 3] = 1.0
    first[..., 7:] = 1.0
    second = first.clone()
    second[..., 7:] = 3.0

    apply_action = controller.apply(first)
    commit_action = committed_reference.commit(first)
    torch.testing.assert_close(apply_action, commit_action, rtol=0.0, atol=0.0)

    reset_preview = controller.preview(second, reset_mask=torch.tensor([True]))
    torch.testing.assert_close(
        reset_preview[..., 7:],
        torch.full((1, 4, 16), 3.0),
        rtol=0.0,
        atol=0.0,
    )
    preview = controller.preview(second)
    repeated_preview = controller.preview(second)
    expected = committed_reference.commit(second)
    torch.testing.assert_close(preview, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeated_preview, expected, rtol=0.0, atol=0.0)
    assert torch.all((preview[..., 7:] > 1.0) & (preview[..., 7:] < 3.0))

    committed = controller.commit(second)
    torch.testing.assert_close(committed, expected, rtol=0.0, atol=0.0)


def test_temporal_ensemble_preview_can_persist_only_confirmed_reset_rows():
    controller = LampTemporalEnsembleController()
    committed_reference = LampTemporalEnsembleController()

    first = torch.zeros(2, 16, 23)
    first[..., 3] = 1.0
    first[0, ..., 7:] = 1.0
    first[1, ..., 7:] = 10.0
    controller.commit(first)
    committed_reference.commit(first)

    candidate = first.clone()
    candidate[0, ..., 7:] = 3.0
    candidate[1, ..., 7:] = 30.0
    reset_mask = torch.tensor([True, False])
    preview = controller.preview(
        candidate,
        reset_mask=reset_mask,
        confirmed_reset=True,
    )
    torch.testing.assert_close(preview[0, ..., 7:], torch.full((4, 16), 3.0))
    assert torch.all((preview[1, ..., 7:] > 10.0) & (preview[1, ..., 7:] < 30.0))

    following = first.clone()
    following[0, ..., 7:] = 5.0
    following[1, ..., 7:] = 50.0
    committed = controller.commit(following)
    expected = committed_reference.commit(following, reset_mask=reset_mask)
    torch.testing.assert_close(committed, expected, rtol=0.0, atol=0.0)


def test_dp_policy_can_execute_newest_plan_without_temporal_ensemble(monkeypatch):
    spec = LampPolicySpec(
        task="pick_bucket",
        policy_family="dp",
        embodiment="single",
        hand_prior_type="mlp",
        action_horizon=16,
        execution_horizon=4,
        core_action_dim=23,
        physical_action_dim=23,
        image_size=32,
        image_keys=("front", "wrist"),
        latent_dims={"single": 0},
    )
    policy = LampPolicy(
        torch.nn.Identity(),
        spec,
        {},
        use_temporal_ensemble=False,
    )
    core = torch.arange(16 * 23, dtype=torch.float32).reshape(1, 16, 23)
    physical = core + 1000.0
    monkeypatch.setattr(policy, "_processed_inputs", lambda _: ())
    monkeypatch.setattr(
        policy,
        "_predict_plan_from_processed",
        lambda: (core, physical),
    )

    actions, extra = policy.predict_action_batch(env_obs={})

    torch.testing.assert_close(actions, physical[:, :4])
    torch.testing.assert_close(extra["core_action_norm"], core)
    assert actions.shape == (1, 4, 23)


def test_dp_policy_can_override_open_loop_execution_horizon(monkeypatch):
    spec = LampPolicySpec(
        task="water_plant",
        policy_family="dp",
        embodiment="single",
        hand_prior_type="mlp",
        action_horizon=16,
        execution_horizon=4,
        core_action_dim=23,
        physical_action_dim=23,
        image_size=32,
        image_keys=("front", "wrist"),
        latent_dims={"single": 0},
    )
    policy = LampPolicy(
        torch.nn.Identity(),
        spec,
        {},
        use_temporal_ensemble=False,
        execution_horizon_override=16,
    )
    core = torch.arange(16 * 23, dtype=torch.float32).reshape(1, 16, 23)
    physical = core + 1000.0
    monkeypatch.setattr(policy, "_processed_inputs", lambda _: ())
    monkeypatch.setattr(
        policy,
        "_predict_plan_from_processed",
        lambda: (core, physical),
    )

    actions, _ = policy.predict_action_batch(env_obs={})

    torch.testing.assert_close(actions, physical)
    assert policy.controller.execution_horizon == 16


def test_dp_policy_rejects_invalid_execution_horizon_override():
    spec = LampPolicySpec(
        task="water_plant",
        policy_family="dp",
        embodiment="single",
        hand_prior_type="mlp",
        action_horizon=16,
        execution_horizon=4,
        core_action_dim=23,
        physical_action_dim=23,
        image_size=32,
        image_keys=("front", "wrist"),
        latent_dims={"single": 0},
    )

    with pytest.raises(ValueError, match="execution_horizon_override"):
        LampPolicy(
            torch.nn.Identity(),
            spec,
            {},
            use_temporal_ensemble=False,
            execution_horizon_override=17,
        )


def test_dp_policy_applies_inference_step_override():
    class _InferenceCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_inference_steps = 16

        def set_num_inference_steps(self, value):
            self.num_inference_steps = int(value)

    spec = LampPolicySpec(
        task="water_plant",
        policy_family="dp",
        embodiment="single",
        hand_prior_type="mlp",
        action_horizon=16,
        execution_horizon=4,
        core_action_dim=23,
        physical_action_dim=23,
        image_size=32,
        image_keys=("front", "wrist"),
        latent_dims={"single": 0},
    )
    core = _InferenceCore()

    LampPolicy(core, spec, {}, num_inference_steps_override=32)

    assert core.num_inference_steps == 32


def test_deterministic_sampler_resumes_from_consumed_batch():
    sampler = DeterministicInfiniteBatchSampler(10, 2, seed=7)
    iterator = iter(sampler)
    batches = [next(iterator) for _ in range(7)]
    resumed = iter(DeterministicInfiniteBatchSampler(10, 2, seed=7, start_batch=6))
    assert next(resumed) == batches[6]


def test_cvae_default_config_matches_jax_launcher_contract():
    cfg = _compose_lamp_config("dexjoco_lamp_prior_cvae")

    assert cfg.runner.max_steps == 20_000
    assert cfg.runner.save_interval == 5_000
    assert cfg.data.dataset_root.endswith(
        "DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"
    )
    assert cfg.actor.seed == 42
    assert cfg.actor.global_batch_size == cfg.actor.micro_batch_size == 256
    assert cfg.actor.eval_batch_size == 512
    assert cfg.actor.validation_interval == 20_000
    assert cfg.actor.validation_batches == -1
    assert cfg.actor.torch_compile is False
    assert OmegaConf.to_container(cfg.actor.optim, resolve=True) == {
        "lr": 3e-4,
        "min_lr": 1e-5,
        "warmup_steps": 500,
        "weight_decay": 1e-4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "clip_grad": 1.0,
        "backbone_lr_ratio": 0.1,
    }
    prior = cfg.actor.model.hand_prior
    assert prior.type == "cvae"
    assert prior.latent_dim == 3
    assert prior.hidden_dim == 1024
    assert prior.posterior_kl_weight == 1e-4
    assert prior.prior_kl_weight == 1e-3
    assert prior.kl_warmup_steps == 2_000


def test_cvae_kl_warmup_converts_torch_step_to_jax_one_based_step():
    first_q, first_p = _cvae_kl_weights(torch.tensor(0))
    full_q, full_p = _cvae_kl_weights(torch.tensor(1_999))

    torch.testing.assert_close(first_q, torch.tensor(1e-4 / 2_000))
    torch.testing.assert_close(first_p, torch.tensor(1e-3 / 2_000))
    torch.testing.assert_close(full_q, torch.tensor(1e-4))
    torch.testing.assert_close(full_p, torch.tensor(1e-3))


@pytest.mark.parametrize(("task", "max_steps"), _CVAE_TASK_STEPS.items())
def test_cvae_dim6_task_configs_match_selected_jax_recipe(task, max_steps):
    cfg = _compose_lamp_config(f"dexjoco_lamp_prior_cvae_dim6_{task}")

    assert cfg.data.task_name == task
    assert cfg.runner.max_steps == max_steps
    assert cfg.actor.validation_interval == max_steps
    assert cfg.actor.model.hand_prior.latent_dim == 6


@pytest.mark.parametrize("task", _SELECTED_Z2_TASKS)
def test_selected_dim2_prior_configs_fix_sweep_hyperparameters(task):
    cvae = _compose_lamp_config(f"dexjoco_lamp_prior_cvae_dim2_selected_{task}")
    prior = cvae.actor.model.hand_prior
    assert cvae.data.task_name == task
    assert prior.latent_dim == 2
    assert prior.posterior_kl_weight == 1e-4
    assert prior.prior_kl_weight == 3e-4

    pca = _compose_lamp_config(f"dexjoco_lamp_prior_pca_dim2_{task}")
    assert pca.data.task_name == task
    assert pca.actor.model.hand_prior.type == "pca"
    assert pca.actor.model.hand_prior.latent_dim == 2


@pytest.mark.parametrize("task", _SELECTED_Z2_TASKS)
@pytest.mark.parametrize("mode", ("decoder_only", "cvae", "pca", "vq", "mlp"))
def test_selected_dim2_dp_configs_fix_sweep_hyperparameters(task, mode):
    cfg = _compose_lamp_config(f"dexjoco_lamp_dp_il_{mode}_{task}")
    prior = cfg.actor.model.hand_prior

    assert cfg.data.task_name == task
    assert cfg.actor.optim.lr == 3e-5
    assert cfg.actor.optim.backbone_lr_ratio == 0.1
    if mode in ("decoder_only", "cvae"):
        assert prior.latent_dim == 2
        assert f"cvae_dim2_selected_{task}/artifact" in prior.artifact_path
    elif mode == "pca":
        assert prior.latent_dim == 2
        assert f"pca_dim2_{task}/artifact" in prior.artifact_path
    elif mode == "vq":
        assert prior.latent_dim == 1
    else:
        assert prior.artifact_path is None


@pytest.mark.parametrize(("task", "steps_per_epoch"), _VQ_STEPS_PER_EPOCH.items())
def test_vq_task_configs_defer_epoch_steps_to_runtime(task, steps_per_epoch):
    cfg = _compose_lamp_config(f"dexjoco_lamp_prior_vq_{task}")
    total_steps = 1_500 * steps_per_epoch
    validate_offline_cfg(cfg)

    assert cfg.data.task_name == task
    assert cfg.data.dataset_root.endswith(
        "DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"
    )
    assert cfg.runner.max_epochs == 1_500
    assert cfg.runner.max_steps == -1
    assert "steps_per_epoch" not in cfg.runner
    assert cfg.runner.save_every_epochs == 100
    assert resolve_training_horizon(cfg.runner, steps_per_epoch=steps_per_epoch) == (
        steps_per_epoch,
        total_steps,
    )
    assert (
        resolve_save_interval(cfg.runner, steps_per_epoch=steps_per_epoch)
        == 100 * steps_per_epoch
    )
    assert cfg.actor.validation_interval == -1
    assert cfg.actor.validate_at_end is True
    assert cfg.actor.validation_batches == -1
    assert cfg.actor.seed == 233
    assert cfg.actor.global_batch_size == cfg.actor.micro_batch_size == 256
    assert cfg.actor.eval_batch_size == 512
    assert cfg.actor.optim.lr == 3e-4
    assert cfg.actor.optim.min_lr == 0.0
    assert cfg.actor.optim.warmup_steps == 150
    assert cfg.actor.optim.weight_decay == 1e-6
    assert cfg.actor.optim.adam_beta1 == 0.95
    assert cfg.actor.optim.adam_beta2 == 0.999
    assert cfg.actor.optim.adam_eps == 1e-8
    assert cfg.actor.optim.clip_grad is None
    contract = _training_contract(
        cfg, steps_per_epoch=steps_per_epoch, max_steps=total_steps
    )
    assert contract["seed"] == 233
    assert contract["optimizer"]["clip_grad"] is None
    assert contract["model"]["hand_prior"]["ema_decay"] == 0.8
    assert _prior_architecture("vq", cfg.actor.model.hand_prior) == {
        "action_dim": 16,
        "latent_dim": 256,
        "hidden_dim": 512,
        "num_quantizers": 2,
        "codebook_size": 4,
        "layer_num": 5,
        "commitment_weight": 1.0,
        "ema_decay": 0.8,
        "epsilon": 1e-5,
        "dead_code_threshold": 0.0,
        "reconstruction_multiplier": 3.0,
        "vq_multiplier": 5.0,
    }


@pytest.mark.parametrize("config_name", _DP_CONFIG_NAMES)
def test_single_arm_dp_configs_use_the_jax_physical_batch(config_name):
    cfg = _compose_lamp_config(config_name)

    assert cfg.algorithm.stage == "dp"
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.micro_batch_size == 512
    assert cfg.actor.eval_batch_size == 512
    assert cfg.actor.global_batch_size // cfg.actor.micro_batch_size == 1


def test_null_gradient_clip_reports_norm_without_changing_gradients():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([6.0, 8.0])
    before = parameter.grad.clone()

    norm = _clip_gradients([parameter], None)

    torch.testing.assert_close(norm, torch.tensor(10.0))
    torch.testing.assert_close(parameter.grad, before)


def test_training_state_resume_tolerates_float_roundoff(tmp_path):
    model = torch.nn.Linear(2, 2)
    stored = {
        "architecture_sha256": "old-derived-hash",
        "artifact": {"architecture": {"mean": [0.032320454716682434]}},
        "max_steps": 30_000,
    }
    save_training_state(
        tmp_path,
        model=model,
        optimizer=None,
        scheduler=None,
        global_step=10_000,
        sampler_state=None,
        metadata=stored,
    )
    expected = {
        "architecture_sha256": "new-derived-hash",
        "artifact": {"architecture": {"mean": [0.03232046216726303]}},
        "max_steps": 30_000,
    }

    step, sampler = load_training_state(
        tmp_path,
        model=model,
        optimizer=None,
        scheduler=None,
        expected_metadata=expected,
    )

    assert step == 10_000
    assert sampler is None


def test_training_state_resume_rejects_material_metadata_change(tmp_path):
    model = torch.nn.Linear(2, 2)
    metadata = {
        "architecture_sha256": "old-derived-hash",
        "artifact": {"training_contract": {"optimizer": {"lr": 1e-5}}},
    }
    save_training_state(
        tmp_path,
        model=model,
        optimizer=None,
        scheduler=None,
        global_step=10_000,
        sampler_state=None,
        metadata=metadata,
    )
    expected = {
        "architecture_sha256": "new-derived-hash",
        "artifact": {"training_contract": {"optimizer": {"lr": 3e-5}}},
    }

    with pytest.raises(ValueError, match=r"metadata\.artifact.*optimizer\.lr"):
        load_training_state(
            tmp_path,
            model=model,
            optimizer=None,
            scheduler=None,
            expected_metadata=expected,
        )


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
