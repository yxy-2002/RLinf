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

"""Portable LAMP contracts and a hardware-free prior/DP/residual pipeline."""

import copy
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import ResNetConfig

from rlinf.data.datasets.lamp.offline_dataset import (
    LampFrameData,
    LampMMapDataset,
    LampSourceMetadata,
    load_cache_metadata,
    load_cache_statistics,
    prepare_lamp_cache,
)
from rlinf.data.datasets.lamp.residual_replay import validate_lamp_residual_trajectory
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.envs.lamp_adapter import (
    LampEnvAdapter,
    LampObservationHistory,
    apply_lamp_execution_feedback,
    validate_lamp_environment,
)
from rlinf.models.embodiment.lamp import get_model
from rlinf.models.embodiment.lamp.artifact_io import (
    load_artifact,
    load_training_state,
    save_training_state,
)
from rlinf.models.embodiment.lamp.policy_wrapper import LampPolicy, LampPolicySpec
from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy
from rlinf.models.embodiment.lamp.resnet18 import HFResNet18Backbone
from rlinf.models.embodiment.lamp.robot_spec import (
    LampRobotSpec,
    dexjoco_robot_spec,
    validate_horizons,
)
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import LAMPDiffusionPolicy
from rlinf.models.embodiment.lamp.vq_action_normalization import (
    denormalize_vq_hand_action,
    normalize_vq_hand_action,
)
from rlinf.workers.actor.fsdp_lamp_residual_sac_policy_worker import (
    LampResidualSACFSDPPolicy,
)
from rlinf.workers.actor.lamp_il_worker import LampILWorker


def synthetic_spec(hand_dim=20):
    """Synthetic command units/bounds; not a Wuji hardware configuration."""
    return LampRobotSpec(
        name=f"synthetic_{hand_dim}",
        arm_action_dim=6,
        hand_action_dim=hand_dim,
        arm_state_dim=9,
        hand_state_dim=hand_dim - 2,
        action_representation="incremental",
        action_frame="synthetic_tool",
        arm_action_units=("scaled_delta",) * 6,
        hand_action_units=("synthetic_rad",) * hand_dim,
        hand_action_names=tuple(f"command_{i}" for i in range(hand_dim)),
        arm_state_semantics="synthetic_pose_velocity",
        hand_state_semantics="synthetic_measured",
        arm_state_names=tuple(f"arm_{i}" for i in range(9)),
        hand_state_names=tuple(f"sensor_{i}" for i in range(hand_dim - 2)),
        hand_action_low=tuple(-1.0 - i / 20 for i in range(hand_dim)),
        hand_action_high=tuple(1.0 + i / 10 for i in range(hand_dim)),
    )


class ArraySource:
    def __init__(self, root, spec):
        self.metadata = LampSourceMetadata(
            "synthetic", Path(root), "a" * 64, "b" * 64, ("a", "b"), spec
        )
        rng = np.random.default_rng(17)
        self.frames = LampFrameData(
            np.repeat(np.arange(10), 4),
            rng.normal(size=(40, spec.arm_state_dim)).astype(np.float32),
            rng.normal(size=(40, spec.hand_state_dim)).astype(np.float32),
            rng.normal(size=(40, spec.action_dim)).astype(np.float32),
            spec,
        )

    def load_frames(self):
        return self.frames

    def images_for(self, rows, image_size, *, label):
        image = np.broadcast_to(
            np.asarray(rows, np.uint8)[:, None, None, None],
            (len(rows), image_size, image_size, 3),
        ).copy()
        return {"front": image, "wrist": image.copy()}


def architecture(spec, prior, h):
    core_dim = (
        spec.action_dim
        if prior == "mlp"
        else spec.arm_action_dim + (1 if prior == "vq_codebook" else 2)
    )
    result = {
        "backbone_config": ResNetConfig(
            depths=[1, 1, 1, 1], hidden_sizes=[64, 128, 256, 512]
        ).to_dict(),
        "robot_spec": spec.to_dict(),
        "action_horizon": h,
        "hand_prior_source": prior,
        "core_action_mean": [0.0] * core_dim,
        "core_action_std": [1.0] * core_dim,
        "hand_action_mean": [0.0] * spec.hand_action_dim,
        "hand_action_std": [1.0] * spec.hand_action_dim,
        "decoder_history_length": 3,
    }
    if prior == "lamplstm":
        result["lamplstm_model_config"] = {
            "action_dim": spec.hand_action_dim,
            "history_dim": spec.hand_state_dim,
            "condition_mode_encoder": "film",
            "condition_mode_decoder": "film",
            "horizon": h,
            "latent_dim": 2,
            "action_hidden_dim": 8,
            "condition_hidden_dim": 8,
        }
    elif prior == "pca":
        result.update(
            pca_mean=[0.0] * spec.hand_action_dim,
            pca_components=np.eye(spec.hand_action_dim)[:2].tolist(),
        )
    elif prior == "vq_codebook":
        result["vq_codebook"] = (
            np.linspace(-0.5, 0.5, 16 * spec.hand_action_dim)
            .reshape(16, spec.hand_action_dim)
            .tolist()
        )
    return result


def make_policy(spec, prior, h, k):
    core = LAMPDiffusionPolicy(**architecture(spec, prior, h)).eval()
    policy_spec = LampPolicySpec(
        "synthetic",
        "dp",
        "single",
        prior,
        h,
        k,
        core.core_dim,
        spec.action_dim,
        32,
        ("a", "b"),
        robot_spec=spec,
    )
    stats = {}
    for name, width in (
        ("arm_state_pair", spec.arm_state_dim),
        ("hand_state_pair", spec.hand_state_dim),
        ("arm_action", spec.arm_action_dim),
        ("hand_action", spec.hand_action_dim),
    ):
        stats[name + "_mean"] = [0.0] * width
        stats[name + "_std"] = [1.0] * width
    return LampPolicy(core, policy_spec, stats, use_temporal_ensemble=False)


def observation(spec, batch=2):
    return {
        "main_images": torch.zeros(batch, 32, 32, 3),
        "wrist_images": torch.zeros(batch, 32, 32, 3),
        "arm_state_pair": torch.randn(batch, 2, spec.arm_state_dim),
        "hand_state_pair": torch.randn(batch, 2, spec.hand_state_dim),
        "hand_history": torch.randn(batch, 3, spec.hand_state_dim),
        "hand_history_mask": torch.ones(batch, 3),
    }


class SyntheticAdapter(LampEnvAdapter):
    """Two synthetic environments with different episode lengths; no SDKs."""

    env_type = "realworld"

    def __init__(self, env, cfg, model_cfg):
        super().__init__(env, cfg, model_cfg)
        self.history = LampObservationHistory(self.robot_spec, 2, 3)
        self.steps = torch.zeros(2, dtype=torch.long)

    def _obs(self):
        return {
            **self.history.observation(),
            "main_images": torch.zeros(2, 32, 32, 3),
            "wrist_images": torch.zeros(2, 32, 32, 3),
        }

    def reset(self, env_idx=None, **kwargs):
        idx = torch.arange(2) if env_idx is None else torch.as_tensor(env_idx)
        self.steps[idx] = 0
        self.history.update(
            torch.zeros(len(idx), 9), torch.zeros(len(idx), 18), idx, reset=True
        )
        return self._obs(), {}

    def step(self, actions, active=None, **kwargs):
        idx = torch.arange(2) if active is None else torch.where(active)[0]
        executed = actions.clamp(-0.2, 0.2)
        self.steps[idx] += 1
        # Measured states intentionally differ from issued commands.
        measured = self.steps[idx, None].float()
        self.history.update(measured.expand(-1, 9), measured.expand(-1, 18), idx)
        terminated = self.steps >= torch.tensor([2, 9])
        return (
            self._obs(),
            torch.ones(2),
            terminated,
            torch.zeros(2, dtype=torch.bool),
            {"executed_action": executed},
        )

    def chunk_step(self, actions):
        active = torch.ones(2, dtype=torch.bool)
        observations, rewards, terms, truncs, infos, valids, executed = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for action in actions.unbind(1):
            valids.append(active.clone())
            obs, reward, term, trunc, info = self.step(action, active=active)
            observations.append(obs)
            rewards.append(reward * active)
            terms.append(term & active)
            truncs.append(trunc)
            executed.append(info["executed_action"])
            infos.append(info)
            active &= ~term
        final_obs = self._obs()
        if (~active).any():
            observations[-1], _ = self.reset(torch.where(~active)[0])
        infos[-1].update(
            final_observation=final_obs,
            primitive_valid=torch.stack(valids, 1),
            executed_action=torch.stack(executed, 1),
        )
        return (
            observations,
            torch.stack(rewards, 1),
            torch.stack(terms, 1),
            torch.stack(truncs, 1),
            infos,
        )


@pytest.mark.parametrize("h,k", [(4, 1), (8, 8), (12, 3), (16, 8), (32, 32)])
@pytest.mark.parametrize("prior", ["lamplstm", "pca", "vq_codebook", "mlp"])
def test_variable_horizons_and_causal_gradients(h, k, prior):
    spec = synthetic_spec()
    base = make_policy(spec, prior, h, k)
    policy = LampResidualSACPolicy(base, contract_version=5)
    obs = observation(spec)
    features = torch.randn(2, 256)
    core = torch.zeros(2, h, base.spec.core_action_dim)
    action, log_prob, context = policy.sac_forward(
        obs, base_core=core, condition=features
    )
    assert action.shape == (2, k * spec.action_dim)
    assert policy.target_entropy == -k * base.spec.core_action_dim
    q = policy.sac_q_forward(obs, action, shared_feature=context, detach_encoder=True)
    (-q.mean() + 0.1 * log_prob.mean()).backward()
    assert all(p.grad is None for p in base.parameters())
    grad = policy.residual_actor.mean_head.weight.grad.reshape(
        h, base.spec.core_action_dim, -1
    )
    assert grad[:k].abs().sum() > 0
    assert not grad[k:].count_nonzero()
    metrics = LampResidualSACFSDPPolicy._residual_actor_metrics(
        context, log_pi=log_prob, residual_scale=policy.residual_scale_per_core
    )
    assert all(np.isfinite(v) for v in metrics.values())
    # Actual U-Net forward exercises down/up sampling for every H.
    prediction = base.core.denoiser(core, torch.zeros(2), global_cond=features)
    assert prediction.shape == core.shape
    # Incremental coordinates must not be normalized as a quaternion.
    torch.testing.assert_close(
        policy._correct_core_quaternion(core + 0.2, core), core + 0.2, rtol=0, atol=0
    )


@pytest.mark.parametrize("h,k", [(0, 1), (3, 1), (6, 1), (8, 0), (8, 9)])
def test_invalid_timing_rejected(h, k):
    with pytest.raises(ValueError):
        validate_horizons(h, k)


def test_cache_semantics_and_old_cache_alias(tmp_path):
    spec = synthetic_spec()
    source = ArraySource(tmp_path, spec)
    first = prepare_lamp_cache(
        source=source, cache_root=tmp_path / "cache", action_horizon=8, history_length=3
    )
    second = prepare_lamp_cache(
        source=source,
        cache_root=tmp_path / "cache",
        action_horizon=12,
        history_length=3,
    )
    source.metadata = replace(
        source.metadata, robot_spec=replace(spec, action_frame="other")
    )
    source.frames = replace(source.frames, robot_spec=source.metadata.robot_spec)
    third = prepare_lamp_cache(
        source=source, cache_root=tmp_path / "cache", action_horizon=8, history_length=3
    )
    assert len({first, second, third}) == 3
    ds = LampMMapDataset(
        first, "train", ["target_action", "lamplstm_decoder_history_norm"]
    )
    assert ds[0]["target_action"].shape == (8, 26)
    assert ds[0]["lamplstm_decoder_history_norm"].shape == (3, 18)
    old_source = ArraySource(tmp_path / "legacy", dexjoco_robot_spec())
    old = prepare_lamp_cache(source=old_source, cache_root=tmp_path / "cache")
    assert (old / "train/target_action23.npy").is_file()
    assert LampMMapDataset(old, "train", ["target_action"])[0][
        "target_action"
    ].shape == (16, 23)


def test_vq_bounds_are_spec_driven():
    spec = synthetic_spec()
    raw = torch.tensor([spec.hand_action_low, spec.hand_action_high])
    norm = normalize_vq_hand_action(raw, spec)
    torch.testing.assert_close(norm, torch.stack((-torch.ones(20), torch.ones(20))))
    torch.testing.assert_close(denormalize_vq_hand_action(norm, spec), raw)
    with pytest.raises(ValueError, match="bounds"):
        normalize_vq_hand_action(
            raw, replace(spec, hand_action_low=(), hand_action_high=())
        )


@pytest.mark.parametrize("prior", ["lamplstm", "pca", "vq", "mlp"])
def test_real_il_worker_to_residual_step_without_hardware(tmp_path, monkeypatch, prior):
    """Exercise production target construction, losses, export and reload."""
    spec = synthetic_spec()
    cache = prepare_lamp_cache(
        source=ArraySource(tmp_path, spec),
        cache_root=tmp_path / "cache",
        action_horizon=8,
        history_length=3,
        include_images=True,
        image_size=32,
    )
    worker = object.__new__(LampILWorker)
    worker.device = torch.device("cpu")
    worker._cache_metadata = load_cache_metadata(cache)
    worker._statistics = load_cache_statistics(cache)
    worker._global_step = 0
    worker._ema_model = None
    worker._policy_spec = None
    worker.cfg = OmegaConf.create(
        {
            "data": {"cache_path": str(cache)},
            "actor": {
                "model": {
                    "hand_prior": {
                        "type": prior,
                        "latent_dim": 2,
                        "history_length": 3,
                        "action_hidden_dim": 8,
                        "condition_hidden_dim": 8,
                        "hidden_dim": 16,
                        "code_latent_dim": 8,
                        "layer_num": 1,
                        "artifact_path": str(tmp_path / "prior"),
                    },
                    "action_horizon": 8,
                    "execution_horizon": 3,
                    "resnet_path": "unused",
                    "robot_spec": spec.to_dict(),
                }
            },
        }
    )
    if prior != "mlp":
        worker.stage = "prior"
        worker._setup_prior()
        if prior != "pca":
            batch = next(
                iter(
                    DataLoader(
                        LampMMapDataset(cache, "train", worker._dataset_keys()),
                        batch_size=2,
                    )
                )
            )
            optimizer = torch.optim.Adam(worker.model.parameters(), lr=1e-4)
            worker._prior_loss(batch, torch.tensor(0))["total_loss"].backward()
            optimizer.step()
        worker._save_deployment_artifact(tmp_path / "prior")
        assert load_artifact(tmp_path / "prior")[0]["robot_spec"] == spec.to_dict()
    backbone_cfg = architecture(spec, "mlp", 8)["backbone_config"]
    backbone = HFResNet18Backbone(backbone_cfg, pooling="avg")
    monkeypatch.setattr(
        "rlinf.workers.actor.lamp_il_worker.load_hf_resnet18_params",
        lambda _: (backbone_cfg, backbone.resnet.state_dict(), None),
    )
    worker.stage = "dp"
    worker._setup_dp()
    batch = worker._prepare_batch(
        next(
            iter(
                DataLoader(
                    LampMMapDataset(cache, "train", worker._dataset_keys()),
                    batch_size=2,
                )
            )
        )
    )
    optimizer = torch.optim.Adam(
        (p for p in worker.model.parameters() if p.requires_grad), lr=1e-4
    )
    worker._dp_loss(batch)["loss"].backward()
    optimizer.step()
    worker._save_deployment_artifact(tmp_path / "dp")
    base = get_model(
        OmegaConf.create(
            {
                "model_type": "lamp_dp",
                "model_path": str(tmp_path / "dp"),
                "robot_spec": spec.to_dict(),
                "use_temporal_ensemble": False,
            }
        )
    )
    base.core.set_num_inference_steps(1)
    residual = LampResidualSACPolicy(base, contract_version=5)
    env_cfg = OmegaConf.create(
        {
            "env_type": "realworld",
            "lamp_adapter": "test_lamp_portable:SyntheticAdapter",
            "lamp_robot_spec": spec.to_dict(),
        }
    )
    env = SyntheticAdapter(object(), env_cfg, worker.cfg.actor.model)
    validate_lamp_environment(env_cfg, worker.cfg.actor.model)
    obs, _ = env.reset()
    actions, extra = residual.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, 3, 26)
    retained = extra["forward_inputs"]["action"]
    next_obs, rewards, terminations, _, infos = env.chunk_step(actions)
    executed = infos[-1]["executed_action"]
    valid = torch.tensor([[True, True, False], [True, True, True]])
    torch.testing.assert_close(infos[-1]["primitive_valid"], valid)
    apply_lamp_execution_feedback(extra["forward_inputs"], infos[-1])
    torch.testing.assert_close(retained, executed.flatten(1), rtol=0, atol=0)
    trajectory = Trajectory(
        max_episode_length=10,
        model_weights_id="v0",
        actions=retained[None],
        rewards=rewards[None],
        forward_inputs={
            key: value[None] for key, value in extra["forward_inputs"].items()
        },
    )
    validate_lamp_residual_trajectory(
        trajectory,
        contract_version=5,
        robot_spec=spec,
        action_horizon=8,
        execution_horizon=3,
        contract_digest=residual.contract_digest,
    )
    # Exercise the production SAC critic, actor and temperature objectives.
    sac = object.__new__(LampResidualSACFSDPPolicy)
    sac.cfg = OmegaConf.create({"algorithm": {"gamma": 0.97, "backup_entropy": True}})
    sac.torch_dtype = torch.float32
    sac.model = residual
    sac.target_model = copy.deepcopy(residual)
    sac.optimizer = torch.optim.Adam(residual.residual_actor.parameters(), lr=1e-4)
    sac.qf_optimizer = torch.optim.Adam(residual.q_head.parameters(), lr=1e-4)
    sac.entropy_temp = SimpleNamespace(
        alpha=torch.tensor(0.1), base_alpha=torch.nn.Parameter(torch.tensor(-2.3))
    )
    sac.target_entropy = residual.target_entropy
    batch = {
        "curr_obs": obs,
        "next_obs": next_obs[-1],
        "actions": retained,
        "rewards": rewards,
        "terminations": terminations,
        "forward_inputs": extra["forward_inputs"],
    }
    q_before = next(residual.q_head.parameters()).detach().clone()
    q_loss, metrics = inspect.unwrap(LampResidualSACFSDPPolicy.forward_critic)(
        sac, batch
    )
    q_loss.backward()
    sac.qf_optimizer.step()
    assert not torch.equal(q_before, next(residual.q_head.parameters()))
    assert metrics["macro_reward"] == float((rewards * valid).sum(-1).mean())
    loss, _, _ = inspect.unwrap(LampResidualSACFSDPPolicy.forward_actor)(sac, batch)
    before = residual.residual_actor.mean_head.weight.detach().clone()
    loss.backward()
    sac.optimizer.step()
    alpha_loss = inspect.unwrap(LampResidualSACFSDPPolicy.forward_alpha)(sac, batch)
    alpha_loss.backward()
    assert torch.isfinite(sac.entropy_temp.base_alpha.grad)
    assert not torch.equal(before, residual.residual_actor.mean_head.weight)
    assert all(p.grad is None for p in base.parameters())
    # Matching widths cannot hide changed units/order in deployment or replay.
    other = replace(spec, action_frame="other")
    with pytest.raises(ValueError, match="robot spec"):
        get_model(
            OmegaConf.create(
                {
                    "model_type": "lamp_dp",
                    "model_path": str(tmp_path / "dp"),
                    "robot_spec": other.to_dict(),
                }
            )
        )
    bad_digest = residual.contract_digest.clone()
    bad_digest[0] ^= 1
    with pytest.raises(ValueError, match="contract"):
        validate_lamp_residual_trajectory(
            trajectory,
            contract_version=5,
            robot_spec=spec,
            action_horizon=8,
            execution_horizon=3,
            contract_digest=bad_digest,
        )
    saved = copy.deepcopy(residual.state_dict())
    saved["contract_digest"][0] ^= 1
    with pytest.raises(RuntimeError, match="contract"):
        residual.load_state_dict(saved)


def test_unadapted_environment_rejected():
    cfg = OmegaConf.create({"env_type": "realworld"})
    model = OmegaConf.create({"robot_spec": synthetic_spec().to_dict()})
    with pytest.raises(ValueError, match="adapter"):
        validate_lamp_environment(cfg, model)


def test_primitive_measured_history_partial_reset_and_final_observation():
    spec = synthetic_spec()
    cfg = OmegaConf.create({"env_type": "realworld", "lamp_robot_spec": spec.to_dict()})
    env = SyntheticAdapter(
        object(), cfg, OmegaConf.create({"robot_spec": spec.to_dict()})
    )
    obs, _ = env.reset()
    torch.testing.assert_close(
        obs["hand_history_mask"], torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    )
    observations, rewards, _, _, infos = env.chunk_step(torch.full((2, 3, 26), 0.1))
    final = infos[-1]["final_observation"]
    torch.testing.assert_close(
        final["hand_history"][:, :, 0], torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    )
    torch.testing.assert_close(
        observations[-1]["hand_history"][:, :, 0],
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
    )
    torch.testing.assert_close(
        final["arm_state_pair"][:, :, 0], torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    )
    torch.testing.assert_close(
        rewards, torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    )


def test_v5_configuration_and_async_preflight():
    from hydra import compose, initialize_config_dir

    from rlinf.config import (
        validate_lamp_async_cfg,
        validate_lamp_residual_contract_cfg,
    )

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="dexjoco_lamp_residual_sac_lamplstm_water_plant")
    OmegaConf.set_struct(cfg, False)
    cfg.actor.model.robot_spec = synthetic_spec().to_dict()
    cfg.actor.model.contract_version = 5
    cfg.actor.model.action_dim = 26
    cfg.actor.model.action_horizon = 12
    cfg.actor.model.num_action_chunks = 3
    assert validate_lamp_residual_contract_cfg(cfg, cfg.actor.model) == 5
    validate_lamp_async_cfg(cfg)


def test_v2_training_resume_preserves_next_update_and_rejects_contract_change(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def update():
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(4, 2)).square().mean().backward()
        optimizer.step()

    update()
    legacy = {
        "training_schema_version": 2,
        "artifact": {
            "architecture": {"action_horizon": 16, "backbone_config": {}},
            "spec": {"action_horizon": 16, "execution_horizon": 8},
            "training_contract": {"optimizer": {"lr": 1e-3}},
        },
        "dataset_fingerprint": "legacy-unchanged",
    }
    save_training_state(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        global_step=1,
        sampler_state={"position": 4},
        metadata=legacy,
    )
    raw = (tmp_path / "training_state.pt").read_bytes()
    update()
    expected_weights = copy.deepcopy(model.state_dict())
    current = copy.deepcopy(legacy)
    current["training_schema_version"] = 3
    current["robot_spec"] = dexjoco_robot_spec().to_dict()
    current["artifact"]["robot_spec"] = current["robot_spec"]
    for field in ("architecture", "spec"):
        current["artifact"][field]["robot_spec"] = current["robot_spec"]
    step, sampler = load_training_state(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        expected_metadata=current,
    )
    assert (step, sampler) == (1, {"position": 4})
    update()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
    assert (tmp_path / "training_state.pt").read_bytes() == raw
    for key, value in (("action_horizon", 8), ("execution_horizon", 4)):
        bad = copy.deepcopy(current)
        bad["artifact"]["spec"][key] = value
        with pytest.raises(ValueError, match="resume metadata"):
            load_training_state(
                tmp_path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                expected_metadata=bad,
            )
    bad = copy.deepcopy(current)
    bad["robot_spec"]["hand_action_names"].reverse()
    with pytest.raises(ValueError, match="resume metadata"):
        load_training_state(
            tmp_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            expected_metadata=bad,
        )


def array_source_factory(cfg):
    return ArraySource(cfg["dataset_root"], synthetic_spec())


def test_custom_factory_and_standalone_prior_windows(tmp_path):
    from examples.embodiment.eval_lamplstm_prior import _build_model
    from examples.embodiment.train_lamplstm_prior import _ensure_data, _load_config
    from rlinf.data.datasets.lamp.action_windows import ActionWindowDataset
    from rlinf.data.datasets.lamp.source_factory import create_lamp_data_source

    cfg = _load_config(None)
    cfg["data"].update(
        source="synthetic",
        source_factory="test_lamp_portable:array_source_factory",
        dataset_root=str(tmp_path),
        action_dim=20,
        horizon=8,
        history_length=3,
    )
    source = create_lamp_data_source(cfg["data"])
    assert source.metadata.robot_spec == synthetic_spec()
    result = _ensure_data(tmp_path / "standalone", cfg)
    assert _ensure_data(tmp_path / "standalone", cfg) == result
    batch = next(
        iter(
            DataLoader(
                ActionWindowDataset(tmp_path / "standalone", "train"), batch_size=2
            )
        )
    )
    assert batch["history"].shape == (2, 3, 18)  # Measured state width is independent.
    model = _build_model(cfg, result["history_feature_dim"])
    output = model(
        batch["history"],
        batch["future_actions"],
        history_mask=batch["history_mask"],
        future_mask=batch["future_mask"],
        sample=False,
    )
    assert torch.isfinite(output.total_loss)
    cfg["data"]["horizon"] = 12
    with pytest.raises(ValueError, match="timing"):
        _ensure_data(tmp_path / "standalone", cfg)


def test_custom_factory_requires_explicit_spec(tmp_path, monkeypatch):
    from rlinf.data.datasets.lamp.source_factory import create_lamp_data_source

    source = ArraySource(tmp_path, dexjoco_robot_spec())
    source.metadata = LampSourceMetadata("legacy", tmp_path, "a", "b", ("a", "b"))
    monkeypatch.setattr("test_lamp_portable.array_source_factory", lambda _: source)
    with pytest.raises(ValueError, match="declare a robot_spec"):
        create_lamp_data_source(
            {"source_factory": "test_lamp_portable:array_source_factory"}
        )


def test_policy_rejects_statistics_width_mismatch():
    base = make_policy(synthetic_spec(), "mlp", 4, 1)
    stats = {
        name.removeprefix("stat_"): value
        for name, value in base.named_buffers()
        if name.startswith("stat_")
    }
    stats["arm_state_pair_mean"] = torch.zeros(6)
    with pytest.raises(ValueError, match="Statistic.*robot spec"):
        LampPolicy(base.core, base.spec, stats)


@pytest.mark.parametrize("prior", ["lamplstm", "pca", "vq_codebook", "mlp"])
@pytest.mark.parametrize("absolute", [False, True])
def test_other_hand_and_state_widths_with_absolute_or_incremental_commands(
    prior, absolute
):
    spec = synthetic_spec(20 if absolute else 16)
    spec = replace(
        spec, arm_state_dim=11, arm_state_names=tuple(f"sensor_{i}" for i in range(11))
    )
    if absolute:
        spec = replace(
            spec,
            arm_action_dim=7,
            arm_action_units=("m",) * 3 + ("unit_quaternion",) * 4,
            action_representation="absolute_pose_quat",
            quaternion_offset=3,
        )
    base = make_policy(spec, prior, 4, 1)
    policy = LampResidualSACPolicy(base, contract_version=5)
    core = torch.zeros(2, 4, base.spec.core_action_dim)
    if absolute:
        core[..., 3] = 1.0
    action, _, _ = policy.sac_forward(
        observation(spec), base_core=core, condition=torch.zeros(2, 256)
    )
    assert action.shape == (2, spec.action_dim)
    assert torch.isfinite(action).all()
    if absolute:
        torch.testing.assert_close(action[:, 3:7].norm(dim=-1), torch.ones(2))


@pytest.mark.parametrize(
    "field,value", [("horizon", 4), ("action_dim", 16), ("history_dim", 20)]
)
def test_prior_robot_and_horizon_mismatch_rejected(field, value):
    cfg = architecture(synthetic_spec(), "lamplstm", 8)
    cfg["lamplstm_model_config"][field] = value
    with pytest.raises(ValueError, match="LSTM"):
        LAMPDiffusionPolicy(**cfg)
