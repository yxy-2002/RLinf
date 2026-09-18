# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Regression coverage for the unified DP and explicit residual decoder context."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import ResNetConfig

from rlinf.models.embodiment.lamp import get_model
from rlinf.models.embodiment.lamp.artifact_io import save_artifact
from rlinf.models.embodiment.lamp.policy_wrapper import LampPolicy, LampPolicySpec
from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
    LAMPDiffusionPolicy,
    vq_index_to_normalized,
    vq_normalized_to_index,
)

ROOT = Path(__file__).resolve().parents[2]


def make_base(prior="lamplstm", latent_dim=2, decoder_mode="film"):
    dim = 23 if prior == "mlp" else 8 if prior == "vq_codebook" else 7 + latent_dim
    architecture = {
        "backbone_config": ResNetConfig(
            depths=[1, 1, 1, 1], hidden_sizes=[64, 128, 256, 512]
        ).to_dict(),
        "hand_prior_source": prior,
        "core_action_mean": [0.0] * dim,
        "core_action_std": [1.0] * dim,
        "hand_action_mean": [0.0] * 16,
        "hand_action_std": [1.0] * 16,
    }
    if prior == "lamplstm":
        architecture.update(
            lamplstm_model_config={
                "latent_dim": latent_dim,
                "action_hidden_dim": 8,
                "condition_hidden_dim": 8,
                "condition_mode_encoder": "film",
                "condition_mode_decoder": decoder_mode,
            },
            decoder_history_contract="primitive_v1",
            decoder_history_length=8,
        )
    elif prior == "pca":
        architecture.update(
            pca_mean=[0.0] * 16,
            pca_components=torch.eye(16)[:latent_dim].tolist(),
            pca_latent_dim=latent_dim,
        )
    elif prior == "vq_codebook":
        architecture["vq_codebook"] = (
            torch.arange(16.0)[:, None] * torch.linspace(0.1, 1.0, 16)[None]
        ).tolist()
    core = LAMPDiffusionPolicy(**architecture).eval()
    spec = LampPolicySpec(
        task="water_plant",
        policy_family="dp",
        embodiment="single",
        hand_prior_type=prior,
        action_horizon=16,
        execution_horizon=8,
        core_action_dim=dim,
        physical_action_dim=23,
        image_size=32,
        image_keys=("front", "wrist"),
        latent_dims={"single": core._hand_latent_dim()},
        policy_version=2,
    )
    stats = {}
    for name, size in [
        ("arm_state_pair", 7),
        ("hand_state_pair", 16),
        ("arm_action", 7),
        ("hand_action", 16),
    ]:
        stats[name + "_mean"] = [0.0] * size
        stats[name + "_std"] = [1.0] * size
    return (
        LampPolicy(core, spec, stats, use_temporal_ensemble=False),
        architecture,
        stats,
    )


def observation(batch=2):
    return {
        "main_images": torch.zeros(batch, 32, 32, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(batch, 32, 32, 3, dtype=torch.uint8),
        "panda_qpos_pair": torch.zeros(batch, 2, 7),
        "hand_state_pair": torch.zeros(batch, 2, 16),
        "hand_history": torch.randn(batch, 8, 16),
        "hand_history_mask": torch.ones(batch, 8),
    }


@pytest.mark.parametrize("prior", ["lamplstm", "pca", "vq_codebook", "mlp"])
def test_residual_dimensions_zero_equivalence_and_gradients(prior):
    base, _, _ = make_base(prior)
    policy = LampResidualSACPolicy(base)
    obs = observation()
    core = torch.randn(2, 16, base.spec.core_action_dim) * 0.1
    core[..., 3] = 1
    ctx = policy._build_context(obs, base_core=core, condition=torch.zeros(2, 256))
    full, executed, _, _, _, _, _ = policy._actor_plan(
        ctx, deterministic=True, residual_mode="zero"
    )
    h, mask = base.decoder_context(obs)
    expected = base.decode_core_action(
        core, decoder_history=h, decoder_history_mask=mask
    )
    torch.testing.assert_close(full, expected, rtol=0, atol=0)
    assert executed.shape == (2, 8, 23)
    assert policy.causal_action_dim == 8 * base.spec.core_action_dim
    assert not policy.causal_mask[8:].any()
    # Use a controlled physical-action objective to verify the path to the latent actor.
    _, action, *_ = policy._actor_plan(ctx, deterministic=True)
    action[..., 7:].sum().backward()
    grad = policy.residual_actor.mean_head.weight.grad.reshape(
        16, base.spec.core_action_dim, -1
    )
    assert grad[:8, 7:].abs().sum() > 0
    assert grad[8:].eq(0).all()
    assert all(p.grad is None for p in base.parameters())


@pytest.mark.parametrize("mode", ["none", "concat", "film"])
@pytest.mark.parametrize("latent_dim", [2, 6])
def test_explicit_history_survives_interleaving_and_future_latent_is_causal(
    mode, latent_dim
):
    base, _, _ = make_base(latent_dim=latent_dim, decoder_mode=mode)
    a, b = observation(), observation()
    za = torch.randn(2, 16, 7 + latent_dim)
    ha, ma = base.decoder_context(a)
    hb, mb = base.decoder_context(b)
    first = base.decode_core_action(za, decoder_history=ha, decoder_history_mask=ma)
    base.decode_core_action(za, decoder_history=hb, decoder_history_mask=mb)
    again = base.decode_core_action(za, decoder_history=ha, decoder_history_mask=ma)
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    altered = za.clone()
    altered[:, 8:, 7:] += 5
    last = base.decode_core_action(altered, decoder_history=ha, decoder_history_mask=ma)
    torch.testing.assert_close(first[:, :8], last[:, :8], rtol=0, atol=0)
    if mode != "none":
        with pytest.raises(ValueError, match="history"):
            base.decode_core_action(za)


def test_vq_half_up_and_straight_through_gradient():
    raw = torch.tensor([-2.0, 0.0, 0.49, 0.5, 1.5, 14.5, 15.0, 20.0])
    torch.testing.assert_close(
        vq_normalized_to_index(vq_index_to_normalized(raw)),
        torch.tensor([0, 0, 0, 1, 2, 15, 15, 15]),
    )
    base, _, _ = make_base("vq_codebook")
    z = torch.zeros(1, 16, 8, requires_grad=True)
    hard = base.core._decode_core(z)[0]
    ste = base.core._decode_core(z, vq_straight_through=True)[0]
    torch.testing.assert_close(hard, ste, rtol=0, atol=0)
    ste[..., 7:].sum().backward()
    expected = 7.5 * (base.core.vq_codebook[8] - base.core.vq_codebook[7]).sum()
    torch.testing.assert_close(z.grad[..., 7], torch.full((1, 16), expected))
    assert not base.core.vq_codebook.requires_grad


def test_old_v2_deployment_loads_strictly_without_rewriting(tmp_path):
    base, arch, stats = make_base()
    save_artifact(
        tmp_path,
        model=base,
        metadata={
            "kind": "policy",
            "model_type": "lamp_dp_v2",
            "policy_version": 2,
            "architecture": arch,
            "spec": vars(base.spec),
        },
        statistics=stats,
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    loaded = get_model(
        OmegaConf.create(
            {
                "model_type": "lamp_dp",
                "model_path": str(tmp_path),
                "use_temporal_ensemble": False,
            }
        )
    )
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize(
    "name",
    sorted(p.stem for p in (ROOT / "examples/embodiment/config").glob("*lamp*.yaml")),
)
def test_all_active_lamp_configs_compose(name):
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name=name)
    if "actor" in cfg and "model" in cfg.actor and "hand_prior" in cfg.actor.model:
        prior = cfg.actor.model.hand_prior
        if prior is not None:
            assert prior.type in ("lamplstm", "pca", "vq", "vq_codebook", "mlp")


def test_replay_history_follows_batch_permutation_with_cached_features():
    base, _, _ = make_base()
    policy = LampResidualSACPolicy(base)
    obs = observation(3)
    core = torch.randn(3, 16, 9) * 0.1
    core[..., 3] = 1
    features = torch.randn(3, 256)
    first, _, _ = policy.sac_forward(
        obs, deterministic=True, base_core=core, condition=features
    )
    order = torch.tensor([2, 0, 1])
    shuffled = {key: value[order] for key, value in obs.items()}
    other, _, _ = policy.sac_forward(
        shuffled, deterministic=True, base_core=core[order], condition=features[order]
    )
    torch.testing.assert_close(first[order], other)
    # Recompute only a missing cache row using that row's history, not the full batch.
    mixed, _, ctx = policy.sac_forward(
        shuffled,
        deterministic=True,
        base_core=core[order],
        condition=features[order],
        base_cache_valid=torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(mixed[[0, 2]], other[[0, 2]])
    torch.testing.assert_close(ctx["decoder_history"], shuffled["hand_history"])


def test_new_training_state_resume_and_explicit_old_state_rejection(tmp_path):
    from rlinf.models.embodiment.lamp.artifact_io import (
        load_training_state,
        save_training_state,
    )

    model = torch.nn.Linear(2, 1)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    optim.step()
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    args = {
        "model": model,
        "optimizer": optim,
        "scheduler": None,
        "global_step": 1,
        "sampler_state": {"consumed_batches": 1},
    }
    save_training_state(
        tmp_path / "new", metadata={"training_schema_version": 2}, **args
    )
    with torch.no_grad():
        model.weight.zero_()
    step, sampler = load_training_state(
        tmp_path / "new",
        model=model,
        optimizer=optim,
        scheduler=None,
        expected_metadata={"training_schema_version": 2},
    )
    assert step == 1 and sampler == {"consumed_batches": 1}
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    save_training_state(tmp_path / "old", metadata={}, **args)
    with pytest.raises(ValueError, match="Old LAMP training checkpoints"):
        load_training_state(
            tmp_path / "old",
            model=model,
            optimizer=optim,
            scheduler=None,
            expected_metadata={"training_schema_version": 2},
        )


@pytest.mark.parametrize("encoder_mode", ["none", "concat", "film"])
@pytest.mark.parametrize("decoder_mode", ["none", "concat", "film"])
def test_independent_prior_condition_modes(encoder_mode, decoder_mode):
    from rlinf.models.embodiment.lamp.lamplstm_prior import LampLSTMPrior

    prior = LampLSTMPrior(
        latent_dim=2,
        action_hidden_dim=8,
        condition_hidden_dim=8,
        condition_mode_encoder=encoder_mode,
        condition_mode_decoder=decoder_mode,
    ).eval()
    history = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.bool)
    actions = torch.randn(2, 16, 16)
    mu, _ = prior.encode(
        history if encoder_mode != "none" else None,
        actions,
        mask if encoder_mode != "none" else None,
    )
    decoded = prior.decode(
        mu,
        history if decoder_mode != "none" else None,
        mask if decoder_mode != "none" else None,
    )
    assert decoded.shape == actions.shape
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize(
    "name", sorted(p.stem for p in (ROOT / "evaluations/dexjoco").glob("*lamp*.yaml"))
)
def test_active_lamp_evaluation_configs_compose(name, monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(ROOT / "evaluations/dexjoco"), version_base="1.1"
    ):
        cfg = compose(config_name=name)
    assert cfg.rollout.model.model_type == "lamp_dp"


def test_lstm_training_resume_reproduces_next_update_and_ema(tmp_path):
    import copy

    from rlinf.models.embodiment.lamp.artifact_io import (
        load_training_state,
        save_training_state,
    )
    from rlinf.models.embodiment.lamp.lamplstm_prior import LampLSTMPrior

    def build():
        model = LampLSTMPrior(
            latent_dim=2,
            action_hidden_dim=8,
            condition_hidden_dim=8,
            condition_mode_encoder="film",
            condition_mode_decoder="film",
            condition_drop_prob=0.1,
        ).train()
        return (
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            copy.deepcopy(model),
        )

    history, actions = torch.randn(2, 8, 16), torch.randn(2, 16, 16)
    mask = torch.ones(2, 8)

    def update(model, optimizer, ema):
        optimizer.zero_grad()
        loss = model(history, actions, history_mask=mask).total_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 0.1)
        return loss.detach()

    model, optimizer, ema = build()
    update(model, optimizer, ema)
    metadata = {"training_schema_version": 2, "stage": "prior"}
    save_training_state(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        global_step=1,
        sampler_state={"consumed_batches": 1},
        metadata=metadata,
        ema_model=ema,
    )
    expected_loss = update(model, optimizer, ema)
    restored, restored_optimizer, restored_ema = build()
    load_training_state(
        tmp_path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=None,
        expected_metadata=metadata,
        ema_model=restored_ema,
    )
    actual_loss = update(restored, restored_optimizer, restored_ema)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for expected, actual in [(model, restored), (ema, restored_ema)]:
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["none", "concat", "film"])
@pytest.mark.parametrize("missing", ["hand_history", "hand_history_mask"])
def test_lstm_observation_requires_explicit_history_and_mask(mode, missing):
    base, _, _ = make_base(decoder_mode=mode)
    obs = observation()
    del obs[missing]
    with pytest.raises(ValueError, match="hand_history_mask"):
        base.decoder_context(obs)


@pytest.mark.parametrize("mode", ["none", "concat", "film"])
def test_lstm_decode_rejects_missing_mask_even_after_valid_context(mode):
    base, _, _ = make_base(decoder_mode=mode)
    history, mask = base.decoder_context(observation())
    base.core.set_decoder_history(history, mask)
    with pytest.raises(ValueError, match="history and mask"):
        base.decode_core_action(torch.zeros(2, 16, 9), decoder_history=history)
    with pytest.raises(ValueError, match="history and mask"):
        base.core.set_decoder_history(history, None)


def test_dp_defaults_to_primitive_history_and_rejects_legacy():
    base, architecture, _ = make_base(prior="mlp")
    assert base.core.decoder_history_contract == "primitive_v1"
    assert base.core.decoder_history_length == 8
    with pytest.raises(ValueError, match="primitive_v1"):
        LAMPDiffusionPolicy(**architecture, decoder_history_contract="legacy")
