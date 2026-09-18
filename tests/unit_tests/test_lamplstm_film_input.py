# Copyright 2026 The RLinf Authors.
"""Verify FiLM placement, checkpoint compatibility, and real DP decoder wiring."""

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.lamp.artifact_io import save_artifact
from rlinf.models.embodiment.lamp.hand_prior_artifact import load_prior_artifact
from rlinf.models.embodiment.lamp.lamplstm_prior import (
    LampLSTMPrior,
)


def small_model(mode: str, layers: int = 1) -> LampLSTMPrior:
    """Build a small codec using the production hand and horizon dimensions."""
    return LampLSTMPrior(
        action_hidden_dim=8,
        condition_hidden_dim=8,
        condition_mode_encoder=mode,
        condition_mode_decoder=mode,
        num_lstm_layers=layers,
    ).eval()


@pytest.mark.parametrize("mode", ["film", "none"])
@pytest.mark.parametrize("layers", [1, 2])
def test_history_changes_recurrent_inputs_only_for_input_film(mode, layers) -> None:
    torch.manual_seed(51)
    model = small_model(mode, layers)
    future, z = torch.randn(2, 16, 16), torch.randn(2, 16, 2)
    history = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8)
    enc_inputs, enc_states, dec_inputs, dec_states = [], [], [], []

    def encoder_hook(module, args, output):
        enc_inputs.append(args[0].detach().clone())
        enc_states.append(torch.stack(output[1]).detach().clone())

    def decoder_hook(module, args, output):
        dec_inputs.append(args[0].detach().clone())
        dec_states.append(torch.stack(output).detach().clone())

    e = model.action_encoder.register_forward_hook(encoder_hook)
    d = model.decoder_cells[0].register_forward_hook(decoder_hook)
    with torch.no_grad():
        for h in (history, history + 4):
            model.encode(h, future, mask)
            model.decode(z, h, mask)
    e.remove()
    d.remove()
    # The first decoder step fixes latent, start action and recurrent states.
    # Later output-FiLM steps can depend on history through previous actions.
    for a, b in (
        (enc_inputs[0], enc_inputs[1]),
        (enc_states[0], enc_states[1]),
        (dec_inputs[0], dec_inputs[16]),
        (dec_states[0], dec_states[16]),
    ):
        if mode == "film":
            assert not torch.allclose(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("layers", [1, 2])
def test_input_film_formula_and_single_injection(layers) -> None:
    model = small_model("film", layers)
    # Constant, nonidentity modulation makes placement and repetition observable.
    with torch.no_grad():
        for head in (model.encoder_input_film_head, model.decoder_input_film_head):
            head.weight.zero_()
            head.bias[:8].fill_(0.5)
            head.bias[8:].fill_(-0.25)
    h, a, z = torch.randn(2, 8, 16), torch.randn(2, 16, 16), torch.randn(2, 16, 2)
    enc, dec, raw_dec, calls = [], [], [], []
    handles = [
        model.action_encoder.register_forward_pre_hook(
            lambda m, x: enc.append(x[0].detach())
        ),
        model.decoder_input.register_forward_hook(
            lambda m, x, y: raw_dec.append(y.detach())
        ),
        model.decoder_cells[0].register_forward_pre_hook(
            lambda m, x: dec.append(x[0].detach())
        ),
        model.encoder_input_film_head.register_forward_hook(
            lambda m, x, y: calls.append("e")
        ),
        model.decoder_input_film_head.register_forward_hook(
            lambda m, x, y: calls.append("d")
        ),
    ]
    with torch.no_grad():
        model.encode(h, a, torch.ones(2, 8))
        model.decode(z, h, torch.ones(2, 8))
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(enc[0], model.encoder_input(a) * 1.5 - 0.25)
    for raw, fused in zip(raw_dec, dec):
        torch.testing.assert_close(fused, raw * 1.5 - 0.25)
    assert calls.count("e") == 1 and calls.count("d") == 16
    assert not hasattr(model, "encoder_film_head")
    assert not hasattr(model, "decoder_film_head")


def test_zero_film_is_identity_and_gradients_reach_both_heads() -> None:
    model = small_model("film")
    plain = small_model("none")
    plain.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k in plain.state_dict()}
    )
    h, a = torch.randn(2, 8, 16), torch.randn(2, 16, 16)
    with torch.no_grad():
        for head in (model.encoder_input_film_head, model.decoder_input_film_head):
            head.weight.zero_()
            head.bias.zero_()
    actual = model(h, a, history_mask=torch.ones(2, 8), sample=False)
    expected = plain(None, a, sample=False)
    torch.testing.assert_close(actual.mu, expected.mu, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.reconstruction, expected.reconstruction, rtol=0, atol=0
    )
    actual.total_loss.backward()
    for head in (model.encoder_input_film_head, model.decoder_input_film_head):
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    # With ordinary initialized heads, history also receives a training signal.
    model = small_model("film")
    model(h, a, history_mask=torch.ones(2, 8), sample=False).total_loss.backward()
    grad = model.history_encoder.input_projection.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["film_input", "film_output", "init_state"])
@pytest.mark.parametrize("side", ["encoder", "decoder"])
def test_removed_mode_names_are_rejected(mode, side) -> None:
    with pytest.raises(ValueError, match="unsupported condition mode"):
        LampLSTMPrior(**{f"condition_mode_{side}": mode})


def test_artifact_roundtrip_and_dp_decoder(tmp_path) -> None:
    from transformers import ResNetConfig

    from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
        LAMPDiffusionPolicy,
    )

    arch = {
        "action_hidden_dim": 8,
        "condition_hidden_dim": 8,
        "condition_mode_encoder": "film",
        "condition_mode_decoder": "film",
    }
    prior = LampLSTMPrior(**arch).eval()
    save_artifact(
        tmp_path,
        model=prior,
        metadata={
            "kind": "prior",
            "prior_type": "lamplstm",
            "architecture": arch,
            "task": "water_plant",
            "history_contract": "primitive_v1",
            "history_length": 8,
        },
        statistics={},
    )
    loaded, meta, _ = load_prior_artifact(tmp_path, expected_type="lamplstm")
    assert meta["architecture"]["condition_mode_encoder"] == "film"
    assert loaded.condition_mode_encoder == loaded.condition_mode_decoder == "film"
    core = LAMPDiffusionPolicy(
        ResNetConfig(depths=[1, 1, 1, 1], hidden_sizes=[64, 128, 256, 512]).to_dict(),
        hand_prior_source="lamplstm",
        lamplstm_model_config=meta["architecture"],
        decoder_history_contract="primitive_v1",
        decoder_history_length=8,
        core_action_mean=np.arange(9, dtype=np.float32) * 0.1,
        core_action_std=np.arange(9, dtype=np.float32) * 0.2 + 0.5,
        hand_action_mean=np.arange(16, dtype=np.float32) * 0.1,
        hand_action_std=np.arange(16, dtype=np.float32) * 0.1 + 0.5,
    ).eval()
    core.lamplstm.load_state_dict(loaded.state_dict(), strict=True)
    h, mask = torch.randn(2, 8, 16), torch.ones(2, 8)
    normalized = torch.randn(2, 16, 9)
    core.set_decoder_history(h, mask)
    with torch.no_grad():
        pred, aux = core._decode_core(normalized, h, mask)
        z = (normalized * core.core_action_std + core.core_action_mean)[..., -2:]
        direct = prior.decode(z, h, mask)
    torch.testing.assert_close(aux["hand_action_norm"], direct, rtol=0, atol=0)
    torch.testing.assert_close(
        pred[..., 7:], direct * core.hand_action_std + core.hand_action_mean
    )
    assert pred.shape == (2, 16, 23)


def test_training_entrypoints_preserve_input_mode() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from examples.embodiment.train_lamplstm_prior import _build_model
    from rlinf.workers.actor.lamp_il_worker import _prior_architecture

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="dexjoco_lamp_prior_lamplstm_water_plant")
    pc = cfg.actor.model.hand_prior
    assert cfg.data.history_contract == "primitive_v1"
    arch = _prior_architecture("lamplstm", pc)
    assert arch["condition_mode_encoder"] == arch["condition_mode_decoder"] == "film"
    assert (
        pc.history_length == 8 and cfg.actor.optim.lr == 5e-5 and cfg.actor.seed == 42
    )
    standalone = _build_model(
        {
            "data": {"action_dim": 16, "horizon": 16},
            "model": OmegaConf.to_container(pc),
            "loss": {"beta": pc.beta},
        }
    )
    assert standalone.encoder_input_film_head is not None
    assert standalone.decoder_input_film_head is not None
