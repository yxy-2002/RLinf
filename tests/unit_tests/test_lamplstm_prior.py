# Copyright 2026 The RLinf Authors.

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import torch

from rlinf.data.datasets.lamp.action_windows import (
    build_action_windows,
)
from rlinf.models.embodiment.lamp.hand_prior_artifact import build_prior_model
from rlinf.models.embodiment.lamp.lamplstm_prior import LampLSTMPrior


@pytest.mark.parametrize(
    ("encoder_mode", "decoder_mode"),
    list(
        product(
            ("none", "concat", "film"),
            repeat=2,
        )
    ),
)
def test_lamplstm_modes_preserve_chunk_shape(encoder_mode, decoder_mode) -> None:
    torch.manual_seed(0)
    model = LampLSTMPrior(
        action_dim=16,
        history_dim=16,
        horizon=16,
        latent_dim=3,
        action_hidden_dim=8,
        condition_hidden_dim=8,
        condition_mode_encoder=encoder_mode,
        condition_mode_decoder=decoder_mode,
    )
    history = torch.randn(2, 16, 16)
    future = torch.randn(2, 16, 16)
    output = model(history, future, torch.ones(2, 16), torch.ones(2, 16))
    assert output.mu.shape == (2, 16, 3)
    assert output.log_var.shape == (2, 16, 3)
    assert output.sampled_latent.shape == (2, 16, 3)
    assert output.reconstruction.shape == (2, 16, 16)
    assert torch.isfinite(output.total_loss)


def test_no_condition_mode_does_not_require_history() -> None:
    model = LampLSTMPrior(action_hidden_dim=8, condition_hidden_dim=8)
    future = torch.randn(2, 16, 16)
    output = model(None, future, future_mask=torch.ones(2, 16))
    assert output.reconstruction.shape == (2, 16, 16)


def test_mask_excludes_padded_future_from_reconstruction_and_kl() -> None:
    model = LampLSTMPrior(action_hidden_dim=8, condition_hidden_dim=8)
    future = torch.randn(1, 16, 16)
    mask = torch.zeros(1, 16)
    mask[:, :4] = 1.0
    output = model(None, future, future_mask=mask)
    expected = ((output.reconstruction[:, :4] - future[:, :4]).square()).mean()
    torch.testing.assert_close(output.reconstruction_loss, expected)


@pytest.mark.parametrize("mode", ["concat", "film"])
def test_history_padding_values_do_not_change_condition_context(mode) -> None:
    torch.manual_seed(4)
    model = LampLSTMPrior(
        action_dim=3,
        history_dim=3,
        horizon=4,
        latent_dim=2,
        action_hidden_dim=8,
        condition_hidden_dim=8,
        condition_mode_encoder=mode,
        condition_mode_decoder="none",
    )
    future = torch.rand(2, 4, 3)
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    history = torch.rand(2, 4, 3)
    changed = history.clone()
    changed[0, :2] = torch.randn_like(changed[0, :2]) * 100.0
    mu_a, _ = model.encode(history, future, mask)
    mu_b, _ = model.encode(changed, future, mask)
    assert torch.allclose(mu_a, mu_b)


def test_action_windows_do_not_cross_episode_boundaries() -> None:
    actions = np.arange(10, dtype=np.float32)[:, None]
    episodes = np.array([0] * 5 + [1] * 5)
    windows = build_action_windows(actions, episodes, history_length=3, horizon=3)
    np.testing.assert_array_equal(windows["history"][5, :, 0], [5, 5, 5])
    np.testing.assert_array_equal(windows["history_mask"][5], [0, 0, 0])
    np.testing.assert_array_equal(windows["future"][4, :, 0], [4, 4, 4])
    np.testing.assert_array_equal(windows["future_mask"][4], [1, 0, 0])


def test_lamplstm_prior_registry_builds_model() -> None:
    model = build_prior_model(
        "lamplstm",
        {
            "action_dim": 16,
            "history_dim": 16,
            "horizon": 16,
            "latent_dim": 2,
            "action_hidden_dim": 8,
            "condition_hidden_dim": 8,
            "condition_mode_encoder": "film",
            "condition_mode_decoder": "film",
            "num_lstm_layers": 1,
            "beta": 5e-4,
            "condition_drop_prob": 0.2,
        },
    )
    assert isinstance(model, LampLSTMPrior)


@pytest.mark.parametrize("mode", ["concat", "film"])
def test_forward_shares_condition_dropout_between_encoder_and_decoder(mode) -> None:
    """Mixed dropout must match two consistent deterministic codec paths."""
    from unittest.mock import patch

    torch.manual_seed(7)
    model = LampLSTMPrior(
        action_dim=3,
        history_dim=3,
        horizon=4,
        latent_dim=2,
        action_hidden_dim=8,
        condition_hidden_dim=8,
        condition_mode_encoder=mode,
        condition_mode_decoder=mode,
        condition_drop_prob=0.5,
    )
    history, future = torch.randn(2, 3, 3), torch.randn(2, 4, 3)
    mask = torch.ones(2, 3)
    model.eval()
    kept = model(history, future, mask, sample=False)
    dropped = model(history, future, torch.zeros_like(mask), sample=False)
    model.train()
    with patch("torch.rand", return_value=torch.tensor([[0.1], [0.9]])) as draw:
        output = model(history, future, mask, sample=False)
    assert draw.call_count == 1
    torch.testing.assert_close(output.mu[0], dropped.mu[0])
    torch.testing.assert_close(output.mu[1], kept.mu[1])
    torch.testing.assert_close(output.reconstruction[0], dropped.reconstruction[0])
    torch.testing.assert_close(output.reconstruction[1], kept.reconstruction[1])
    output.total_loss.backward()
    assert model.history_encoder.input_projection.weight.grad is not None


@pytest.mark.parametrize("probability", [0.0, 1.0])
@pytest.mark.parametrize("mode", ["film", "concat"])
def test_condition_dropout_extremes_and_eval(probability, mode) -> None:
    """Zero/full dropout follow their reference paths and eval never drops."""
    torch.manual_seed(11)
    model = LampLSTMPrior(
        action_dim=3,
        history_dim=3,
        horizon=4,
        latent_dim=2,
        action_hidden_dim=8,
        condition_hidden_dim=8,
        condition_mode_encoder=mode,
        condition_mode_decoder=mode,
        condition_drop_prob=probability,
    )
    history, future = torch.randn(2, 3, 3), torch.randn(2, 4, 3)
    mask = torch.ones(2, 3)
    model.eval()
    reference = model(history, future, mask * (1 - probability), sample=False)
    full = model(history, future, mask, sample=False)
    model.train()
    output = model(history, future, mask, sample=False)
    torch.testing.assert_close(output.reconstruction, reference.reconstruction)
    model.eval()
    torch.testing.assert_close(
        model(history, future, mask, sample=False).reconstruction, full.reconstruction
    )


def test_action_artifact_roundtrip_preserves_episode_split(tmp_path):
    from rlinf.data.datasets.lamp.action_windows import (
        ActionWindowDataset,
        save_action_artifact,
    )

    actions = np.arange(160, dtype=np.float32).reshape(10, 16)
    episodes = np.repeat([0, 1], 5)
    save_action_artifact(
        tmp_path,
        actions,
        episodes,
        history_length=3,
        horizon=3,
        train_ratio=0.5,
        source_metadata={"type": "dexjoco_lerobot"},
    )
    train = ActionWindowDataset(tmp_path, "train")
    validation = ActionWindowDataset(tmp_path, "validation")
    assert len(train) == len(validation) == 5
    assert set(episodes[train.indices]).isdisjoint(episodes[validation.indices])
    assert train[0]["history"].shape == (3, 16)
    assert train[0]["future_actions"].shape == (3, 16)


def test_standalone_default_is_real_data_and_rejects_removed_source(tmp_path):
    from examples.embodiment.train_lamplstm_prior import _ensure_data, _load_config

    config = _load_config(None)
    assert config["data"]["source"] == "dexjoco_lerobot"
    config["data"]["source"] = "unsupported_debug_source"
    with pytest.raises(ValueError, match="dexjoco_lerobot"):
        _ensure_data(tmp_path, config)


def test_action_loader_and_training_reject_legacy_debug_cache(tmp_path):
    from examples.embodiment.train_lamplstm_prior import _ensure_data, _load_config
    from rlinf.data.datasets.lamp.action_windows import ActionWindowDataset

    (tmp_path / "metadata.json").write_text('{"phases_file": "phases.npy"}')
    np.save(tmp_path / "sequence.npy", np.zeros((10, 16), np.float32))
    with pytest.raises(ValueError, match="DexJoCo LeRobot"):
        ActionWindowDataset(tmp_path, "all")
    with pytest.raises(ValueError, match="DexJoCo LeRobot"):
        _ensure_data(tmp_path, _load_config(None))


@pytest.mark.parametrize("mode", ["concat", "film"])
def test_conditioned_prior_rejects_missing_history_mask(mode):
    model = LampLSTMPrior(
        action_hidden_dim=8,
        condition_hidden_dim=8,
        condition_mode_decoder=mode,
    )
    with pytest.raises(ValueError, match="history_mask is required"):
        model.decode(torch.zeros(2, 16, model.latent_dim), torch.zeros(2, 8, 16))


@pytest.mark.parametrize("frozen_history", [False, True])
def test_decoder_history_gradient_contract_preserves_latent_gradient(frozen_history):
    model = LampLSTMPrior(
        action_hidden_dim=8,
        condition_hidden_dim=8,
        condition_mode_decoder="film",
        condition_drop_prob=0.0,
    )
    history = torch.randn(2, 8, 16, requires_grad=True)
    latent = torch.randn(2, 16, model.latent_dim, requires_grad=True)
    action = model.decoder.decode(
        latent, history, torch.ones(2, 8), frozen_history=frozen_history
    )
    action.square().sum().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    history_grad = model.history_encoder.input_projection.weight.grad
    if frozen_history:
        assert history.grad is None
        assert history_grad is None
    else:
        assert history.grad is not None and history.grad.abs().sum() > 0
        assert history_grad is not None and history_grad.abs().sum() > 0
