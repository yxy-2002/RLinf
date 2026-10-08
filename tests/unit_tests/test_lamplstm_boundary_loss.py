# Copyright 2026 The RLinf Authors.

"""Execution-boundary losses, including pairing, gradients and opt-out parity."""

from unittest.mock import patch

import pytest
import torch

from rlinf.models.embodiment.lamp.chunk_boundary_loss import (
    chunk_boundary_loss,
    validate_boundary_loss,
)
from rlinf.models.embodiment.lamp.lamplstm_prior import LampLSTMPrior


def _loss(
    current, previous, target, old_target, mask, old_mask, pairs, mode="mse", k=2
):
    return chunk_boundary_loss(
        current,
        previous,
        target,
        old_target,
        mask,
        old_mask,
        pairs,
        execution_horizon=k,
        loss_type=mode,
    )


def test_boundary_compares_last_executed_frame_and_detaches_reference():
    current = torch.tensor([[[5.0], [9.0], [9.0], [9.0]]], requires_grad=True)
    previous = torch.tensor([[[0.0], [2.0], [50.0], [100.0]]], requires_grad=True)
    mask = torch.ones(1, 4)
    loss, fraction = _loss(
        current, previous, current, previous, mask, mask, torch.ones(1)
    )
    assert loss.item() == 9.0  # (5 - 2)^2, not (5 - 100)^2
    assert fraction.item() == 1.0
    loss.backward()
    torch.testing.assert_close(
        current.grad, torch.tensor([[[6.0], [0.0], [0.0], [0.0]]])
    )
    assert previous.grad is None


def test_delta_mse_preserves_expert_motion_and_is_not_first_frame_bc():
    target = torch.tensor([[[3.0], [4.0], [5.0], [6.0]]])
    old_target = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    # Both predictions have a reconstruction offset but the correct increment.
    current, previous = target + 10, old_target + 10
    mask = torch.ones(1, 4)
    pair = torch.ones(1)
    delta, _ = _loss(
        current, previous, target, old_target, mask, mask, pair, "delta_mse"
    )
    direct, _ = _loss(current, previous, target, old_target, mask, mask, pair)
    assert delta.item() == 0.0
    assert direct.item() == 1.0
    assert (current[:, 0] - target[:, 0]).square().mean().item() == 100.0
    previous[:, 1] += 2
    changed, _ = _loss(
        current, previous, target, old_target, mask, mask, pair, "delta_mse"
    )
    assert changed.item() == 4.0


@pytest.mark.parametrize("all_invalid", [False, True])
def test_boundary_masks_reset_padding_and_missing_execution_prefix(all_invalid):
    current = torch.ones(4, 4, 2, requires_grad=True)
    previous = torch.zeros_like(current)
    masks = torch.ones(4, 4)
    old_masks = masks.clone()
    masks[1, 0] = 0
    old_masks[2, 0] = 0
    pairs = torch.tensor([not all_invalid, True, True, False])
    loss, fraction = _loss(
        current, previous, current.detach(), previous, masks, old_masks, pairs
    )
    assert loss.item() == (0.0 if all_invalid else 1.0)
    assert fraction.item() == (0.0 if all_invalid else 0.25)
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[1:].count_nonzero() == 0


@pytest.mark.parametrize("k", [1, 4])
def test_boundary_supports_single_step_and_full_horizon(k):
    values = torch.arange(8.0).reshape(1, 4, 2)
    masks = torch.ones(1, 4)
    loss, _ = _loss(
        values + 8, values, values, values, masks, masks, torch.ones(1), k=k
    )
    assert loss.item() == (8 - 2 * (k - 1)) ** 2


@pytest.mark.parametrize(
    "mode,weight",
    [
        ("bad", 1.0),
        ("none", 1.0),
        ("mse", -1),
        ("mse", float("nan")),
        ("mse", float("inf")),
    ],
)
def test_boundary_rejects_invalid_configuration(mode, weight):
    with pytest.raises(ValueError):
        validate_boundary_loss(mode, weight)


def _model(mode="none", drop=0.0):
    return LampLSTMPrior(
        action_dim=2,
        history_dim=2,
        horizon=4,
        latent_dim=2,
        action_hidden_dim=4,
        condition_hidden_dim=4,
        condition_mode_encoder=mode,
        condition_mode_decoder=mode,
        condition_drop_prob=drop,
    )


def _pair():
    return {
        "history": torch.randn(2, 3, 2),
        "history_mask": torch.ones(2, 3),
        "future_actions": torch.randn(2, 4, 2, requires_grad=True),
        "future_mask": torch.ones(2, 4),
        "pair_mask": torch.tensor([True, False]),
    }


@pytest.mark.parametrize("sample", [True, False])
def test_zero_weight_preserves_original_objective_and_rng(sample):
    model = _model("film", drop=0.5)
    history, future = torch.randn(2, 3, 2), torch.randn(2, 4, 2)
    rng = torch.get_rng_state()
    baseline = model(history, future, history_mask=torch.ones(2, 3), sample=sample)
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    disabled = model(
        history,
        future,
        history_mask=torch.ones(2, 3),
        sample=sample,
        boundary_loss_type="delta_mse",
        boundary_loss_weight=0,
    )
    for key in vars(baseline):
        torch.testing.assert_close(
            getattr(baseline, key), getattr(disabled, key), rtol=0, atol=0
        )
    assert torch.equal(after, torch.get_rng_state())
    torch.testing.assert_close(
        baseline.total_loss,
        baseline.reconstruction_loss + model.beta * baseline.kl_loss,
    )


@pytest.mark.parametrize("mode", ["mse", "delta_mse"])
def test_prior_boundary_uses_means_and_only_adds_configured_objective(mode):
    model = _model()
    history, future, pair = torch.randn(2, 3, 2), torch.randn(2, 4, 2), _pair()
    rng = torch.get_rng_state()
    baseline = model(history, future)
    torch.set_rng_state(rng)
    output = model(
        history,
        future,
        previous_chunk=pair,
        execution_horizon=2,
        boundary_loss_type=mode,
        boundary_loss_weight=0.2,
    )
    current_mean = model(history, future, sample=False)
    previous_mean = model(pair["history"], pair["future_actions"], sample=False)
    expected, _ = _loss(
        current_mean.reconstruction,
        previous_mean.reconstruction,
        future,
        pair["future_actions"],
        torch.ones(2, 4),
        pair["future_mask"],
        pair["pair_mask"],
        mode,
    )
    torch.testing.assert_close(output.boundary_loss, expected)
    torch.testing.assert_close(output.total_loss, baseline.total_loss + 0.2 * expected)
    torch.testing.assert_close(
        output.reconstruction, baseline.reconstruction, rtol=0, atol=0
    )
    output.boundary_loss.backward()
    assert pair["future_actions"].grad is None
    assert model.decoder_output.weight.grad.abs().sum() > 0


def test_boundary_shares_condition_dropout_across_paired_codec_paths():
    model = _model("film", drop=0.5)
    with (
        patch.object(model, "_encode_impl", wraps=model._encode_impl) as encode,
        patch.object(model, "_decode_impl", wraps=model._decode_impl) as decode,
    ):
        model(
            torch.randn(2, 3, 2),
            torch.randn(2, 4, 2),
            history_mask=torch.ones(2, 3),
            previous_chunk=_pair(),
            execution_horizon=2,
            boundary_loss_type="mse",
            boundary_loss_weight=1.0,
        )
    masks = [call.args[3] for call in encode.call_args_list + decode.call_args_list]
    assert len(masks) == 5
    assert masks[0] is not None
    assert all(torch.equal(masks[0], mask) for mask in masks[1:])


def test_enabled_boundary_requires_pair_and_valid_execution_horizon():
    model = _model()
    with pytest.raises(ValueError, match="previous_chunk"):
        model(
            None, torch.zeros(2, 4, 2), boundary_loss_type="mse", boundary_loss_weight=1
        )
    for k in (None, 0, 5):
        with pytest.raises(ValueError, match="execution_horizon"):
            model(
                None,
                torch.zeros(2, 4, 2),
                previous_chunk=_pair(),
                execution_horizon=k,
                boundary_loss_type="mse",
                boundary_loss_weight=1,
            )
