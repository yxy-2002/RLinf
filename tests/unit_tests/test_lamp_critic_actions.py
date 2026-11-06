# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for near-constant hand joints and partial chunks."""

import pytest
import torch
from test_lamp_residual_model import _observation, _policy


def test_critic_hand_scale_floor_preserves_pretrained_statistics():
    policy = _policy("pca")
    base = policy.base_policy
    base.stat_hand_action_mean.fill_(0.0021978)
    base.stat_hand_action_std.fill_(1e-6)
    base.stat_hand_action_std[-1] = 0.2
    base.stat_arm_action_std.fill_(0.005)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    actions = torch.full((2, 8, 23), 0.0061978)
    normalized = policy._physical_action_norm(actions)
    torch.testing.assert_close(normalized[..., 7:-1], torch.full((2, 8, 15), 0.4))
    torch.testing.assert_close(normalized[..., -1], torch.full((2, 8), 0.02))
    torch.testing.assert_close(normalized[..., :7], actions[..., :7] / 0.005)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_partial_chunk_q_ignores_padding_and_its_gradients():
    policy = _policy("pca")
    policy.base_policy.stat_hand_action_mean.fill_(0.0021978)
    policy.base_policy.stat_hand_action_std.fill_(1e-6)
    with torch.no_grad():
        for parameter in policy.q_head.parameters():
            parameter.fill_(0.01)
    obs = _observation()
    valid = torch.arange(8)[None, :] < torch.tensor([[2], [5]])
    actions = torch.full((2, 8, 23), 0.006).requires_grad_()
    changed = actions.detach().clone()
    changed[~valid] = 1000.0
    seen = []
    hook = policy.q_head.register_forward_pre_hook(
        lambda module, args: seen.append(args[1])
    )
    q = policy.sac_q_forward(obs, actions.flatten(1), primitive_valid=valid)
    other = policy.sac_q_forward(obs, changed.flatten(1), primitive_valid=valid)
    hook.remove()
    torch.testing.assert_close(q, other, rtol=0, atol=0)
    assert torch.count_nonzero(seen[0].reshape(2, 8, 23)[~valid]) == 0
    q.sum().backward()
    assert torch.count_nonzero(actions.grad[~valid]) == 0
    assert torch.count_nonzero(actions.grad[valid]) > 0
    full = torch.ones_like(valid)
    torch.testing.assert_close(
        policy.sac_q_forward(obs, actions.flatten(1)),
        policy.sac_q_forward(obs, actions.flatten(1), primitive_valid=full),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("mask", [torch.ones(2, 8), torch.ones(2, 7, dtype=torch.bool)])
def test_q_rejects_invalid_execution_mask(mask):
    policy = _policy("pca")
    with pytest.raises(ValueError, match="primitive_valid"):
        policy.sac_q_forward(_observation(), torch.zeros(2, 184), primitive_valid=mask)


def test_learner_masks_data_q_but_not_next_policy_chunk(monkeypatch):
    from test_lamp_realworld_recipe import learner

    worker, batch = learner(monkeypatch)
    valid = batch["forward_inputs"]["primitive_valid"]
    valid[0, 2:] = False
    valid[1, 5:] = False
    worker.model.base_policy.stat_hand_action_mean.fill_(0.2)
    batch["actions"].fill_(0.1)
    seen = []
    hook = worker.model.q_head.register_forward_pre_hook(
        lambda module, args: seen.append(args[1].detach().clone())
    )
    calls = []
    original = worker.target_model.sac_q_forward

    def target_q(*args, **kwargs):
        calls.append(kwargs.get("primitive_valid"))
        return original(*args, **kwargs)

    monkeypatch.setattr(worker.target_model, "sac_q_forward", target_q)
    loss, _ = worker.forward_critic(batch)
    hook.remove()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(seen[0].reshape(2, 8, 23)[~valid]) == 0
    assert torch.count_nonzero(seen[0].reshape(2, 8, 23)[valid]) > 0
    assert calls == [None]
