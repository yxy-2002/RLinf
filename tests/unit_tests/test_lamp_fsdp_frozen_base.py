# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""CUDA regression for frozen LAMP priors inside an FSDP residual actor."""

import pytest
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from test_lamp_refactor import ROOT, make_base, observation
from torch.distributed.device_mesh import init_device_mesh

from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.backends.cudnn.is_available(),
    reason="Requires CUDA and cuDNN to exercise eval-mode RNN backward",
)
def test_fsdp_keeps_base_frozen_while_updating_residual_actor(tmp_path, monkeypatch):
    """Keep frozen RNN views off the graph without detaching latent actions."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
    )
    try:
        with initialize_config_dir(
            config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
        ):
            cfg = compose(config_name="dexjoco_lamp_residual_sac")
        policy = LampResidualSACPolicy(make_base()[0])
        frozen_before = {
            name: tensor.detach().cpu().clone()
            for name, tensor in policy.base_policy.state_dict().items()
        }
        strategy = FSDPStrategy(cfg.actor, world_size=1)
        wrapped = strategy.wrap_model(policy, init_device_mesh("cuda", (1,)))
        wrapped.train()
        optimizer = torch.optim.Adam(policy.residual_actor.parameters(), lr=1e-3)
        actor_before = policy.residual_actor.mean_head.weight.detach().clone()
        observed_flags = []

        def record_rnn_flags(module, args):
            observed_flags.append((module.training, torch.is_grad_enabled()))

        hook = policy.base_policy.core.lamplstm.history_encoder.lstm.register_forward_pre_hook(
            record_rnn_flags
        )
        obs = {key: value.cuda() for key, value in observation().items()}
        core = torch.zeros(2, 16, 9, device="cuda")
        core[..., 3] = 1
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            action, log_pi, context = wrapped(
                forward_type=ForwardType.SAC,
                obs=obs,
                base_core=core,
                condition=torch.zeros(2, 256, device="cuda"),
                deterministic=True,
            )
            if step == 0:
                # Isolate the hand decoder's gradient, without entropy or Q.
                loss = action.reshape(2, 8, 23)[..., 7:].sum()
            else:
                q_values = wrapped(
                    forward_type=ForwardType.SAC_Q,
                    obs=obs,
                    actions=action,
                    shared_feature=context,
                    detach_encoder=True,
                )
                loss = (
                    0.01 * log_pi - q_values.min(dim=-1, keepdim=True).values
                ).mean()
            loss.backward()
            grad = policy.residual_actor.mean_head.weight.grad
            assert grad is not None and torch.isfinite(grad).all()
            assert grad.reshape(16, 9, -1)[:8, 7:].abs().sum() > 0
            assert all(p.grad is None for p in policy.base_policy.parameters())
            optimizer.step()
        hook.remove()
        assert observed_flags and all(
            flags == (False, False) for flags in observed_flags
        )
        assert not torch.equal(actor_before, policy.residual_actor.mean_head.weight)
        assert all(not p.requires_grad for p in policy.base_policy.parameters())
        for name, tensor in policy.base_policy.state_dict().items():
            torch.testing.assert_close(
                tensor.cpu(), frozen_before[name], rtol=0, atol=0
            )
        # Frozen parameters remain in the normal FSDP checkpoint.
        saved = wrapped.state_dict()
        for name, tensor in frozen_before.items():
            torch.testing.assert_close(
                saved[f"base_policy.{name}"].cpu(), tensor, rtol=0, atol=0
            )
    finally:
        dist.destroy_process_group()
