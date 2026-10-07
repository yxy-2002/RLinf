# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Evaluation must count episodes ending before a cancelled chunk suffix."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.workers.env.env_worker import EnvWorker


@pytest.mark.parametrize("model_type", ["lamp_dp", "lamp_residual_sac"])
@pytest.mark.parametrize("auto_reset", [False, True])
def test_eval_counts_all_chunk_terminal_positions(monkeypatch, model_type, auto_reset):
    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.prepare_actions",
        lambda **kw: kw["raw_chunk_actions"],
    )
    # One termination and one truncation at each of the eight chunk positions.
    terminated = torch.zeros(16, 8, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated[torch.arange(8), torch.arange(8)] = True
    truncated[torch.arange(8, 16), torch.arange(8)] = True
    episode = {
        "success_once": torch.cat([torch.ones(8), torch.zeros(8)]),
        "episode_len": torch.arange(1, 9).repeat(2),
        "return": torch.cat([torch.ones(8), torch.zeros(8)]),
    }
    infos = {"final_info": {"episode": episode}} if auto_reset else {"episode": episode}
    env = SimpleNamespace(
        chunk_step=lambda _: ([{}], None, terminated, truncated, [infos])
    )
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {"env": {"eval": {"env_type": "dexjoco", "auto_reset": auto_reset}}}
        ),
        model_cfg=OmegaConf.create(
            {"model_type": model_type, "num_action_chunks": 8, "action_dim": 23}
        ),
        eval_env_list=[env],
        eval_prev_done=[torch.zeros(16, dtype=torch.bool)],
        use_external_reward_model=False,
    )
    _, metrics = EnvWorker.env_evaluate_step(worker, torch.zeros(16, 8, 23), 0)
    result = compute_evaluate_metrics([metrics])
    assert result["num_trajectories"] == 16
    assert result["success_once"] == pytest.approx(0.5)
    assert result["episode_len"] == pytest.approx(4.5)
    # Finished non-resetting environments may report done again on later chunks.
    _, repeated = EnvWorker.env_evaluate_step(worker, torch.zeros(16, 8, 23), 0)
    if auto_reset:
        assert compute_evaluate_metrics([repeated])["num_trajectories"] == 16
    else:
        assert repeated == {}
