# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline DP image augmentation without training assets or hardware."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils.drq import crop_bchw_fast
from rlinf.workers.actor.lamp_il_worker import LampILWorker


@pytest.mark.parametrize("enabled", [False, True])
def test_dp_crop_matches_drq_and_preserves_other_inputs(enabled):
    worker = SimpleNamespace(
        device="cpu",
        stage="dp",
        cfg=OmegaConf.create({"actor": {"enable_drq": enabled}}),
    )
    batch = {
        "front": torch.arange(4 * 12 * 12 * 3).reshape(4, 12, 12, 3).to(torch.uint8),
        "wrist": torch.arange(4 * 12 * 12 * 3)
        .reshape(4, 12, 12, 3)
        .flip(1)
        .to(torch.uint8),
        "target_action": torch.randn(4, 8, 26),
        "arm_state_pair_norm": torch.randn(4, 2, 6),
        "hand_state_pair_norm": torch.randn(4, 2, 20),
        "mask": torch.ones(4, 8),
    }
    originals = {key: value.clone() for key, value in batch.items()}
    torch.manual_seed(12)
    actual = LampILWorker._prepare_batch(worker, batch, train=True)
    torch.manual_seed(12)
    for key in ("front", "wrist"):
        normalized = batch[key].permute(0, 3, 1, 2).float() / 255
        expected = crop_bchw_fast(normalized, pad=4) if enabled else normalized
        torch.testing.assert_close(actual[key], expected)
        assert actual[key].shape == (4, 3, 12, 12)
        assert actual[key].is_contiguous()
        if enabled:
            assert not torch.equal(actual[key], normalized)
    for key in batch:
        torch.testing.assert_close(batch[key], originals[key])
        if key not in ("front", "wrist"):
            torch.testing.assert_close(actual[key], batch[key])


@pytest.mark.parametrize("stage,train", [("dp", False), ("prior", True)])
def test_validation_and_prior_skip_augmentation_and_preserve_rng(stage, train):
    worker = SimpleNamespace(
        device="cpu",
        stage=stage,
        cfg=OmegaConf.create({"actor": {"enable_drq": True}}),
    )
    batch = {"front": torch.full((2, 12, 12, 3), 100, dtype=torch.uint8)}
    state = torch.get_rng_state().clone()
    actual = LampILWorker._prepare_batch(worker, batch, train=train)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(actual["front"], torch.full((2, 3, 12, 12), 100 / 255))


def test_realworld_dp_config_defaults_off_and_allows_override(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(root))
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="realworld_lamp_dp_lamplstm_concat")
        assert cfg.actor.enable_drq is False
        cfg = compose(
            config_name="realworld_lamp_dp_lamplstm_concat",
            overrides=["actor.enable_drq=true"],
        )
        assert cfg.actor.enable_drq is True
