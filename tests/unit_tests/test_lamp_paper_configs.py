# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Protect the paper schedules, task inheritance, and prior-to-DP bindings."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
METHODS = ("lstm", "lstm_none", "vq", "pca", "mlp")
CASES = [(stage, method) for stage in ("prior", "dp") for method in METHODS]
CASES.remove(("prior", "mlp"))


def config_name(stage: str, method: str, task: str) -> str:
    return f"dexjoco_lamp_paper/{stage}/dexjoco_lamp_paper_{method}_{stage}_{task}"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("stage,method", CASES)
def test_paper_training_contract(stage: str, method: str, task: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.1"):
        cfg = compose(config_name=config_name(stage, method, task))
        assert not OmegaConf.missing_keys(cfg)
        assert "dexjoco_lamp_paper" not in cfg  # Nested files populate the root.
        assert cfg.data.task_name == task
        assert cfg.data.history_contract == "primitive_v1"
        assert cfg.data.num_workers == 0 and not cfg.data.persistent_workers
        assert cfg.actor.seed == 42 and not cfg.actor.torch_compile
        assert cfg.actor.model.model_type == "lamp_dp"
        assert cfg.algorithm.stage == stage
        expected_eval_batch = 256 if (stage, method) == ("prior", "pca") else 512
        assert cfg.actor.eval_batch_size == expected_eval_batch
        assert cfg.runner.resume_dir is None

        if stage == "dp":
            assert cfg.runner.max_steps == 40000
            assert cfg.actor.global_batch_size == cfg.actor.micro_batch_size == 512
            assert cfg.actor.optim.lr == 6e-5
            assert cfg.actor.optim.min_lr == 1e-6
            assert cfg.actor.optim.warmup_steps == 1000
            assert cfg.actor.optim.weight_decay == 1e-4
            assert cfg.actor.optim.backbone_lr_ratio == 0.1
            assert cfg.actor.validation_interval == 1000
            assert cfg.actor.model.execution_horizon == 8
            assert cfg.actor.model.num_action_chunks == 8
            assert cfg.runner.save_interval == 10000
            if method == "mlp":
                assert cfg.actor.model.hand_prior.artifact_path is None
            else:
                prior = compose(config_name=config_name("prior", method, task))
                expected = (
                    Path(prior.runner.logger.log_path)
                    / prior.runner.logger.experiment_name
                    / "artifact"
                )
                assert Path(cfg.actor.model.hand_prior.artifact_path) == expected
        elif method.startswith("lstm"):
            assert cfg.runner.max_steps == 20000
            assert cfg.actor.global_batch_size == cfg.actor.micro_batch_size == 512
            assert cfg.actor.optim.lr == 5e-5
            assert cfg.actor.optim.warmup_steps == 500
            assert cfg.actor.optim.weight_decay == 0.0
        elif method == "vq":
            assert cfg.runner.max_steps == 30000
            assert cfg.runner.save_every_epochs == 10
            assert cfg.actor.global_batch_size == cfg.actor.micro_batch_size == 256
            assert cfg.actor.optim.lr == 3e-4
            assert cfg.actor.optim.min_lr == 0.0
            assert cfg.actor.optim.warmup_steps == 150
            assert cfg.actor.optim.weight_decay == 1e-6
            assert cfg.actor.optim.adam_beta1 == 0.95
            assert cfg.actor.optim.clip_grad is None
            assert cfg.actor.validation_interval == -1 and cfg.actor.validate_at_end
        else:
            assert cfg.runner.max_steps == cfg.runner.local_update_steps == 1
            assert cfg.runner.save_interval == 1

        hp = cfg.actor.model.hand_prior
        if method.startswith("lstm"):
            assert hp.type == "lamplstm"
            assert hp.latent_dim == 2 and hp.history_length == 8
            if stage == "prior":
                mode = {"lstm": "film", "lstm_none": "none"}[method]
                assert hp.encoder_condition_mode == hp.decoder_condition_mode == mode
                assert hp.horizon == 16
                assert hp.beta == 5e-4 and hp.beta_warmup_steps == 0
                assert hp.condition_drop_prob == 0.1
        else:
            assert hp.type == method

        assert "bc_loss" not in cfg.algorithm
        assert not {
            "right",
            "left",
            "posterior_kl_weight",
            "prior_kl_weight",
            "kl_warmup_steps",
        }.intersection(hp)
        assert not {
            "source",
            "horizon",
            "action_dim",
            "history_length",
            "train_ratio",
            "split_seed",
        }.intersection(cfg.data)
        if stage == "dp":
            assert set(hp) <= {
                "type",
                "hand_side",
                "artifact_path",
                "latent_dim",
                "history_length",
            }
        elif method in ("pca", "vq"):
            assert "beta" not in hp and "beta_warmup_steps" not in hp
        if method != "vq":
            assert "hidden_dim" not in hp


def test_task_files_only_override_task_identity_and_artifact() -> None:
    for stage, method in CASES:
        for task in TASKS:
            raw = OmegaConf.load(
                CONFIG_ROOT / f"{config_name(stage, method, task)}.yaml"
            )
            expected = {"defaults", "runner", "data"}
            if stage == "dp" and method != "mlp":
                expected.add("actor")
                assert set(raw.actor) == {"model"}
                assert set(raw.actor.model) == {"hand_prior"}
                assert set(raw.actor.model.hand_prior) == {"artifact_path"}
            assert set(raw) == expected
            assert set(raw.runner) == {"logger"}
            assert set(raw.runner.logger) == {"experiment_name"}
            assert set(raw.data) == {"task_name"}
            assert list(raw.defaults) == [
                f"/dexjoco_lamp_paper/{stage}/dexjoco_lamp_paper_{method}_{stage}",
                "_self_",
            ]


def test_cli_overrides_reach_nested_task_config() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.1"):
        cfg = compose(
            config_name=config_name("dp", "lstm", "water_plant"),
            overrides=[
                "actor.optim.lr=0.0002",
                "actor.model.hand_prior.artifact_path=/tmp/prior/artifact",
                "cluster.component_placement.actor=1-1",
            ],
        )
    assert cfg.actor.optim.lr == 2e-4
    assert cfg.actor.model.hand_prior.artifact_path == "/tmp/prior/artifact"
    assert cfg.cluster.component_placement.actor == "1-1"
