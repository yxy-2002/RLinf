# Copyright 2026 The RLinf Authors.
"""Protect the selected policy's training schedule and prior-to-DP contract."""

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_default_training_chain() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        prior = compose(config_name="dexjoco_lamp_prior_lamplstm_water_plant")
        dp = compose(config_name="dexjoco_lamp_dp_lamplstm")
        override = compose(
            config_name="dexjoco_lamp_dp_lamplstm",
            overrides=["actor.model.hand_prior.artifact_path=/tmp/custom/artifact"],
        )
    for cfg in (prior, dp):
        assert cfg.runner.save_interval == 10000
        assert cfg.actor.seed == 42
        assert cfg.actor.global_batch_size == cfg.actor.micro_batch_size == 512
        assert (
            cfg.actor.model.execution_horizon == cfg.actor.model.num_action_chunks == 8
        )
        assert cfg.data.task_name == "water_plant"
        assert cfg.data.history_contract == "primitive_v1"
        assert cfg.data.history_length == 8
        assert cfg.data.train_ratio == 0.9
        assert cfg.data.split_seed == 42
        assert cfg.actor.optim.clip_grad == 1.0
        pc = cfg.actor.model.hand_prior
        assert pc.type == "lamplstm"
        assert pc.encoder_condition_mode == pc.decoder_condition_mode == "film"
        assert pc.history_length == 8 and pc.horizon == 16 and pc.latent_dim == 2
        assert pc.action_hidden_dim == pc.condition_hidden_dim == 256
        assert pc.num_lstm_layers == 1
        assert pc.beta == 5e-4 and pc.beta_warmup_steps == 0
        assert pc.condition_drop_prob == 0.1
    assert prior.runner.max_steps == 20000
    assert dp.runner.max_steps == 40000
    assert prior.actor.optim.lr == 5e-5
    assert prior.actor.optim.warmup_steps == 500
    assert prior.actor.optim.weight_decay == 0.0
    assert dp.actor.optim.lr == 6e-5
    assert dp.actor.optim.warmup_steps == 1000
    assert dp.actor.optim.weight_decay == 1e-4
    assert prior.algorithm.stage == "prior" and dp.algorithm.stage == "dp"
    expected = (
        Path(prior.runner.logger.log_path)
        / prior.runner.logger.experiment_name
        / "artifact"
    )
    assert Path(dp.actor.model.hand_prior.artifact_path) == expected
    assert override.actor.model.hand_prior.artifact_path == "/tmp/custom/artifact"
