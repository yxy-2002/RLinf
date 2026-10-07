# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check hardware-free Wuji evaluation configuration and chunk termination."""

from pathlib import Path
from unittest.mock import MagicMock

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]


def eval_config(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(ROOT / "evaluations/realworld"), version_base="1.1"
    ):
        return compose(config_name="realworld_lamp_dp_stack_cube_eval")


def test_eval_config_uses_normalized_wuji_adapter(monkeypatch):
    from rlinf.envs.lamp_adapter import validate_lamp_environment

    cfg = eval_config(monkeypatch)
    validate_lamp_environment(cfg.env.eval, cfg.rollout.model)
    assert cfg.runner.only_eval
    assert cfg.env.eval.lamp_history_length == 8
    assert cfg.rollout.model.num_action_chunks == 8
    assert cfg.rollout.model.execution_horizon_override == 8
    assert cfg.rollout.model.action_dim == 26
    assert cfg.rollout.model.robot_spec.arm_state_dim == 6
    assert cfg.env.eval.glove_config.release_behavior == "policy"
    assert not cfg.env.eval.auto_reset
    assert cfg.cluster.component_placement.env.node_group == "franka"
    assert cfg.cluster.component_placement.rollout.node_group == "reward_gpu"
    assert not cfg.reward.use_reward_model
    assert not cfg.reward.standalone_realworld
    assert cfg.reward.model.model_path is None
    assert not cfg.env.eval.override_cfg.use_reward_model
    assert not cfg.env.eval.override_cfg.reward_success_confirmation
    assert not cfg.env.eval.override_cfg.enable_pose_reward


def test_dp_eval_counts_terminal_prefix_before_cancelled_suffix(monkeypatch):
    from types import SimpleNamespace

    import torch

    import rlinf.workers.env.env_worker as module

    cfg = eval_config(monkeypatch)
    obs = {"main_images": torch.zeros(1, 2, 2, 3)}
    env = MagicMock()
    env.chunk_step.return_value = (
        [obs] * 8,
        torch.zeros(1, 8),
        torch.tensor([[True, False, False, False, False, False, False, False]]),
        torch.zeros(1, 8, dtype=torch.bool),
        [{}] * 7 + [{"episode": {"success": torch.tensor([True])}}],
    )
    worker = SimpleNamespace(
        cfg=cfg,
        model_cfg=cfg.rollout.model,
        eval_env_list=[env],
        use_external_reward_model=False,
        eval_prev_done=[torch.tensor([False])],
    )
    monkeypatch.setattr(
        module, "prepare_actions", lambda **kwargs: kwargs["raw_chunk_actions"]
    )
    _, metrics = module.EnvWorker.env_evaluate_step(worker, torch.zeros(1, 8, 26), 0)
    assert metrics["success"].item()
    _, metrics = module.EnvWorker.env_evaluate_step(worker, torch.zeros(1, 8, 26), 0)
    assert metrics == {}
