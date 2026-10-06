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

"""Native Ray worker/runner smoke with an injected, hardware-free environment."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from rlinf.config import validate_cfg
from rlinf.data.datasets.lamp.realworld import wuji_robot_spec
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.async_fsdp_lamp_residual_sac_policy_worker import (
    AsyncLampResidualSACFSDPPolicy,
)
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


class FakeWorld:
    num_envs = 1
    auto_reset = True

    def __init__(self, **kwargs):
        self.t = 0

    def obs(self):
        return {
            "states": torch.full((1, 38), self.t / 100),
            "main_images": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(1, 1, 128, 128, 3, dtype=torch.uint8),
        }

    def reset(self, **kwargs):
        self.t = 0
        return self.obs(), {}

    def step(self, actions, auto_reset=False):
        self.t += 1
        action = torch.as_tensor(actions).clone().clamp(-1, 1)
        action[:, 6:].clamp_(0, 1)
        human = self.t == 2
        if human:
            action.fill_(0.25)
        done = torch.tensor([self.t == 5])
        return (
            self.obs(),
            done.float(),
            done,
            torch.zeros_like(done),
            {"executed_action": action, "intervene_flag": torch.tensor([human])},
        )

    def close(self):
        pass

    def update_reset_state_ids(self):
        pass


class FakeEnvWorker(AsyncEnvWorker):
    def init_worker(self):
        import rlinf.workers.env.env_worker as module

        module.get_env_cls = lambda *args: FakeWorld
        super().init_worker()


def main() -> None:
    """Run the production path with explicit, local acceptance artifacts."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--actor-device", default="0")
    parser.add_argument("--rollout-device", default="1")
    parser.add_argument("--env-device", default="0")
    parser.add_argument("--lockstep", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(
        config_dir=str(Path("examples/embodiment/config").resolve()), version_base="1.1"
    ):
        cfg = compose(config_name="dexjoco_lamp_residual_sac_water_plant")
    with open_dict(cfg):
        cfg.cluster.component_placement = {
            "actor": args.actor_device,
            "rollout": args.rollout_device,
            "env": args.env_device,
        }
        cfg.runner.max_steps = args.max_steps
        cfg.runner.max_epochs = args.max_steps
        cfg.runner.val_check_interval = -1
        cfg.runner.save_interval = 1
        cfg.runner.resume_dir = (
            str(Path(args.resume).resolve()) if args.resume else None
        )
        cfg.runner.logger.log_path = str(root / "ray")
        cfg.runner.logger.logger_backends = ["tensorboard"]
        cfg.runner.logger.experiment_name = "hardware_free"
        cfg.actor.global_batch_size = 4
        cfg.actor.micro_batch_size = 4
        cfg.actor.model.contract_version = 5
        cfg.actor.model.action_dim = 26
        cfg.actor.model.robot_spec = wuji_robot_spec().to_dict()
        cfg.actor.model.model_path = str(root / "training/lamplstm/dp/artifact")
        cfg.actor.model.eval_base_noise_seed = 42
        cfg.actor.model.num_inference_steps_override = 2
        cfg.actor.model.learning_starts_macro_transitions = 0
        cfg.algorithm.learning_starts_macro_transitions = 0
        cfg.algorithm.utd_ratio = 4.0
        cfg.algorithm.critic_actor_ratio = 4
        cfg.algorithm.actor_agg_q = "mean"
        cfg.algorithm.backup_entropy = False
        cfg.algorithm.demo_fraction = 0.5
        cfg.algorithm["async"].lockstep_updates = args.lockstep
        cfg.algorithm.replay_buffer.auto_save = False
        cfg.algorithm.replay_buffer.enable_preload = False
        cfg.algorithm.replay_buffer.min_buffer_size = 1
        cfg.algorithm.demo_buffer = {
            "enable_cache": True,
            "cache_size": 100,
            "sample_window_size": 100,
            "min_buffer_size": 1,
            "auto_save": False,
            "load_path": str(root / "rlpd/lamplstm/demos"),
        }
        cfg.reward.use_reward_model = False
        cfg.env.train.lamp_history_length = 16
        cfg.env.train.env_type = "realworld"
        cfg.env.train.total_num_envs = 1
        cfg.env.train.max_steps_per_rollout_epoch = 8
        cfg.env.train.lamp_adapter = (
            "rlinf.envs.lamp_realworld_adapter:RealWorldLampAdapter"
        )
        cfg.env.train.lamp_robot_spec = wuji_robot_spec().to_dict()
        cfg.env.train.video_cfg.save_video = False
        cfg.env.train.auto_reset = True
        cfg.env.eval.total_num_envs = 1
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = AsyncLampResidualSACFSDPPolicy.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    rollout = AsyncMultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = FakeEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = AsyncEmbodiedRunner(cfg, actor, rollout, env, None)
    runner.init_workers()
    runner.run()
    print("RAY HARDWARE-FREE SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()
