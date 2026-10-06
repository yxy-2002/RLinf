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

"""Offline integration: native realenv replay/mixing, SAC update and checkpoints."""

import copy
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch.distributed.device_mesh import init_device_mesh

from rlinf.data.datasets.lamp.realworld import (
    RealWorldTrajectorySource,
    wuji_robot_spec,
)
from rlinf.data.datasets.lamp.realworld_replay import convert_demo_trajectories
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.models.embodiment.lamp import get_residual_model
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType
from rlinf.workers.actor.async_fsdp_lamp_residual_sac_policy_worker import (
    AsyncLampResidualSACFSDPPolicy,
)
from toolkits.convert_lamp_demos import convert


def main() -> None:
    """Run the production path with explicit, local acceptance artifacts."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_RANK"] = "0"
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{root / 'rlpd_rendezvous'}", rank=0, world_size=1
    )
    mesh = init_device_mesh("cuda", (1,))
    audit = json.loads((root / "data_audit.json").read_text())
    source = RealWorldTrajectorySource(audit.get("source", root / "raw/demos"))
    demo_ids = audit["train_episodes"][:9]
    online_ids = audit["train_episodes"][9:]
    assert not set(demo_ids) & set(online_ids)
    results = {}
    try:
        for prior in ("lamplstm", "vq", "pca", "mlp"):
            torch.manual_seed(43)
            out = root / "rlpd" / prior
            out.mkdir(parents=True, exist_ok=True)
            with initialize_config_dir(
                config_dir=str(Path("examples/embodiment/config").resolve()),
                version_base="1.1",
            ):
                cfg = compose(config_name="dexjoco_lamp_residual_sac")
            with open_dict(cfg):
                cfg.actor.global_batch_size = 4
                cfg.actor.micro_batch_size = 4
                cfg.actor.model.model_path = str(
                    root / "training" / prior / "dp/artifact"
                )
                cfg.actor.model.robot_spec = wuji_robot_spec().to_dict()
                cfg.actor.model.contract_version = 5
                cfg.actor.model.action_dim = 26
                cfg.actor.model.eval_base_noise_seed = 42
                cfg.actor.model.num_inference_steps_override = 2
                cfg.actor.model.learning_starts_macro_transitions = 0
                cfg.runner.logger.log_path = str(out)
                cfg.algorithm.actor_agg_q = "mean"
                cfg.algorithm.backup_entropy = False
                cfg.algorithm.critic_actor_ratio = 4
                cfg.algorithm.utd_ratio = 4.0
                cfg.algorithm.learning_starts_macro_transitions = 0
                cfg.algorithm.demo_fraction = 0.5
                cfg.algorithm.replay_buffer.auto_save = False
                cfg.algorithm.replay_buffer.enable_preload = False
                cfg.algorithm.replay_buffer.cache_size = 20
                cfg.algorithm.replay_buffer.sample_window_size = 20
                cfg.algorithm.demo_buffer = {
                    "enable_cache": True,
                    "cache_size": 30,
                    "sample_window_size": 30,
                    "min_buffer_size": 1,
                    "load_path": str(out / "demos"),
                    "auto_save": False,
                }
            policy = get_residual_model(cfg.actor.model).cuda()
            if not (out / "demos").exists():
                convert(
                    source,
                    policy,
                    out / "demos",
                    base_path=Path(cfg.actor.model.model_path),
                    episodes=demo_ids,
                    history_length=16,
                )
            w = object.__new__(AsyncLampResidualSACFSDPPolicy)
            w.cfg = cfg
            w._rank = 0
            w._world_size = 1
            w.device = torch.device("cuda:0")
            w.torch_dtype = torch.float32
            w._timer_metrics = {}
            w._accelerator_type = AcceleratorType.NV_GPU
            w.gradient_accumulation = 1
            w.enable_drq = False
            w.update_step = 0
            w.demo_buffer = None
            w.is_weight_offloaded = False
            w.is_optimizer_offloaded = False
            strategy = FSDPStrategy(cfg.actor, world_size=1)
            w._strategy = strategy
            target = copy.deepcopy(policy)
            frozen = {
                k: v.detach().cpu().clone()
                for k, v in policy.base_policy.state_dict().items()
            }
            w.model = strategy.wrap_model(policy, mesh)
            w.target_model = strategy.wrap_model(target, mesh)
            w.target_model_initialized = True
            w.target_update_type = "all"
            w.optimizer = torch.optim.Adam(policy.residual_actor.parameters(), lr=1e-4)
            w.qf_optimizer = torch.optim.Adam(policy.q_head.parameters(), lr=1e-4)
            w.entropy_temp = EntropyTemperature(0.01, "exp", device=w.device)
            w.target_entropy = policy.target_entropy
            w.alpha_optimizer = torch.optim.Adam(w.entropy_temp.parameters(), lr=1e-4)
            w.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(w.optimizer, lambda _: 1)
            w.qf_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                w.qf_optimizer, lambda _: 1
            )
            w.alpha_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                w.alpha_optimizer, lambda _: 1
            )
            w.setup_sac_components()
            w._ensure_lamp_progress()
            for traj in convert_demo_trajectories(
                source, policy, episodes=online_ids, history_length=16
            ):
                # The original demos label every step as human. Mark this separate test
                # input as simulated policy experience to test the exact initial mix.
                traj.intervene_flags.zero_()
                w._recv_queue.put(traj)
                w._drain_received_trajectories()
            actor_updates = 0
            w.model.train()
            w.target_model.eval()
            for step in range(64):
                metrics = w.update_one_epoch()
                actor_updates += int("sac/actor_loss" in metrics)
                assert all(
                    torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values()
                )
                w.update_step += 1
                w._optimizer_updates_completed += 1
                assert all(p.grad is None for p in policy.base_policy.parameters())
                if step % 16 == 15:
                    print(
                        prior,
                        step + 1,
                        "critic",
                        metrics["sac/critic_loss"],
                        "actor_updates",
                        actor_updates,
                        flush=True,
                    )
            assert actor_updates == 16
            for key, value in policy.base_policy.state_dict().items():
                torch.testing.assert_close(value.cpu(), frozen[key], rtol=0, atol=0)
            saved = out / "checkpoint"
            w.save_checkpoint(str(saved), 64)
            original_alpha = w.entropy_temp.base_alpha.detach().clone()
            demo_count = w.demo_buffer.get_stats()["total_samples"]
            next_online = w.replay_buffer.sample(4)["actions"].clone()
            next_demo = w.demo_buffer.sample(4)["actions"].clone()
            with torch.no_grad():
                w.entropy_temp.base_alpha.add_(1)
            w.update_step = 0
            w.load_checkpoint(str(saved))
            torch.testing.assert_close(
                w.entropy_temp.base_alpha, original_alpha, rtol=0, atol=0
            )
            assert (
                w.update_step == 64
                and w.demo_buffer.get_stats()["total_samples"] == demo_count
            )
            torch.testing.assert_close(
                w.replay_buffer.sample(4)["actions"], next_online, rtol=0, atol=0
            )
            torch.testing.assert_close(
                w.demo_buffer.sample(4)["actions"], next_demo, rtol=0, atol=0
            )
            gamma = cfg.algorithm.gamma
            cfg.algorithm.gamma = 0.5
            try:
                w.load_checkpoint(str(saved))
            except ValueError as error:
                assert "training contract differs" in str(error)
            else:
                raise AssertionError("Changed training contracts must reject resume")
            finally:
                cfg.algorithm.gamma = gamma
            results[prior] = {
                "critic_updates": 64,
                "actor_updates": actor_updates,
                "alpha_updates": actor_updates,
                "progress": w.get_lamp_progress(),
                "demo_episodes": demo_ids,
                "simulated_online_episodes": online_ids,
                "ddim_steps": 2,
                "checkpoint": str(saved),
                "sampling_rng_restored": True,
                "changed_contract_rejected": True,
            }
            (root / "rlpd_results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            w.replay_buffer.close()
            w.demo_buffer.close()
            del w, policy, target, strategy, frozen
            gc.collect()
            torch.cuda.empty_cache()
        print("ALL OFFLINE RLPD PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
