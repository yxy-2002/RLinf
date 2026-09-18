# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Teleoperated successful trajectories gated by a dual-view reward model."""

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from examples.reward.ruiyan_demo_protocol import (
    RewardClient,
    SuccessGate,
    executed_action,
)
from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.data.reward_views import select_reward_views
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


def tensor_obs(obs):
    return {
        key: torch.as_tensor(value).detach().cpu().clone()
        for key, value in obs.items()
        if key != "task_descriptions"
    }


class DemoCollector(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        cfg = self.cfg
        root = Path(cfg.runner.logger.log_path)
        root.mkdir(parents=True, exist_ok=True)
        if (root / "manifest.jsonl").exists():
            raise ValueError(
                "Use a fresh output directory; demo resume is not supported"
            )
        client = RewardClient(cfg.runner.reward_url, cfg.runner.rpc_timeout)
        identity = client.request("/health")
        if identity["image_keys"] != ["global", "wrist_1"]:
            raise ValueError("Reward camera order mismatch")
        (root / "reward_model.json").write_text(json.dumps(identity, indent=2))
        OmegaConf.save(cfg, root / "collection_config.yaml")
        env = None
        buffer = TrajectoryReplayBuffer(
            seed=1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=str(root / "demos"),
            trajectory_format="pt",
        )
        successes = episodes = 0
        rollout = None
        probabilities = []
        phase = "waiting"
        try:
            while successes < cfg.runner.num_data_episodes:
                command = client.status(
                    phase, successes=successes, episodes=episodes
                ).get("command")
                if command == "quit":
                    break
                if phase == "waiting":
                    if command != "start":
                        time.sleep(0.1)
                        continue
                    if env is None:
                        env = RealWorldEnv(
                            cfg.env.eval,
                            num_envs=1,
                            seed_offset=0,
                            total_num_processes=1,
                            worker_info=self.worker_info,
                        )
                    # Reset is explicit and outside the recorded trajectory.
                    client.status("resetting")
                    obs, _ = env.reset()
                    if env.action_space.shape[-1] != 12:
                        raise ValueError("Ruiyan demo requires 12D actions")
                    rollout = EmbodiedRolloutResult(
                        max_episode_length=cfg.env.eval.max_episode_steps
                    )
                    probabilities = []
                    gate = SuccessGate(
                        cfg.runner.success_threshold, cfg.runner.success_hold_steps
                    )
                    phase = "recording"
                    continue
                if phase == "candidate":
                    if command not in ("accept", "discard"):
                        time.sleep(0.1)
                        continue
                    save = command == "accept"
                    reason = (
                        "model_success_accepted" if save else "model_success_rejected"
                    )
                else:
                    if command == "discard":
                        save, reason = False, "manual_abort"
                    else:
                        next_obs, _, _, truncated, info = env.step(
                            np.zeros((1, 12), dtype=np.float32)
                        )
                        paired = select_reward_views(
                            next_obs, ["global", "wrist_1"], "global", ["wrist_1"]
                        )
                        probability = client.predict(paired)
                        probabilities.append(probability)
                        # Determine success from post-action images, not the positional reward.
                        success = gate.update(probability)
                        if len(probabilities) < cfg.runner.min_episode_steps:
                            gate.count = 0
                            success = False
                        timeout = (
                            bool(torch.as_tensor(truncated).any())
                            or len(probabilities) >= cfg.env.eval.max_episode_steps
                        )
                        action = executed_action(info)
                        terminal = torch.tensor([[success]], dtype=torch.bool)
                        truncation = torch.tensor(
                            [[timeout and not success]], dtype=torch.bool
                        )
                        rollout.append_step_result(
                            ChunkStepResult(
                                actions=action,
                                rewards=torch.tensor([[float(success)]]),
                                dones=terminal | truncation,
                                terminations=terminal,
                                truncations=truncation,
                                forward_inputs={"action": action},
                            )
                        )
                        rollout.append_transitions(
                            curr_obs=tensor_obs(obs), next_obs=tensor_obs(next_obs)
                        )
                        obs = next_obs
                        if len(probabilities) % 10 == 0:
                            self.log_info(
                                f"Demo step={len(probabilities)} probability={probability:.4f} hold={gate.count}"
                            )
                        if success and cfg.runner.require_confirmation:
                            phase = "candidate"
                            continue
                        if not success and not timeout:
                            continue
                        save, reason = (
                            success,
                            "model_success" if success else "timeout",
                        )
                episodes += 1
                length = len(probabilities)
                if rollout is not None and length:
                    trajectory = rollout.to_trajectory()
                    trajectory.intervene_flags = torch.ones_like(
                        trajectory.intervene_flags
                    )
                    if save:
                        buffer.add_trajectories([trajectory])
                        successes += 1
                    else:
                        # Store rejected or aborted trajectories separately from the success buffer.
                        (root / "rejected").mkdir(exist_ok=True)
                        torch.save(
                            trajectory, root / "rejected" / f"episode_{episodes:04d}.pt"
                        )
                with (root / "manifest.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "episode": episodes,
                                "saved": save,
                                "reason": reason,
                                "steps": length,
                                "probabilities": probabilities,
                            }
                        )
                        + "\n"
                    )
                self.log_info(
                    f"Episode {episodes}: {reason}, steps={length}; saved={successes}/{cfg.runner.num_data_episodes}"
                )
                rollout = None
                phase = "waiting"
        finally:
            # Quit/failure never promotes a partial trajectory to a successful demo.
            try:
                buffer.close()
            finally:
                if env is not None:
                    try:
                        for subenv in getattr(env.env, "envs", []):
                            controller = getattr(subenv.unwrapped, "_controller", None)
                            if controller is not None:
                                controller.shutdown_demo_control().wait()
                    finally:
                        env.env.close()  # Close cameras and the glove reader.
            try:
                client.status("finished", successes=successes)
            except Exception:
                pass
        self.log_info(
            f"Finished: {successes} successful demos in {root / 'demos'}; no final reset"
        )


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_collect_ruiyan_demos",
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster).get_strategy("env")
    collector = DemoCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
