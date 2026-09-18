# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline replay/model/gradient smoke test. Never imports a robot environment."""

import argparse
import json
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNConfig, CNNPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--demo-path", default="datasets/ruiyan_rlpd/demo_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "examples/embodiment/config")
    ):
        cfg = compose(config_name="realworld_ruiyan_rlpd_local")
    # validate_cfg starts Ray: deliberately use Hydra and the actual model offline.
    OmegaConf.resolve(cfg)
    meta = json.loads((Path(args.demo_path) / "action_convention.json").read_text())
    assert meta["arm_scale"] == cfg.env.train.ruiyan_rlpd.arm_scale
    assert meta["image_keys"] == ["global", "wrist_1"]
    buffer = TrajectoryReplayBuffer(seed=1, enable_cache=True, cache_size=2)
    try:
        buffer.load_checkpoint(args.demo_path)
        batch = buffer.sample(4)
        print(
            "Replay shapes:",
            {
                k: tuple(v.shape)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            },
        )
        obs = {
            k: v.to(args.device)
            for k, v in batch["curr_obs"].items()
            if isinstance(v, torch.Tensor)
        }
        actions = batch["actions"].to(args.device).reshape(4, 12)
        model_cfg = CNNConfig()
        model_cfg.update_from_dict(OmegaConf.to_container(cfg.actor.model))
        model = CNNPolicy(model_cfg).to(args.device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        pi, log_pi, _ = model(forward_type=ForwardType.SAC, obs=obs)
        q_demo = model(forward_type=ForwardType.SAC_Q, obs=obs, actions=actions)
        q_pi = model(forward_type=ForwardType.SAC_Q, obs=obs, actions=pi)
        assert pi.shape == (4, 12) and q_demo.shape == (4, 10)
        loss = (
            q_demo.square().mean()
            + (0.01 * log_pi - q_pi.mean(-1, keepdim=True)).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step()
        print(
            f"PASS: dual-view 24D state / 12D action / 10 critics; finite gradient update, loss={loss.item():.6f}"
        )
    finally:
        buffer.close()


if __name__ == "__main__":
    main()
