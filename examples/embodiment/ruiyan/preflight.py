# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate demo/action metadata and record the selected reward checkpoint."""

import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from examples.reward.ruiyan_demo_protocol import RewardClient


def main():
    root = Path(__file__).resolve().parents[3]
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
    output = Path(sys.argv[1])
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "examples/embodiment/config")
    ):
        cfg = compose(config_name="realworld_ruiyan_rlpd_local", overrides=sys.argv[2:])
    data = Path(cfg.algorithm.demo_buffer.load_path)
    convention = json.loads((data / "action_convention.json").read_text())
    if convention["arm_scale"] != cfg.env.train.ruiyan_rlpd.arm_scale or convention[
        "image_keys"
    ] != ["global", "wrist_1"]:
        raise ValueError(
            "Demo action convention / camera order differs from environment"
        )
    if convention["hand"] != "2*absolute_target-1":
        raise ValueError("Unrecognized hand action convention")
    if (
        not (data / "metadata.json").is_file()
        or not (data / "trajectory_index.json").is_file()
    ):
        raise ValueError("Incomplete native demo buffer")
    # Called on GPU host; both ends of the reverse tunnel use localhost:8770.
    identity = RewardClient(cfg.env.train.ruiyan_rlpd.reward_url, 3).request("/health")
    if identity["image_keys"] != ["global", "wrist_1"]:
        raise ValueError("Reward server cameras mismatch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward_model.json").write_text(json.dumps(identity, indent=2))
    (output / "action_convention.json").write_text(json.dumps(convention, indent=2))
    OmegaConf.save(cfg, output / "requested_config.yaml", resolve=True)
    print("Preflight passed. Reward checkpoint:", identity["checkpoint"])


if __name__ == "__main__":
    main()
