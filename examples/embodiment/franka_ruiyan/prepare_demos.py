# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Convert trusted Ruiyan absolute-hand demos to the RLPD policy convention."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.utils.ruiyan_rlpd import decode_action, encode_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-scale", type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; preserve original demos")
    paths = sorted(args.source.glob("trajectory_*.pt"))
    if not paths:
        raise ValueError("No native trajectory files found")
    converted = []
    for path in paths:
        d = torch.load(path, map_location="cpu", weights_only=False)
        a = d["actions"]
        if a.ndim != 3 or a.shape[1:] != (1, 12):
            raise ValueError("Expected T,1,12 actions")
        for key in ["main_images", "extra_view_images", "states"]:
            if not torch.equal(d["next_obs"][key][:-1], d["curr_obs"][key][1:]):
                raise ValueError(f"Broken observation continuity: {path} {key}")
        if d["curr_obs"]["states"].shape[-1] != 24:
            raise ValueError("Expected 24D state")
        if not bool(d["terminations"][-1].all()) or d["rewards"][-1].item() != 1:
            raise ValueError("Expected successful terminated demo")
        normalized = encode_action(a.numpy(), args.arm_scale)
        np.testing.assert_allclose(
            decode_action(normalized, args.arm_scale), a.numpy(), atol=1e-6
        )
        d["actions"] = torch.from_numpy(normalized)
        d["forward_inputs"]["action"] = d["actions"].clone()
        converted.append(Trajectory(**d))
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=False,
        auto_save=True,
        auto_save_path=str(args.output),
        trajectory_format="pt",
    )
    try:
        buffer.add_trajectories(converted)
    finally:
        buffer.close()
    (args.output / "action_convention.json").write_text(
        json.dumps(
            {
                "arm_scale": args.arm_scale,
                "hand": "2*absolute_target-1",
                "image_keys": ["global", "wrist_1"],
                "source": str(args.source.resolve()),
                "trajectories": len(converted),
                "steps": sum(t.actions.shape[0] for t in converted),
            },
            indent=2,
        )
    )
    print("Converted", len(converted), "trajectories to", args.output)


if __name__ == "__main__":
    main()
