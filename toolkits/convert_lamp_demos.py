# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Convert recorded primitive demos to RLinf's native macro replay format."""

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.data.datasets.lamp.realworld import RealWorldTrajectorySource
from rlinf.data.datasets.lamp.realworld_replay import convert_demo_trajectories
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models import get_model
from rlinf.models.embodiment.lamp.artifact_io import resolve_artifact_dir


def convert(
    source, policy, output: Path, *, base_path: Path, episodes, history_length: int
) -> dict:
    """Persist native trajectories and a strict source/model conversion contract."""
    base_path = resolve_artifact_dir(base_path)
    output.mkdir(parents=True, exist_ok=False)
    buffer = TrajectoryReplayBuffer(
        auto_save=False, sample_window_size=len(episodes), cache_size=len(episodes)
    )
    count = 0
    try:
        for trajectory in convert_demo_trajectories(
            source, policy, history_length=history_length, episodes=episodes
        ):
            buffer.add_trajectories([trajectory])
            count += len(trajectory.actions)
        if count == 0:
            raise ValueError("No demo episodes selected")
        buffer.save_checkpoint(str(output))
    finally:
        buffer.close()
    manifest = {
        "conversion_version": 1,
        "source_sha256": source.metadata.data_sha256,
        "robot_spec": source.spec.to_dict(),
        "episodes": sorted(episodes),
        "action_horizon": policy.horizon,
        "execution_horizon": policy.execution_horizon,
        "history_length": history_length,
        "model_contract": policy.contract_digest.cpu().tolist(),
        "base_artifact": hashlib.sha256(
            (resolve_artifact_dir(base_path) / "artifact.json").read_bytes()
        ).hexdigest(),
        "macro_transitions": count,
    }
    (output / "lamp_demo_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Resolved residual model YAML, including base artifact and spec",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        required=True,
        help="Explicit training episode IDs; never include validation episodes",
    )
    parser.add_argument("--history-length", type=int, default=16)
    args = parser.parse_args()
    config = OmegaConf.load(args.model_config)
    policy = get_model(config)
    convert(
        RealWorldTrajectorySource(args.source),
        policy,
        args.output,
        base_path=Path(config.model_path),
        episodes=args.episodes,
        history_length=args.history_length,
    )


if __name__ == "__main__":
    main()
