# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collect explicitly delimited, hand-only episodes without a Franka robot."""

import argparse
import select
import sys
import time
from pathlib import Path

from rlinf_dexhand.pipeline import load_config

from rlinf.envs.dexhand.hand_env import HandCollectionEnv


def main(backend_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--backend-factory", help="RLinf 执行后端工厂，格式 module:function"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay", help="Recorded ADC JSONL instead of hardware")
    parser.add_argument(
        "--repeat", action="store_true", help="Loop replay for soak tests"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        help="Explicit timed episode, otherwise Enter starts/ends",
    )
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    cfg = load_config(args.config)
    if cfg["hand"]["type"] != "wuji1hand":
        parser.error("Use existing Franka collector for Ruiyan 12-D data")
    if cfg.get("frequency", 30) <= 0:
        parser.error("frequency must be positive")
    from rlinf.envs.dexhand.recording import EpisodeWriter

    pipeline = None
    if args.replay:
        from rlinf_dexhand.pipeline import TeleopPipeline

        from rlinf.envs.dexhand.replay import ReplayGlove

        cfg["replay_file"] = str(Path(args.replay).resolve())
        pipeline = TeleopPipeline(
            cfg,
            ReplayGlove(
                args.replay, cfg["glove"]["type"], cfg["glove"]["side"], args.repeat
            ),
        )
    if backend_factory is None and args.backend_factory:
        import importlib

        module, name = args.backend_factory.rsplit(":", 1)
        backend_factory = getattr(importlib.import_module(module), name)
    if backend_factory is None:
        parser.error(
            "Supply an execution backend from the RLinf caller; for RViz use python -m toolkits.dexhand.test_retargeting"
        )
    from rlinf_dexhand.pipeline import TeleopPipeline

    pipeline = pipeline or TeleopPipeline(cfg)
    env = HandCollectionEnv(
        cfg, pipeline=pipeline, backend=backend_factory(cfg, pipeline.spec)
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            if args.seconds is None:
                if input("Enter: start episode; q: exit > ").strip().lower() == "q":
                    break
            env.reset()
            episode = EpisodeWriter(output, cfg, env.spec)
            start = time.monotonic()
            try:
                with episode:
                    while (
                        args.seconds is None or time.monotonic() - start < args.seconds
                    ):
                        tick = time.monotonic()
                        if (
                            args.seconds is None
                            and select.select([sys.stdin], [], [], 0)[0]
                        ):
                            sys.stdin.readline()
                            break
                        target = env.read_expert_target()
                        _, _, _, _, info = env.step(target.values)
                        episode.append(info)
                        time.sleep(
                            max(
                                0,
                                1 / cfg.get("frequency", 30)
                                - (time.monotonic() - tick),
                            )
                        )
                print(f"Saved {episode.path}")
            finally:
                env.close()
            if args.seconds is not None:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
