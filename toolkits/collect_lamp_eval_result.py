#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persist a verified 50-seed DexJoCo LAMP evaluation result."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SEED_START = 20260803
SEED_END = 20260852
EPISODES = 50


def latest_scalar(log_dir: Path, tag: str) -> float:
    """Return the final scalar for ``tag`` from a TensorBoard event directory."""
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        raise RuntimeError(f"Missing TensorBoard scalar {tag!r} in {log_dir}")
    return float(accumulator.Scalars(tag)[-1].value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.eval_root / args.cfg
    tensorboard_dir = output_dir / "tensorboard"
    success_rate = latest_scalar(tensorboard_dir, "eval/success_once")
    trajectory_count = latest_scalar(tensorboard_dir, "eval/num_trajectories")
    if round(trajectory_count) != EPISODES:
        raise RuntimeError(
            f"Expected {EPISODES} evaluated trajectories, got {trajectory_count}"
        )

    mode = args.cfg.removeprefix("dexjoco_lamp_dp_il_").removesuffix(f"_{args.task}")
    row = {
        "config": args.cfg,
        "task": args.task,
        "mode": mode,
        "episodes": str(EPISODES),
        "seed_start": str(SEED_START),
        "seed_end": str(SEED_END),
        "success_rate": f"{success_rate:.6f}",
        "video_dir": str(output_dir / "video" / "eval"),
    }
    csv_path = args.eval_root / "results_50seed.csv"
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = [existing for existing in rows if existing.get("config") != args.cfg]
    rows.append(row)
    rows.sort(key=lambda item: item["config"])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = args.eval_root / "results_50seed.md"
    lines = [
        "# DexJoCo LAMP DP: 50-seed online evaluation",
        "",
        f"All rows use seeds {SEED_START}--{SEED_END} (one episode per seed).",
        "",
        "| Task | Mode | Success rate | Episodes | Video directory |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in rows:
        lines.append(
            f"| {item['task']} | {item['mode']} | {float(item['success_rate']):.2%} "
            f"| {item['episodes']} | `{item['video_dir']}` |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Recorded {args.cfg}: success_rate={success_rate:.6f}")


if __name__ == "__main__":
    main()
