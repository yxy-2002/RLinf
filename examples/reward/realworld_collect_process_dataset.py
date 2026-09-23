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

"""Collect keyboard-labeled frames or multi-episode SpaceMouse-labeled views.

The dexhand_reward_model config uses the right button for positive frames,
retains raw episodes, and splits complete episodes into train/validation sets.
The realworld_collect_dataset config retains the legacy keyboard workflow.
"""

import asyncio
import json
import os
import random
import signal
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker
from rlinf.utils.logging import get_logger

logger = get_logger()


class FrameCollector(Worker):
    """Collect labeled frames using the configured input and persistence mode."""

    def __init__(self, cfg):
        super().__init__()
        self._quit = False
        self.cfg = cfg
        self.target_success = cfg.runner.num_success_frames
        self.target_fail = cfg.runner.num_fail_frames
        self.val_split = cfg.runner.get("val_split", 0.2)
        self.fail_success_ratio = cfg.runner.get("fail_success_ratio", 2.0)
        self.random_seed = cfg.runner.get("random_seed", 42)

        self.success_frames: list[torch.Tensor] = []
        self.fail_frames: list[torch.Tensor] = []

        self.label_source = cfg.runner.get("label_source", "keyboard")
        if self.label_source not in ("keyboard", "spacemouse_right"):
            raise ValueError(f"Unknown label_source: {self.label_source}")
        if self.label_source == "spacemouse_right":
            env_cfg = cfg.env.eval
            if env_cfg.auto_reset or env_cfg.ignore_terminations:
                raise ValueError(
                    "Frame collection owns resets and requires terminations"
                )
            if env_cfg.override_cfg.get(
                "enable_pose_reward", True
            ) or env_cfg.override_cfg.get("use_reward_model", False):
                raise ValueError("Frame labels require pose and model rewards disabled")
            if env_cfg.get("keyboard_reward_wrapper"):
                raise ValueError(
                    "SpaceMouse labeling cannot use a keyboard reward wrapper"
                )
            if env_cfg.max_episode_steps != env_cfg.override_cfg.max_num_steps:
                raise ValueError("Inner and outer episode limits must match")
            if min(self.target_success, self.target_fail, cfg.runner.fps) <= 0:
                raise ValueError("Frame targets and fps must be positive")
            available = list(env_cfg.override_cfg.camera_names.values())
            if any(key not in available for key in cfg.runner.camera_keys):
                raise ValueError("Reward camera_keys must name configured cameras")

        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        self.listener = KeyboardListener() if self.label_source == "keyboard" else None
        self.step_count = 0

    def _nhwc_to_chw(self, img: torch.Tensor) -> torch.Tensor:
        """Convert NHWC (H, W, C) image to CHW (C, H, W)."""
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            return img.permute(2, 0, 1)
        return img

    def _extract_main_image(self, obs: dict) -> torch.Tensor | None:
        """Extract and normalize the main camera image from observation dict.

        Prioritizes 'main_images', falling back to 'images'.  Both inputs
        may arrive as [1, H, W, C] (batch dim = 1); this is squeezed to
        [H, W, C] so the resulting .pt stores NHWC images without a leading
        batch dimension, consistent with RewardBinaryDataset expectations.
        """
        obs = dict(obs)
        obs.pop("task_descriptions", None)

        img: torch.Tensor | None = None
        if "main_images" in obs:
            img = obs["main_images"]
        elif "images" in obs:
            img = self._nhwc_to_chw(obs["images"])

        if img is None:
            return None

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.cpu()
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        return img

    def _print_progress(self):
        s_ok = len(self.success_frames)
        f_ok = len(self.fail_frames)
        s_bar = "#" * s_ok + "-" * max(0, self.target_success - s_ok)
        f_bar = "#" * f_ok + "-" * max(0, self.target_fail - f_ok)
        print(
            f"\r  success: {s_ok}/{self.target_success} [{s_bar}]  "
            f"fail: {f_ok}/{self.target_fail} [{f_bar}]",
            end="",
            flush=True,
        )

    def _check_key(self):
        key = self.listener.get_key()
        if key == "c":
            return "success"
        elif key == "a":
            return "fail"
        return None

    async def request_stop(self) -> None:
        """Request a graceful flush after the current step."""
        self._quit = True

    async def run(self) -> None:
        """Collect in a thread so the actor can receive stop requests."""
        if self.label_source == "keyboard":
            self._run_legacy()
            self.env.close()
            return
        try:
            await asyncio.to_thread(self._run_spacemouse)
        finally:
            self.env.close()

    def _run_spacemouse(self):
        from rlinf.data.reward_collection import (
            save_reward_episode,
            select_reward_images,
            split_reward_episodes,
        )

        cfg = self.cfg
        camera_keys = list(cfg.runner.camera_keys)
        available = list(cfg.env.eval.override_cfg.camera_names.values())
        metadata = {
            "camera_keys": camera_keys,
            "preprocessing": {
                "layout": "VHWC",
                "dtype": "uint8",
                "color_space": "RGB",
                "camera_crop_regions": OmegaConf.to_container(
                    cfg.env.eval.override_cfg.get("camera_crop_regions")
                    or OmegaConf.create({}),
                    resolve=True,
                ),
            },
        }
        raw_dir = os.path.join(cfg.runner.logger.log_path, "raw_reward_episodes")
        images, labels, steps = [], [], []
        episode_id = 0
        success_count = fail_count = 0
        step = 0
        period = 1.0 / float(cfg.runner.get("fps", 10))
        try:
            self.env.reset()
            while not self._quit:
                started = time.monotonic()
                obs, _, terminated, truncated, info = self.env.step(
                    np.zeros(self.env.action_space.shape, dtype=np.float32)
                )
                step += 1
                label = int(bool(np.asarray(info["right"]).reshape(-1)[0]))
                images.append(
                    select_reward_images(
                        obs, camera_keys, cfg.env.eval.main_image_key, available
                    )
                )
                labels.append(label)
                steps.append(step)
                success_count += label
                fail_count += 1 - label
                if bool(terminated.any() or truncated.any()):
                    save_reward_episode(
                        raw_dir, episode_id, images, labels, steps, metadata
                    )
                    images, labels, steps = [], [], []
                    episode_id += 1
                    step = 0
                    self.log_info(
                        f"Reward frames: success={success_count}/{self.target_success}, failure={fail_count}/{self.target_fail}; episodes={episode_id}"
                    )
                    self.env.reset()
                    if (
                        success_count >= self.target_success
                        and fail_count >= self.target_fail
                    ):
                        break
                time.sleep(max(0, period - (time.monotonic() - started)))
        finally:
            save_reward_episode(raw_dir, episode_id, images, labels, steps, metadata)
        split_reward_episodes(
            raw_dir,
            cfg.runner.logger.log_path,
            self.val_split,
            self.fail_success_ratio,
            self.random_seed,
        )

    def _run_legacy(self):
        self._extract_main_image(self.env.reset()[0])
        max_steps = self.cfg.env.eval.max_episode_steps

        logger.info(
            f"Starting frame collection (single episode): "
            f"target {self.target_success} success frames, "
            f"{self.target_fail} fail frames | 'c'=success 'a'=fail"
        )

        while not self._quit:
            self._print_progress()

            s_ok = len(self.success_frames)
            f_ok = len(self.fail_frames)
            if s_ok >= self.target_success and f_ok >= self.target_fail:
                logger.info("Target frame counts reached, ending collection.")
                break

            action = np.zeros(self.env.action_space.shape, dtype=np.float32)
            next_obs, reward, done, _, info = self.env.step(action)

            if "intervene_action" in info:
                action = info["intervene_action"]

            img = self._extract_main_image(next_obs)
            if img is not None:
                label = self._check_key()
                if (
                    label == "success"
                    and len(self.success_frames) < self.target_success
                ):
                    self.success_frames.append(img.clone())
                elif label == "fail" and len(self.fail_frames) < self.target_fail:
                    self.fail_frames.append(img.clone())

            self.step_count += 1

            if self.step_count >= max_steps:
                logger.warning(
                    f"Max steps {max_steps} reached, exiting early. "
                    f"success {len(self.success_frames)}/{self.target_success}, "
                    f"fail {len(self.fail_frames)}/{self.target_fail}"
                )
                break

        print()
        self._save_pt()

    def _save_pt(self):
        out_dir = self.cfg.runner.logger.log_path
        os.makedirs(out_dir, exist_ok=True)

        success_frames = self.success_frames
        fail_frames = self.fail_frames

        total_frames = len(success_frames) + len(fail_frames)
        total_success = len(success_frames)

        logger.info(
            f"Loaded 1 episode, {total_frames} frames (all): "
            f"{total_success} success, {total_frames - total_success} fail"
        )

        rng = random.Random(self.random_seed)

        pairs = [(f, 1) for f in success_frames] + [(f, 0) for f in fail_frames]
        rng.shuffle(pairs)
        all_images, all_labels = zip(*pairs) if pairs else ([], [])

        n = len(all_images)
        n_val = max(1, int(n * self.val_split))
        val_images, val_labels = list(all_images[:n_val]), list(all_labels[:n_val])
        train_images, train_labels = (
            list(all_images[n_val:]),
            list(all_labels[n_val:]),
        )

        num_train_success = sum(train_labels)
        num_val_success = sum(val_labels)

        logger.info(
            f"Episode split: {1 if len(train_images) > 0 else 0} train eps, "
            f"{1 if len(val_images) > 0 else 0} val eps"
        )

        logger.info("Processing train set:")
        logger.info(
            f"  Raw: {num_train_success} success, "
            f"{len(train_labels) - num_train_success} fail"
        )

        if self.fail_success_ratio > 0 and num_train_success > 0:
            target_train_fail = int(num_train_success * self.fail_success_ratio)
            fail_indices = [i for i, l in enumerate(train_labels) if l == 0]
            rng.shuffle(fail_indices)
            fail_keep = set(fail_indices[:target_train_fail])
            train_keep = [i for i, l in enumerate(train_labels) if l == 1]
            train_keep += list(fail_keep)
            rng.shuffle(train_keep)
            train_images = [train_images[i] for i in train_keep]
            train_labels = [train_labels[i] for i in train_keep]
            num_train_success = sum(train_labels)
            logger.info(
                f"  After {self.fail_success_ratio}:1 ratio: {num_train_success} success, "
                f"{len(train_labels) - num_train_success} fail"
            )

        logger.info("Processing val set:")
        logger.info(
            f"  Raw: {num_val_success} success, "
            f"{len(val_labels) - num_val_success} fail"
        )

        metadata = {
            "num_success_frames": total_success,
            "num_fail_frames": total_frames - total_success,
            "total_frames": total_frames,
            "val_split": self.val_split,
            "fail_success_ratio": self.fail_success_ratio,
            "random_seed": self.random_seed,
            "num_train_samples": len(train_images),
            "num_val_samples": len(val_images),
        }

        train_path = f"{out_dir}/train.pt"
        val_path = f"{out_dir}/val.pt"

        RewardDatasetPayload(
            images=train_images, labels=train_labels, metadata=metadata
        ).save(train_path)
        RewardDatasetPayload(
            images=val_images, labels=val_labels, metadata=metadata
        ).save(val_path)

        logger.info(
            f"Episode-based split complete - "
            f"Train: {len(train_images)} frames "
            f"({num_train_success} success), "
            f"Val: {len(val_images)} frames ({sum(val_labels)} success)"
        )

        logger.info("=" * 60)
        logger.info("Reward dataset preprocessing complete")
        logger.info(
            f"Train split: {train_path} ({metadata['num_train_samples']} samples)"
        )
        logger.info(f"Val split:   {val_path} ({metadata['num_val_samples']} samples)")
        logger.info("Metadata:")
        logger.info(json.dumps(metadata, indent=2))
        logger.info("=" * 60)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_collect_dataset",
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = FrameCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    if cfg.runner.get("label_source", "keyboard") != "spacemouse_right":
        collector.run().wait()
        return

    def request_stop(signum, frame):
        collector.request_stop()

    previous_handlers = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, request_stop)
        collector.run().wait()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
