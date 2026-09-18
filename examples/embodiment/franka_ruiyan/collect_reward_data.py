# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Manual reward-frame collection with held-success, single-frame, and pause labels.

SpaceMouse buttons are exclusively teleoperation inputs. Labels describe the
pre-action image. Raw samples are checkpointed independently of train/val splits.
"""

import os
import random

import hydra
import numpy as np
import torch

from examples.reward.realworld_collect_process_dataset import FrameCollector
from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.scheduler import Cluster, ComponentPlacement


def frame_label(held, pressed):
    """Return a manual label, or None to skip; never inspect teleop buttons."""
    if held == "b" or "b" in pressed:
        return None
    if held == "c" or "c" in pressed or "Key.space" in pressed:
        return 1
    return 0


class RuiyanRewardCollector(FrameCollector):
    def _reward_image_keys(self):
        return list(self.cfg.runner.get("reward_image_keys", []))

    def _extract_reward_images(self, obs):
        keys = self._reward_image_keys()
        if not keys:
            return self._extract_main_image(obs)
        from rlinf.data.reward_views import select_reward_views

        main = self.cfg.env.eval.main_image_key
        names = list(self.cfg.env.eval.override_cfg.camera_names.values())
        extras = sorted(name for name in names if name != main)
        images = select_reward_views(obs, keys, main, extras)
        if images.shape[0] != 1:
            raise ValueError("Reward collector requires exactly one environment")
        return images[0].detach().cpu()

    def _create_listener(self):
        from examples.embodiment.franka_ruiyan.remote_reward_labels import RemoteLabels

        return RemoteLabels(port=self.cfg.runner.get("label_port", 8766))

    def _save_raw(self):
        directory = self.cfg.runner.logger.log_path
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "raw_reward_frames.pt")
        torch.save(
            {
                "success_images": self.success_frames,
                "failure_images": self.fail_frames,
                "metadata": {
                    "step_count": self.step_count,
                    "image_key": self.cfg.env.eval.get("main_image_key", "wrist_1"),
                    "label_observation": "observation",
                    "label_source": "manual_keyboard",
                    "image_keys": self._reward_image_keys(),
                },
            },
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)

    def _save_pt(self):
        # Stratify the frame split so both classes occur in both outputs.
        # Independent recording sessions are still needed for final evaluation.
        rng = random.Random(self.random_seed)
        train, val = [], []
        for label, frames in ((1, self.success_frames), (0, self.fail_frames)):
            indices = list(range(len(frames)))
            rng.shuffle(indices)
            count = max(1, min(len(frames) - 1, int(len(frames) * self.val_split)))
            val.extend((frames[i], label) for i in indices[:count])
            train.extend((frames[i], label) for i in indices[count:])
        positive = [item for item in train if item[1] == 1]
        negative = [item for item in train if item[1] == 0]
        if self.fail_success_ratio > 0:
            negative = negative[: int(len(positive) * self.fail_success_ratio)]
        train = positive + negative
        for name, pairs in (("train", train), ("val", val)):
            rng.shuffle(pairs)
            images, labels = zip(*pairs)
            path = os.path.join(self.cfg.runner.logger.log_path, name + ".pt")
            RewardDatasetPayload(
                images=list(images),
                labels=list(labels),
                metadata={
                    "split_method": "stratified_frames",
                    "label_source": "manual_keyboard",
                    "image_keys": self._reward_image_keys(),
                },
            ).save(path)
            print(f"Saved {path}: {len(pairs)} frames", flush=True)

    def run(self):
        """Collect until success quota, q, step limit, or environment termination."""
        try:
            previous_obs, _ = self.env.reset()
            # Ignore key events generated during initialization and robot reset.
            self.listener.pop_pressed_keys()
            print(
                "Ready: connect remote label client; hold c=continuous success; Space=one success; initially paused; b=pause; a=record failures; q=save/exit",
                flush=True,
            )
            while len(self.success_frames) < self.target_success:
                shape = (1, self.env.action_space.shape[-1])
                obs, _, done, truncated, _ = self.env.step(
                    np.zeros(shape, dtype=np.float32)
                )
                # Consume label events after the step and apply them to obs_t.
                pressed = self.listener.pop_pressed_keys()
                held = self.listener.get_key()
                if held == "q" or "q" in pressed:
                    break
                label = frame_label(held, pressed)
                image = self._extract_reward_images(previous_obs)
                previous_obs = obs
                if image is None:
                    raise ValueError(
                        "No main camera image; refusing to collect unlabeled observations"
                    )
                if label == 1:
                    self.success_frames.append(image.clone())
                    self._save_raw()
                    self.listener.confirm_success()
                    print(
                        f"Success frame saved: {len(self.success_frames)}.",
                        flush=True,
                    )
                elif label == 0:
                    self.fail_frames.append(image.clone())
                self.step_count += 1
                if self.step_count % self.cfg.runner.get("raw_save_interval", 50) == 0:
                    self._save_raw()
                    self._print_progress()
                if bool(torch.as_tensor(done).any()) or bool(
                    torch.as_tensor(truncated).any()
                ):
                    print(
                        "Environment ended; saving without an automatic reset.",
                        flush=True,
                    )
                    break
                limit = self.cfg.env.eval.max_episode_steps
                if limit is not None and self.step_count >= limit:
                    break
        finally:
            try:
                self._save_raw()
                if len(self.success_frames) >= 2 and len(self.fail_frames) >= 2:
                    self._save_pt()
                else:
                    print(
                        "Raw frames saved; need at least 2 frames per class to export splits.",
                        flush=True,
                    )
            finally:
                try:
                    self.env.close()
                finally:
                    self.listener.close()


@hydra.main(
    version_base="1.1",
    config_path="../../reward/config",
    config_name="realworld_collect_ruiyan_dataset",
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster).get_strategy("env")
    collector = RuiyanRewardCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
