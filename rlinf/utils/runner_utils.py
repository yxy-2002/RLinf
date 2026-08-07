# Copyright 2025 The RLinf Authors.
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

import os
import tempfile
from collections.abc import Mapping
from typing import Any, Union

from rlinf.utils.logging import get_logger

logger = get_logger()


def resolve_training_horizon(
    runner_cfg: Mapping[str, Any], *, steps_per_epoch: int | None = None
) -> tuple[int, int]:
    """Resolve optimizer steps from epoch and optional step limits.

    A runtime ``steps_per_epoch`` may be supplied after a dataset is prepared.
    An explicit config value remains supported and must agree with the runtime
    value. When neither exists, one preserves the historical runner behavior.
    A negative ``max_steps`` selects the epoch-derived horizon without a cap.
    """
    configured_steps = runner_cfg.get("steps_per_epoch", None)
    if steps_per_epoch is None:
        steps_per_epoch = 1 if configured_steps is None else int(configured_steps)
    else:
        steps_per_epoch = int(steps_per_epoch)
        if configured_steps is not None and int(configured_steps) != steps_per_epoch:
            raise ValueError(
                "runner.steps_per_epoch does not match the runtime dataset: "
                f"configured={int(configured_steps)}, runtime={steps_per_epoch}"
            )
    max_epochs = int(runner_cfg.get("max_epochs", 1))
    configured_max_steps = int(runner_cfg.get("max_steps", -1))
    if steps_per_epoch < 1:
        raise ValueError(f"runner.steps_per_epoch must be >= 1, got {steps_per_epoch}")
    if max_epochs < 1:
        raise ValueError(f"runner.max_epochs must be >= 1, got {max_epochs}")
    if configured_max_steps < -1:
        raise ValueError(
            "runner.max_steps must be -1 (no step cap) or non-negative, "
            f"got {configured_max_steps}"
        )

    max_steps = steps_per_epoch * max_epochs
    if configured_max_steps >= 0:
        max_steps = min(max_steps, configured_max_steps)
    return steps_per_epoch, max_steps


def resolve_save_interval(
    runner_cfg: Mapping[str, Any], *, steps_per_epoch: int
) -> int:
    """Resolve an epoch-based checkpoint cadence after dataset preparation."""

    save_every_epochs = runner_cfg.get("save_every_epochs", None)
    if save_every_epochs is None:
        return int(runner_cfg.get("save_interval", -1))
    save_every_epochs = int(save_every_epochs)
    if save_every_epochs < 1:
        raise ValueError(
            f"runner.save_every_epochs must be >= 1, got {save_every_epochs}"
        )
    if int(steps_per_epoch) < 1:
        raise ValueError(f"steps_per_epoch must be >= 1, got {steps_per_epoch}")
    return save_every_epochs * int(steps_per_epoch)


def safe_is_divisible(a, b):
    """a safe divisible check to allow b to be 0"""
    if a == 0 or b == 0:
        return False
    return a % b == 0


def check_progress(
    step: int,
    max_steps: int,
    val_check_interval: int,
    save_interval: int,
    limit_val_batches: Union[int, float, None],
    run_time_exceeded: bool = False,
):
    is_validation_enabled = limit_val_batches != 0 and val_check_interval > 0
    is_save_enabled = save_interval > 0
    is_train_end = step == max_steps

    if is_validation_enabled:
        assert save_interval < 0 or save_interval % val_check_interval == 0, (
            f"{save_interval=} must be divisible by {val_check_interval=}"
        )

    # run validation on the last step
    # or when we hit the val check interval
    run_val = (
        safe_is_divisible(step, val_check_interval) or is_train_end or run_time_exceeded
    )
    run_val &= is_validation_enabled

    # save model at save intervals or last step
    save_model = (
        safe_is_divisible(step, save_interval) or is_train_end or run_time_exceeded
    )
    # sometimes the user will provide a validation metric
    # to save against, so we need to run val when we save
    save_model &= is_save_enabled

    return run_val, save_model, is_train_end


def local_mkdir_safe(path):
    from filelock import FileLock

    if not os.path.isabs(path):
        working_dir = os.getcwd()
        path = os.path.join(working_dir, path)

    # Using hash value of path as lock file name to avoid long file name
    lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
    lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

    try:
        with FileLock(lock_path, timeout=60):  # Add timeout
            # make a new dir
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to acquire lock for {path}: {e}")
        # Even if the lock is not acquired, try to create the directory
        os.makedirs(path, exist_ok=True)

    return path


class EarlyStopController:
    """Track validation metrics and decide whether to stop early."""

    def __init__(self, cfg: Any) -> None:
        self.enabled = cfg.get("enabled", False)
        self.patience = cfg.get("patience", 5)
        self.min_delta = cfg.get("min_delta", 0.001)
        self.monitor = cfg.get("monitor", "val_loss")

        self.counter = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

    def update(self, metrics: dict[str, float]) -> tuple[bool, bool]:
        """Return (should_stop, best_val_acc_improved)."""
        improved_for_monitor = False
        best_val_acc_improved = False

        if "val_loss" in metrics:
            val_loss = metrics["val_loss"]
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                if self.monitor == "val_loss":
                    improved_for_monitor = True

        if "val_accuracy" in metrics:
            val_acc = metrics["val_accuracy"]
            if val_acc > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_acc
                best_val_acc_improved = True
                if self.monitor == "val_accuracy":
                    improved_for_monitor = True

        if not self.enabled:
            return False, best_val_acc_improved

        has_monitored_metrics = "val_loss" in metrics or "val_accuracy" in metrics
        if not has_monitored_metrics:
            return False, best_val_acc_improved

        if improved_for_monitor:
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"Early stop counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            logger.info(
                f"Early stopping triggered! No improvement for {self.patience} checks."
            )
            return True, best_val_acc_improved

        return False, best_val_acc_improved
