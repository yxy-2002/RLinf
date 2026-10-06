# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small framework-neutral/Torch training utilities shared by LAMP."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import socket
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch


def episode_ids(history_mask: np.ndarray, future_mask: np.ndarray) -> np.ndarray:
    """Recover episode boundaries from valid-future tails; validate history resets."""
    ends = np.flatnonzero(future_mask.sum(1) == 1)
    starts = np.r_[0, ends[:-1] + 1]
    assert ends[-1] == len(future_mask) - 1
    assert np.all(history_mask[starts].sum(1) <= 1)
    return np.repeat(np.arange(len(ends)), ends - starts + 1)


class CosineSchedule:
    """Absolute learning-rate schedule matching the old warmup/cosine recipe."""

    def __init__(
        self,
        lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        self.lr = float(lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.min_lr = float(min_lr)
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.lr < 0.0 or self.min_lr < 0.0 or self.min_lr > self.lr:
            raise ValueError("learning rates must satisfy 0 <= min_lr <= lr")
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps cannot exceed total_steps")

    def __call__(self, step: int | torch.Tensor) -> float | torch.Tensor:
        if isinstance(step, torch.Tensor):
            value = step.to(dtype=torch.float32)
            if self.warmup_steps:
                warm = self.lr * value / float(self.warmup_steps)
            else:
                warm = torch.full_like(value, self.lr)
            decay_steps = max(self.total_steps - self.warmup_steps, 1)
            progress = ((value - self.warmup_steps) / decay_steps).clamp(0.0, 1.0)
            cosine = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1.0 + torch.cos(math.pi * progress)
            )
            return torch.where(value < self.warmup_steps, warm, cosine)
        index = int(step)
        if index < 0:
            raise ValueError("schedule step must be non-negative")
        if self.warmup_steps and index < self.warmup_steps:
            return self.lr * index / self.warmup_steps
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = min(max((index - self.warmup_steps) / decay_steps, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def state_dict(self) -> dict[str, float | int]:
        return {
            "lr": self.lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.state_dict():
            raise ValueError(
                "checkpoint learning-rate schedule does not match this run"
            )


def cosine_schedule(
    lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> CosineSchedule:
    return CosineSchedule(lr, total_steps, warmup_steps, min_lr)


def configure_torch_runtime() -> None:
    """Configure the default fast FP32/TF32 runtime used by LAMP in RLinf."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)


def beta_warmup(
    step: int | torch.Tensor,
    beta: float,
    warmup_steps: int,
) -> float | torch.Tensor:
    if warmup_steps <= 0:
        if isinstance(step, torch.Tensor):
            return torch.full_like(step, float(beta), dtype=torch.float32)
        return float(beta)
    if isinstance(step, torch.Tensor):
        return float(beta) * (step.to(torch.float32) / float(warmup_steps)).clamp(
            max=1.0
        )
    return float(beta) * min(1.0, max(float(step), 0.0) / float(warmup_steps))


class StatefulBatchSampler(Iterator[np.ndarray]):
    """Infinite deterministic minibatch sampler with exact resume state."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        n: int,
        batch_size: int,
        *,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if self.n < 1 or self.batch_size < 1:
            raise ValueError("n and batch_size must be positive")
        if self.drop_last and self.n < self.batch_size:
            raise ValueError("drop_last=True requires n >= batch_size")
        self.rng = np.random.default_rng(self.seed)
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.emitted = 0

    def __iter__(self) -> "StatefulBatchSampler":
        return self

    def __next__(self) -> np.ndarray:
        while True:
            if self.cursor >= len(self.order):
                self.order = self.rng.permutation(self.n).astype(np.int64, copy=False)
                self.cursor = 0
            end = min(self.cursor + self.batch_size, self.n)
            batch = self.order[self.cursor : end].copy()
            self.cursor = end
            if len(batch) < self.batch_size and self.drop_last:
                continue
            self.emitted += 1
            return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "n": self.n,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "rng_state": self.rng.bit_generator.state,
            "order": self.order.copy(),
            "cursor": self.cursor,
            "emitted": self.emitted,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "n",
            "batch_size",
            "seed",
            "drop_last",
            "rng_state",
            "order",
            "cursor",
            "emitted",
        }
        if set(state) != expected:
            raise ValueError("sampler state has missing or unexpected fields")
        contract = (state["n"], state["batch_size"], state["seed"], state["drop_last"])
        if contract != (self.n, self.batch_size, self.seed, self.drop_last):
            raise ValueError("sampler state does not match the live sampler contract")
        if state["schema_version"] != self.SCHEMA_VERSION:
            raise ValueError("unsupported sampler state schema")
        order = np.asarray(state["order"], dtype=np.int64)
        cursor = int(state["cursor"])
        emitted = int(state["emitted"])
        if order.shape not in ((0,), (self.n,)):
            raise ValueError("sampler permutation has an invalid shape")
        if order.size and not np.array_equal(np.sort(order), np.arange(self.n)):
            raise ValueError("sampler permutation is invalid")
        if not 0 <= cursor <= len(order) or emitted < 0:
            raise ValueError("sampler cursor/emitted values are invalid")

        def numpy_value(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy()
                return array.item() if array.ndim == 0 else array
            if isinstance(value, Mapping):
                return {key: numpy_value(item) for key, item in value.items()}
            if isinstance(value, list):
                return [numpy_value(item) for item in value]
            return value

        self.rng.bit_generator.state = numpy_value(state["rng_state"])
        self.order = order.copy()
        self.cursor = cursor
        self.emitted = emitted


def minibatch_indices(
    n: int,
    batch_size: int,
    *,
    seed: int,
    steps: int | None = None,
    drop_last: bool = False,
) -> Iterator[np.ndarray]:
    sampler = StatefulBatchSampler(n, batch_size, seed=seed, drop_last=drop_last)
    if steps is None:
        return sampler

    def finite() -> Iterator[np.ndarray]:
        for _ in range(int(steps)):
            yield next(sampler)

    return finite()


def split_episodes(
    episode_index: np.ndarray, train_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    if len(episodes) <= 1:
        return np.arange(len(episode_index)), np.arange(0, dtype=np.int64)
    n_train = min(max(int(round(len(episodes) * train_ratio)), 1), len(episodes) - 1)
    train_eps = set(episodes[:n_train].tolist())
    train_idx = np.flatnonzero(np.isin(episode_index, list(train_eps)))
    val_idx = np.flatnonzero(~np.isin(episode_index, list(train_eps)))
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def tree_to_numpy(tree: Any) -> Any:
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().numpy()
    if isinstance(tree, Mapping):
        return type(tree)((key, tree_to_numpy(value)) for key, value in tree.items())
    if isinstance(tree, tuple):
        return tuple(tree_to_numpy(value) for value in tree)
    if isinstance(tree, list):
        return [tree_to_numpy(value) for value in tree]
    return tree


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
    *,
    backbone_ratio: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr) * (
            float(backbone_ratio) if group.get("group_name") == "backbone" else 1.0
        )


def clip_optimizer_groups(
    optimizer: torch.optim.Optimizer, max_norm: float
) -> dict[str, float]:
    norms: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        parameters = [p for p in group["params"] if p.grad is not None]
        name = str(group.get("group_name", index))
        if parameters:
            norm = torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
            norms[name] = float(norm.detach().cpu())
    return norms


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _best_effort_command(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output or None if result.returncode == 0 else None


def run_metadata(started_monotonic: float | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "git_commit": _best_effort_command(["git", "rev-parse", "HEAD"]),
        "git_dirty": None,
        "gpu": _best_effort_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "started_at_utc": _utc_now_iso(),
        "finished_at_utc": None,
        "duration_seconds": None,
    }
    dirty = _best_effort_command(["git", "status", "--porcelain"])
    if dirty is not None:
        metadata["git_dirty"] = bool(dirty)
    if started_monotonic is not None:
        metadata["_started_monotonic"] = started_monotonic
    return metadata


def finish_run_metadata(
    metadata: dict[str, Any], started_monotonic: float | None = None
) -> dict[str, Any]:
    import time

    metadata = dict(metadata)
    start = started_monotonic
    if start is None:
        start = metadata.pop("_started_monotonic", None)
    else:
        metadata.pop("_started_monotonic", None)
    metadata["finished_at_utc"] = _utc_now_iso()
    if start is not None:
        metadata["duration_seconds"] = round(time.monotonic() - float(start), 3)
    return metadata


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        raise TypeError(type(value).__name__)

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=default))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_stats(x: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
    return {
        "mean": x.mean(axis=0).astype(np.float32),
        "std": (x.std(axis=0) + eps).astype(np.float32),
    }
