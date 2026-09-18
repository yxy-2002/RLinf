# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Episode-aware action windows for standalone DexJoCo prior training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def build_action_windows(
    actions: np.ndarray,
    episode_index: np.ndarray | None = None,
    *,
    history_length: int = 16,
    horizon: int = 16,
) -> dict[str, np.ndarray]:
    """Build history/future windows, never crossing episode boundaries."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or history_length < 1 or horizon < 1:
        raise ValueError("actions must be [N, action_dim] and lengths positive")
    episodes = (
        np.zeros(len(values), dtype=np.int64)
        if episode_index is None
        else np.asarray(episode_index, dtype=np.int64)
    )
    if len(episodes) != len(values):
        raise ValueError("episode_index must have one entry per action")
    history = np.empty((len(values), history_length, values.shape[1]), np.float32)
    future = np.empty((len(values), horizon, values.shape[1]), np.float32)
    history_mask = np.zeros((len(values), history_length), np.float32)
    future_mask = np.zeros((len(values), horizon), np.float32)
    for i in range(len(values)):
        valid = np.flatnonzero(episodes == episodes[i])
        pos = int(np.flatnonzero(valid == i)[0])
        hi = pos - history_length + np.arange(history_length)
        fi = pos + np.arange(horizon)
        history[i] = values[valid[np.clip(hi, 0, len(valid) - 1)]]
        future[i] = values[valid[np.clip(fi, 0, len(valid) - 1)]]
        history_mask[i] = ((hi >= 0) & (hi < len(valid))).astype(np.float32)
        future_mask[i] = ((fi >= 0) & (fi < len(valid))).astype(np.float32)
    return {
        "history": history,
        "future": future,
        "history_mask": history_mask,
        "future_mask": future_mask,
        "anchors": np.arange(len(values)),
        "episode_index": episodes,
    }


class ActionWindowDataset(Dataset):
    """Torch dataset over persisted DexJoCo action windows."""

    def __init__(self, artifact_dir: str | Path, split: str) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        root = Path(artifact_dir).expanduser().resolve()
        metadata = json.loads((root / "metadata.json").read_text())
        if metadata.get("source", {}).get("type") != "dexjoco_lerobot":
            raise ValueError("Only DexJoCo LeRobot action artifacts are supported")
        self.history = np.load(root / "history.npy", mmap_mode="r")
        self.future = np.load(root / "future.npy", mmap_mode="r")
        self.history_mask = np.load(root / "history_mask.npy", mmap_mode="r")
        self.future_mask = np.load(root / "future_mask.npy", mmap_mode="r")
        if split == "train":
            indices = np.load(root / "train_anchor_indices.npy")
        elif split == "validation":
            indices = np.load(root / "validation_anchor_indices.npy")
        else:
            indices = np.arange(len(self.history), dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        return {
            "history": torch.from_numpy(np.array(self.history[row], copy=True)).float(),
            "future_actions": torch.from_numpy(
                np.array(self.future[row], copy=True)
            ).float(),
            "history_mask": torch.from_numpy(
                np.array(self.history_mask[row], copy=True)
            ).float(),
            "future_mask": torch.from_numpy(
                np.array(self.future_mask[row], copy=True)
            ).float(),
            "anchor": torch.tensor(row, dtype=torch.long),
        }


def save_action_artifact(
    output_dir: str | Path,
    actions: np.ndarray,
    episode_index: np.ndarray | None = None,
    *,
    history_length: int,
    horizon: int,
    split_seed: int = 42,
    train_ratio: float = 0.95,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Persist episode-aware action windows and their train/validation split."""
    if source_metadata.get("type") != "dexjoco_lerobot":
        raise ValueError("Only DexJoCo LeRobot action artifacts are supported")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    windows = build_action_windows(
        actions, episode_index, history_length=history_length, horizon=horizon
    )
    for name in ("history", "future", "history_mask", "future_mask", "anchors"):
        np.save(output / f"{name}.npy", windows[name])
    np.save(output / "sequence.npy", np.asarray(actions, dtype=np.float32))
    np.save(output / "episode_index.npy", windows["episode_index"])
    rng = np.random.default_rng(split_seed)
    episodes = np.unique(windows["episode_index"])
    perm = rng.permutation(episodes)
    n_train = int(len(perm) * train_ratio)
    train_eps = set(perm[:n_train])
    train = np.flatnonzero(np.isin(windows["episode_index"], list(train_eps)))
    val = np.flatnonzero(~np.isin(windows["episode_index"], list(train_eps)))
    np.save(output / "train_anchor_indices.npy", train)
    np.save(output / "validation_anchor_indices.npy", val)
    metadata = {
        "sequence_length": int(len(actions)),
        "feature_dim": int(actions.shape[1]),
        "history_length": int(history_length),
        "future_length": int(horizon),
        "train_ratio": float(train_ratio),
        "train_count": int(len(train)),
        "validation_count": int(len(val)),
        "padding": "edge_replicate_with_mask",
        "source": source_metadata,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


__all__ = ["ActionWindowDataset", "build_action_windows", "save_action_artifact"]
