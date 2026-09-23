# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Keep control-node reward data imports independent of language model packages."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def test_reward_data_without_transformers():
    script = """
import importlib.abc
import sys
import tempfile
from pathlib import Path

class NoTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "transformers" or fullname.startswith("transformers."):
            raise ModuleNotFoundError("transformers blocked for control-node test")

sys.meta_path.insert(0, NoTransformers())
import torch
from rlinf.data.datasets.reward_model import RewardDatasetPayload, RewardBinaryDataset
from rlinf.data.reward_collection import stack_camera_frames
from rlinf.data.datasets import DatasetItem, collate_fn, sft_collate_fn

with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory) / "frames.pt")
    RewardDatasetPayload(
        images=[torch.zeros(2, 8, 8, 3, dtype=torch.uint8)],
        labels=[1], metadata={"camera_keys": ["wrist_1", "global"]},
    ).save(path)
    assert RewardDatasetPayload.load(path).labels == [1]
    assert len(RewardBinaryDataset(path, ["wrist_1", "global"])) == 1
assert "transformers" not in sys.modules
assert "rlinf.data.datasets.reasoning" not in sys.modules
assert "rlinf.data.datasets.vlm" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "kind,name,module",
    [
        ("reasoning", "ReasoningDataset", "reasoning"),
        ("math", "ReasoningDataset", "reasoning"),
        ("rstar2", "Rstar2Dataset", "rstar2"),
        ("wideseek_r1", "WideSeekR1Dataset", "wideseek_r1"),
    ],
)
@pytest.mark.parametrize("eval_only", [False, True])
def test_existing_factory_and_exports(monkeypatch, kind, name, module, eval_only):
    import rlinf.data.datasets as datasets

    class FakeDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = ModuleType(f"rlinf.data.datasets.{module}")
    setattr(fake_module, name, FakeDataset)
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    # Register restoration before the package caches its lazy export.
    monkeypatch.setitem(vars(datasets), name, None)
    monkeypatch.delitem(vars(datasets), name)
    assert getattr(datasets, name) is FakeDataset
    cfg = OmegaConf.create({
        "runner": {"task_type": "reasoning_eval" if eval_only else "reasoning"},
        "data": {"type": kind, "train_data_paths": "train", "val_data_paths": "val"},
    })
    tokenizer = object()
    train, val = datasets.create_rl_dataset(cfg, tokenizer)
    assert val.kwargs == {"data_paths": "val", "config": cfg, "tokenizer": tokenizer}
    if eval_only:
        assert train is None
    else:
        assert train.kwargs["data_paths"] == "train"


def test_vlm_factory_remains_available(monkeypatch):
    import rlinf.data.datasets as datasets

    class Registry:
        @staticmethod
        def create(name, **kwargs):
            return name, kwargs["data_paths"]

    fake_module = ModuleType("rlinf.data.datasets.vlm")
    fake_module.VLMDatasetRegistry = Registry
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    cfg = OmegaConf.create({
        "data": {
            "type": "vision_language",
            "dataset_name": "example",
            "train_data_paths": "train",
            "val_data_paths": "val",
        }
    })
    assert datasets.create_rl_dataset(cfg, object()) == (
        ("example", "train"),
        ("example", "val"),
    )
