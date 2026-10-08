# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free serialization of aligned online evaluation action traces."""

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def jsonable(value: Any) -> Any:
    """Snapshot tensor/array data as JSON primitives, without retaining buffers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


class EvalActionDebugWriter:
    """Drain remote environment records into per-episode driver-side JSONL files."""

    def __init__(self, log_path: str, channel: Any) -> None:
        self.directory = Path(log_path) / "debug_actions"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.channel = channel

    def drain(self) -> None:
        """Persist all available records; close each write for immediate visibility."""
        while True:
            try:
                record = self.channel.get_nowait(key="records")
            except asyncio.QueueEmpty:
                return
            path = self.directory / (
                f"episode_{record['episode']:04d}_env_{record['rank']:03d}.jsonl"
            )
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
