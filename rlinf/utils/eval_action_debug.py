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


def enrich_action_chunk(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize executed primitives without treating planned suffixes as sent."""
    if record.get("type") != "action_chunk":
        return record
    record = dict(record)
    steps = record["steps"]
    sent = record["sent_action_chunk"]
    count = len(steps)
    discarded = max(0, len(sent[0]) - count)
    last = steps[-1] if steps else {}
    control = last.get("control_info", {})

    def flagged(value: Any) -> bool:
        return bool(np.asarray(value).any()) if value is not None else False

    reason = None
    if discarded:
        if flagged(control.get("manual_failure")):
            reason = "manual_failure"
        elif flagged(last.get("terminated")):
            reason = "terminated"
        elif flagged(last.get("truncated")):
            reason = "truncated"
        elif flagged(control.get("lamp_control_released")):
            reason = "control_released"
        elif flagged(last.get("intervene_flag")):
            reason = "intervention"
        elif flagged(control.get("lamp_control_changed")):
            reason = "control_changed"
        else:
            reason = "unknown"
    record.update(
        schema_version=2,
        primitive_step_start=steps[0]["step"] if steps else None,
        executed_step_count=count,
        discarded_suffix_count=discarded,
        suffix_discard_reason=reason,
        applied_env_action_chunk=[step.get("executed_action") for step in steps],
        applied_action_source=[
            "intervention" if flagged(step.get("intervene_flag")) else "policy"
            for step in steps
        ],
    )
    return record


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
            record = enrich_action_chunk(jsonable(record))
            path = self.directory / (
                f"episode_{record['episode']:04d}_env_{record['rank']:03d}.jsonl"
            )
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
