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

"""Self-contained, versioned artifacts for LAMP priors and policies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 1


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize metadata deterministically for hashing and validation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def metadata_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def save_artifact(
    output_dir: str | Path,
    *,
    model: torch.nn.Module,
    metadata: Mapping[str, Any],
    statistics: Mapping[str, np.ndarray],
) -> Path:
    """Write an atomic-ish deployment artifact without optimizer state."""

    from safetensors.torch import save_file

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = _json_value(dict(metadata))
    payload["schema_version"] = SCHEMA_VERSION
    payload["metadata_sha256"] = metadata_sha256(
        {key: value for key, value in payload.items() if key != "metadata_sha256"}
    )
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(tensors, str(output / "model.safetensors"))
    np.savez(
        output / "statistics.npz",
        **{name: np.asarray(value) for name, value in statistics.items()},
    )
    (output / "artifact.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def load_artifact(
    artifact_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Load and strictly validate a native RLinf LAMP artifact."""

    from safetensors.torch import load_file

    root = Path(artifact_dir).expanduser().resolve()
    required = ("artifact.json", "model.safetensors", "statistics.npz")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"LAMP artifact {root} is missing {missing}")
    metadata = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported LAMP artifact schema {metadata.get('schema_version')!r}"
        )
    stored_hash = metadata.get("metadata_sha256")
    expected_hash = metadata_sha256(
        {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    )
    if stored_hash != expected_hash:
        raise ValueError("LAMP artifact metadata hash mismatch")
    state = load_file(str(root / "model.safetensors"), device="cpu")
    with np.load(root / "statistics.npz", allow_pickle=False) as data:
        statistics = {name: np.asarray(data[name]).copy() for name in data.files}
    expected_keys = set(metadata.get("statistics_keys", ()))
    if expected_keys and set(statistics) != expected_keys:
        raise ValueError("LAMP artifact statistics keys do not match artifact.json")
    return metadata, state, statistics


def save_training_state(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    global_step: int,
    sampler_state: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> Path:
    """Save exact-resume state separately from the deployment artifact."""

    output = Path(checkpoint_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "sampler_state": None if sampler_state is None else dict(sampler_state),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
        },
        "metadata": _json_value(dict(metadata)),
    }
    torch.save(payload, output / "training_state.pt")
    return output


def load_training_state(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    expected_metadata: Mapping[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    """Restore a native checkpoint and return step plus sampler state."""

    path = Path(checkpoint_dir).expanduser().resolve() / "training_state.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported LAMP training checkpoint schema")
    if payload.get("metadata") != _json_value(dict(expected_metadata)):
        raise ValueError("LAMP resume metadata differs from the current config/data")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        if payload.get("optimizer_state") is None:
            raise ValueError("LAMP checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload.get("scheduler_state") is None:
            raise ValueError("LAMP checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler_state"])
    rng = payload["rng_state"]
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    np.random.set_state(rng["numpy"])
    return int(payload["global_step"]), payload.get("sampler_state")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported artifact metadata type {type(value).__name__}")


__all__ = [
    "SCHEMA_VERSION",
    "load_artifact",
    "load_training_state",
    "metadata_sha256",
    "save_artifact",
    "save_training_state",
]
