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
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 1
RESUME_FLOAT_RTOL = 1e-7
RESUME_FLOAT_ATOL = 1e-7
_ARTIFACT_FILENAMES = ("artifact.json", "model.safetensors", "statistics.npz")


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize metadata deterministically for hashing and validation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def metadata_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics_sha256(statistics: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(statistics):
        value = np.ascontiguousarray(np.asarray(statistics[name]))
        header = {
            "name": name,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _copy_file_portably(source: Path, destination: Path) -> None:
    """Copy a locally staged file without relying on filesystem rename support."""

    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    destination.chmod(0o644)


def _validate_safetensors_header(path: Path) -> None:
    """Parse a safetensors header without materializing tensor payloads."""

    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu", backend="pread") as data:
        tuple(data.keys())


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
    metadata_path = output / "artifact.json"
    metadata_path.unlink(missing_ok=True)
    payload = _json_value(dict(metadata))
    payload["schema_version"] = SCHEMA_VERSION
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    model_path = output / "model.safetensors"
    statistics_path = output / "statistics.npz"
    statistic_arrays = {name: np.asarray(value) for name, value in statistics.items()}
    staging_root = os.environ.get("RLINF_ARTIFACT_STAGING_DIR")
    with tempfile.TemporaryDirectory(
        prefix="rlinf-lamp-artifact-", dir=staging_root
    ) as staging_dir:
        staged_model = Path(staging_dir) / "model.safetensors"
        staged_statistics = Path(staging_dir) / "statistics.npz"
        save_file(tensors, str(staged_model))
        _validate_safetensors_header(staged_model)
        np.savez(staged_statistics, **statistic_arrays)
        expected_model_sha = _file_sha256(staged_model)
        _copy_file_portably(staged_model, model_path)
        _copy_file_portably(staged_statistics, statistics_path)
    if _file_sha256(model_path) != expected_model_sha:
        raise OSError(f"LAMP artifact copy verification failed for {model_path}")
    _validate_safetensors_header(model_path)
    with np.load(statistics_path, allow_pickle=False) as stored_statistics:
        if set(stored_statistics.files) != set(statistic_arrays):
            raise OSError(
                f"LAMP artifact statistics verification failed for {statistics_path}"
            )
    payload["model_sha256"] = expected_model_sha
    payload["statistics_sha256"] = _statistics_sha256(statistic_arrays)
    payload["metadata_sha256"] = metadata_sha256(
        {key: value for key, value in payload.items() if key != "metadata_sha256"}
    )
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata_path.chmod(0o644)
    return output


def load_artifact(
    artifact_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Load and strictly validate a native RLinf LAMP artifact."""

    from safetensors.torch import load_file

    root = resolve_artifact_dir(artifact_dir)
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
    expected_model_checksum = metadata.get("model_sha256")
    if (
        expected_model_checksum is not None
        and _file_sha256(root / "model.safetensors") != expected_model_checksum
    ):
        raise ValueError("LAMP artifact model_sha256 mismatch")
    state = load_file(str(root / "model.safetensors"), device="cpu")
    with np.load(root / "statistics.npz", allow_pickle=False) as data:
        statistics = {name: np.asarray(data[name]).copy() for name in data.files}
    expected_statistics_checksum = metadata.get("statistics_sha256")
    if (
        expected_statistics_checksum is not None
        and _statistics_sha256(statistics) != expected_statistics_checksum
    ):
        raise ValueError("LAMP artifact statistics_sha256 mismatch")
    expected_keys = set(metadata.get("statistics_keys", ()))
    if expected_keys and set(statistics) != expected_keys:
        raise ValueError("LAMP artifact statistics keys do not match artifact.json")
    return metadata, state, statistics


def resolve_artifact_dir(artifact_path: str | Path) -> Path:
    """Resolve a LAMP artifact from one of three deterministic locations.

    ``artifact_path`` may be the artifact itself, a training run containing an
    ``artifact`` directory, or a run whose actor owns the artifact. Deliberately
    avoid recursive search and latest-checkpoint selection: an ambiguous run
    layout must be fixed by the caller instead of being guessed here.
    """

    root = Path(artifact_path).expanduser().resolve()
    candidates = (root, root / "artifact", root / "actor" / "artifact")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in _ARTIFACT_FILENAMES):
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    required = ", ".join(_ARTIFACT_FILENAMES)
    raise FileNotFoundError(
        "Could not resolve a complete LAMP artifact. "
        f"Checked [{checked}] for [{required}]"
    )


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
    stored_metadata = payload.get("metadata")
    expected_metadata = _json_value(dict(expected_metadata))
    mismatches = _resume_metadata_mismatches(stored_metadata, expected_metadata)
    if mismatches:
        details = "; ".join(mismatches[:8])
        raise ValueError(
            "LAMP resume metadata differs from the current config/data: " + details
        )
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


def _resume_metadata_mismatches(
    stored: Any,
    expected: Any,
    *,
    path: str = "metadata",
) -> list[str]:
    """Return semantic resume-metadata differences with float tolerance."""

    if isinstance(stored, Mapping) and isinstance(expected, Mapping):
        mismatches = []
        stored_keys = set(stored)
        expected_keys = set(expected)
        for key in sorted(stored_keys ^ expected_keys):
            mismatches.append(f"{path}.{key}: missing key")
        for key in sorted(stored_keys & expected_keys):
            # This is derived from the architecture payload compared below. A tiny
            # floating-point tail changes the hash despite semantic compatibility.
            if path == "metadata" and key == "architecture_sha256":
                continue
            mismatches.extend(
                _resume_metadata_mismatches(
                    stored[key], expected[key], path=f"{path}.{key}"
                )
            )
        return mismatches
    if isinstance(stored, list) and isinstance(expected, list):
        if len(stored) != len(expected):
            return [f"{path}: length {len(stored)} != {len(expected)}"]
        mismatches = []
        for index, (stored_item, expected_item) in enumerate(zip(stored, expected)):
            mismatches.extend(
                _resume_metadata_mismatches(
                    stored_item, expected_item, path=f"{path}[{index}]"
                )
            )
        return mismatches
    if (
        isinstance(stored, (int, float))
        and not isinstance(stored, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        if isinstance(stored, int) and isinstance(expected, int):
            return [] if stored == expected else [f"{path}: {stored!r} != {expected!r}"]
        if math.isclose(
            float(stored),
            float(expected),
            rel_tol=RESUME_FLOAT_RTOL,
            abs_tol=RESUME_FLOAT_ATOL,
        ):
            return []
    elif type(stored) is not type(expected):
        return [f"{path}: type {type(stored).__name__} != {type(expected).__name__}"]
    if stored != expected:
        return [f"{path}: {stored!r} != {expected!r}"]
    return []


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
    "resolve_artifact_dir",
    "save_artifact",
    "save_training_state",
]
