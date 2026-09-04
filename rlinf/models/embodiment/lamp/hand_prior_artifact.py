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

"""Construction and strict loading of native LAMP hand-prior artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rlinf.models.embodiment.lamp.artifact_io import load_artifact
from rlinf.models.embodiment.lamp.hand_ae import DexJoCoHandAE
from rlinf.models.embodiment.lamp.hand_cvae import DexJoCoHandCVAE
from rlinf.models.embodiment.lamp.hand_pca import HandPCA
from rlinf.models.embodiment.lamp.hand_vae import DexJoCoHandVAE
from rlinf.models.embodiment.lamp.hand_vq_vae import (
    HandVQVAE,
    decode_all_code_combinations,
)
from rlinf.models.embodiment.lamp.vq_action_normalization import (
    denormalize_vq_hand_action,
)


class TorchHandPCA(nn.Module):
    """Buffer-only Torch representation of a deterministic PCA codec."""

    def __init__(
        self,
        mean: np.ndarray | list[float],
        components: np.ndarray | list[list[float]],
        explained_variance: np.ndarray | list[float],
        explained_variance_ratio: np.ndarray | list[float],
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer(
            "components", torch.as_tensor(components, dtype=torch.float32)
        )
        self.register_buffer(
            "explained_variance",
            torch.as_tensor(explained_variance, dtype=torch.float32),
        )
        self.register_buffer(
            "explained_variance_ratio",
            torch.as_tensor(explained_variance_ratio, dtype=torch.float32),
        )

    @classmethod
    def from_pca(cls, model: HandPCA) -> "TorchHandPCA":
        return cls(
            model.mean,
            model.components,
            model.explained_variance,
            model.explained_variance_ratio,
        )

    def encode(self, values: torch.Tensor, latent_dim: int) -> torch.Tensor:
        return (values - self.mean) @ self.components[: int(latent_dim)].T

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent @ self.components[: latent.shape[-1]] + self.mean


def build_prior_model(prior_type: str, architecture: dict[str, Any]) -> nn.Module:
    """Instantiate one supported trainable prior architecture."""

    if prior_type == "vae":
        return DexJoCoHandVAE(**architecture)
    if prior_type == "cvae":
        return DexJoCoHandCVAE(**architecture)
    if prior_type == "ae":
        return DexJoCoHandAE(**architecture)
    if prior_type == "vq":
        return HandVQVAE(**architecture)
    if prior_type == "pca":
        return TorchHandPCA(**architecture)
    raise ValueError(f"Unsupported LAMP prior type {prior_type!r}")


def load_prior_artifact(
    artifact_dir: str | Path,
    *,
    expected_type: str | None = None,
    expected_task: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_hand_side: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any], dict[str, np.ndarray]]:
    """Load a prior and reject any provenance/config mismatch."""

    metadata, state, statistics = load_artifact(artifact_dir)
    if metadata.get("kind") != "prior":
        raise ValueError("Expected a LAMP prior artifact")
    checks = {
        "prior_type": expected_type,
        "task": expected_task,
        "dataset_fingerprint": expected_dataset_fingerprint,
        "hand_side": expected_hand_side,
    }
    for name, expected in checks.items():
        if expected is not None and metadata.get(name) != expected:
            raise ValueError(
                f"LAMP prior {name}={metadata.get(name)!r}, expected {expected!r}"
            )
    architecture = metadata.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("LAMP prior artifact is missing architecture")
    model = build_prior_model(str(metadata["prior_type"]), architecture)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    return model, metadata, statistics


def sorted_vq_codebook(model: HandVQVAE) -> np.ndarray:
    """Decode, physicalize, and deterministically order all 16 VQ actions.

    DQ-RISE trains its tokenizer in ``[-1, 1]`` but exports the codebook in
    the physical hand-action space.  PCA ordering must happen after this
    conversion because per-joint actuator ranges are anisotropic.
    """

    decoded = (
        decode_all_code_combinations(model).detach().cpu().numpy().astype(np.float32)
    )
    raw = np.asarray(denormalize_vq_hand_action(decoded), dtype=np.float32)
    centered = raw - raw.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        projection = np.zeros((len(raw),), dtype=np.float32)
    else:
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        direction = right[0]
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction = -direction
        projection = (centered @ direction).astype(np.float32)
    return raw[np.argsort(projection, kind="stable")].astype(np.float32)


__all__ = [
    "TorchHandPCA",
    "build_prior_model",
    "load_prior_artifact",
    "sorted_vq_codebook",
]
