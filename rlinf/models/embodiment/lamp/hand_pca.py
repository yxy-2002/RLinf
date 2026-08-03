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

"""Deterministic PCA codec for normalized 16D DexJoCo hand actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlinf.models.embodiment.lamp.constants import HAND_ACTION_DIM


@dataclass(frozen=True)
class HandPCA:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray

    def encode(self, actions: np.ndarray, latent_dim: int) -> np.ndarray:
        actions = _actions(actions)
        latent_dim = validate_latent_dim(latent_dim)
        return ((actions - self.mean) @ self.components[:latent_dim].T).astype(
            np.float32
        )

    def decode(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float32)
        if latent.ndim < 1 or not 1 <= latent.shape[-1] <= HAND_ACTION_DIM:
            raise ValueError(f"Invalid PCA latent shape {latent.shape}")
        if not np.isfinite(latent).all():
            raise ValueError("PCA latent contains NaN or infinity")
        components = self.components[: latent.shape[-1]]
        return (latent @ components + self.mean).astype(np.float32)


def validate_latent_dim(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"PCA latent_dim must be an integer in [1, {HAND_ACTION_DIM}]")
    value = int(value)
    if not 1 <= value <= HAND_ACTION_DIM:
        raise ValueError(
            f"PCA latent_dim must be in [1, {HAND_ACTION_DIM}], got {value}"
        )
    return value


def fit_hand_pca(actions: np.ndarray) -> HandPCA:
    """Fit raw-centered PCA in the supplied normalized action space."""

    actions64 = _actions(actions).astype(np.float64)
    mean = actions64.mean(axis=0)
    centered = actions64 - mean
    denominator = max(len(actions64) - 1, 1)
    covariance = centered.T @ centered / float(denominator)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    explained = np.maximum(eigenvalues[order], 0.0)
    components = _canonical_component_signs(eigenvectors[:, order].T)
    total = float(explained.sum())
    ratio = explained / total if total > 0.0 else np.zeros_like(explained)
    return HandPCA(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        explained_variance=explained.astype(np.float32),
        explained_variance_ratio=ratio.astype(np.float32),
    )


def reconstruction_metrics(
    model: HandPCA,
    normalized_actions: np.ndarray,
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> list[dict[str, float | int]]:
    normalized = _actions(normalized_actions)
    action_mean = _vector(action_mean, "action_mean")
    action_std = _vector(action_std, "action_std")
    if np.any(action_std < 1e-6):
        raise ValueError("action_std must be >= 1e-6")
    raw = normalized * action_std + action_mean
    rows: list[dict[str, float | int]] = []
    for latent_dim in range(1, HAND_ACTION_DIM + 1):
        reconstructed = model.decode(model.encode(normalized, latent_dim))
        reconstructed_raw = reconstructed * action_std + action_mean
        normalized_error = reconstructed - normalized
        raw_error = reconstructed_raw - raw
        rows.append(
            {
                "latent_dim": latent_dim,
                "normalized_mse": float(np.mean(np.square(normalized_error))),
                "normalized_mae": float(np.mean(np.abs(normalized_error))),
                "raw_mse": float(np.mean(np.square(raw_error))),
                "raw_mae": float(np.mean(np.abs(raw_error))),
                "cumulative_explained_variance_ratio": float(
                    model.explained_variance_ratio[:latent_dim].sum()
                ),
            }
        )
    return rows


def _canonical_component_signs(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for index in range(result.shape[0]):
        pivot = int(np.argmax(np.abs(result[index])))
        if result[index, pivot] < 0.0:
            result[index] *= -1.0
    return result


def _actions(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != HAND_ACTION_DIM or not len(value):
        raise ValueError(
            f"Hand actions must have shape (N, {HAND_ACTION_DIM}), got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("Hand actions contain NaN or infinity")
    return value


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (HAND_ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite ({HAND_ACTION_DIM},)")
    return value


__all__ = [
    "HandPCA",
    "fit_hand_pca",
    "reconstruction_metrics",
    "validate_latent_dim",
]
