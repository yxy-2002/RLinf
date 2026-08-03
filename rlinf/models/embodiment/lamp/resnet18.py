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

"""Hugging Face ResNet-18 wrapper used by the Torch LAMP policies."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn


def _sanitize_import_paths() -> None:
    sys.path[:] = [
        os.fsdecode(path) if isinstance(path, (bytes, bytearray)) else path
        for path in sys.path
    ]


_sanitize_import_paths()

from transformers import ResNetConfig, ResNetModel  # noqa: E402

BackbonePooling = Literal["avg"]
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class HFResNet18Backbone(nn.Module):
    """Local Hugging Face ResNet with the LAMP image preprocessing contract.

    Images are NCHW float tensors in ``[0, 1]``.  LAMP checkpoints use global
    average pooling only, so unsupported historical pooling experiments fail
    immediately instead of silently changing the feature contract.
    """

    def __init__(
        self,
        config_dict: Mapping[str, Any],
        pooling: BackbonePooling = "avg",
        spatial_blocks: int = 8,
        spatial_bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        del spatial_blocks, spatial_bottleneck_dim
        if pooling != "avg":
            raise ValueError(
                f"LAMP only supports ResNet average pooling, got {pooling!r}"
            )
        self.pooling = pooling
        self.config_dict = _string_key_dict(dict(config_dict))
        self.resnet = ResNetModel(ResNetConfig(**_tuple_to_list(self.config_dict)))
        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        image_nchw: torch.Tensor,
        *,
        train: bool | None = None,
    ) -> torch.Tensor:
        if image_nchw.ndim != 4 or image_nchw.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB images, got {tuple(image_nchw.shape)}")
        pixels = (image_nchw.to(dtype=torch.float32) - self._mean) / self._std
        requested_mode = self.resnet.training if train is None else bool(train)
        previous_mode = self.resnet.training
        if requested_mode != previous_mode:
            self.resnet.train(requested_mode)
        try:
            pooled = self.resnet(pixel_values=pixels).pooler_output
        finally:
            if requested_mode != previous_mode:
                self.resnet.train(previous_mode)
        return pooled.flatten(1)

    @classmethod
    def from_pretrained(
        cls,
        resnet_path: str | Path,
        *,
        pooling: BackbonePooling = "avg",
    ) -> "HFResNet18Backbone":
        config_dict, state_dict, _ = load_hf_resnet18_params(resnet_path)
        model = cls(config_dict, pooling=pooling)
        model.resnet.load_state_dict(state_dict, strict=True)
        return model


def load_hf_resnet18_params(
    resnet_path: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], None]:
    """Load local HF weights for direct use by ``HFResNet18Backbone.resnet``.

    The three-item return shape is retained for migrated callers.
    The second item is now a Torch ``state_dict`` and batch statistics are part
    of that state, so the third item is always ``None``.
    """

    _sanitize_import_paths()
    path = Path(resnet_path).expanduser().resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{path} must contain config.json")
    if not any(
        (path / name).is_file()
        for name in (
            "pytorch_model.bin",
            "model.safetensors",
        )
    ):
        raise FileNotFoundError(
            f"{path} must contain model.safetensors or pytorch_model.bin"
        )
    model = ResNetModel.from_pretrained(str(path), local_files_only=True).float()
    config_dict = _string_key_dict(model.config.to_dict())
    config_dict.pop("id2label", None)
    config_dict.pop("label2id", None)
    state_dict = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    return config_dict, state_dict, None


def load_hf_resnet18_model(
    resnet_path: str | Path,
    *,
    pooling: BackbonePooling = "avg",
) -> HFResNet18Backbone:
    """Build a ready-to-run backbone from a local HF checkpoint."""

    return HFResNet18Backbone.from_pretrained(resnet_path, pooling=pooling)


def _string_key_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _string_key_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_string_key_dict(item) for item in value]
    return value


def _tuple_to_list(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _tuple_to_list(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_tuple_to_list(item) for item in value]
    if isinstance(value, list):
        return [_tuple_to_list(item) for item in value]
    return value


__all__ = [
    "BackbonePooling",
    "HFResNet18Backbone",
    "load_hf_resnet18_model",
    "load_hf_resnet18_params",
]
