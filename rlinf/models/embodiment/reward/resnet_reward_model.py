# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet-based reward model for embodied RL.

This module implements a ResNet-based reward model that uses binary
cross-entropy loss for training. It is designed for fast inference during
online RL training, similar to the HIL-SERL approach.
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class ResNetRewardModel(BaseRewardModel):
    """ResNet-based reward model using binary classification loss.

    This model uses a pretrained ResNet backbone followed by a linear head
    to output scalar rewards. It is trained using binary cross-entropy loss
    on individual images with success/fail labels.

    Training input: images [B,C,H,W], or [B,V,H,W,C] with camera_keys.
    Inference input: main_images for single-view, reward_images for multi-view.

    Attributes:
        backbone: ResNet feature extractor with modified final layer.
        arch: Architecture name (e.g., "resnet18", "resnet50").
    """

    # Supported ResNet architectures
    SUPPORTED_ARCHS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, cfg: DictConfig):
        """Initialize the ResNet reward model.

        Args:
            cfg: Configuration dictionary containing:
                - arch: ResNet architecture (default: "resnet18").
                - pretrained: Whether to use pretrained weights (default: True).
                - hidden_dim: Optional hidden dimension for MLP head.
                - dropout: Dropout rate for classification head (default: 0.1).
        """
        super().__init__(cfg)

        self.cfg = cfg
        self.image_size = cfg.get("image_size", [3, 224, 224])
        self.normalize = cfg.get("normalize", True)

        # Register normalization constants as buffers (move with model).
        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        self.camera_keys = list(cfg.camera_keys) if cfg.get("camera_keys") else None
        if self.camera_keys and len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be unique")

        self.arch = cfg.get("arch", "resnet18")
        if self.arch not in self.SUPPORTED_ARCHS:
            raise ValueError(
                f"Unsupported architecture: {self.arch}. "
                f"Supported: {self.SUPPORTED_ARCHS}"
            )

        self.pretrained = cfg.get("pretrained", True)
        self.hidden_dim = cfg.get("hidden_dim", None)
        self.dropout_rate = cfg.get("dropout", 0.1)

        # Build model architecture
        self._build_model()

        self._load_model()

        torch_dtype = torch_dtype_from_precision(cfg.precision)
        self.to(torch_dtype)

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for ResNet backbone input.

        This method accepts image tensors in ``uint8`` or floating-point format.
        ``uint8`` inputs are interpreted as ``[0, 255]`` and scaled to ``[0, 1]``.
        Floating-point inputs must already be in ``[0, 1]``.

        Args:
            images: Image tensor in ``NCHW`` or ``NHWC`` layout.

        Returns:
            A preprocessed tensor in ``NCHW`` layout, resized to
            ``self.image_size[1:]`` and optionally ImageNet-normalized.

        Raises:
            ValueError: If floating-point inputs are outside ``[0, 1]``.
            TypeError: If ``images`` is neither ``uint8`` nor floating point.
        """
        if images.dim() == 4 and images.shape[-1] in [1, 3, 4]:
            images = images.permute(0, 3, 1, 2)

        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif torch.is_floating_point(images):
            min_val = float(images.min().detach().item())
            max_val = float(images.max().detach().item())
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(
                    "ResNetRewardModel expects floating-point images in [0, 1]. "
                    f"Got min={min_val:.6f}, max={max_val:.6f}. "
                    "Please normalize upstream or pass uint8 images."
                )
        else:
            raise TypeError(
                "ResNetRewardModel expects images to be uint8 or floating point. "
                f"Got dtype={images.dtype}."
            )

        target_h, target_w = self.image_size[1], self.image_size[2]
        if images.shape[2] != target_h or images.shape[3] != target_w:
            images = F.interpolate(
                images,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        if self.normalize:
            images = (images - self._mean) / self._std

        return images

    def _build_model(self) -> None:
        """Build the ResNet backbone and reward head."""
        # Load pretrained ResNet backbone
        weights = "IMAGENET1K_V1" if self.pretrained else None
        self.backbone = getattr(models, self.arch)(weights=weights)

        # Get the number of features from the original fc layer
        num_features = self.backbone.fc.in_features

        if self.camera_keys:
            self.backbone.fc = nn.Identity()
            self.head = nn.Sequential(
                nn.Linear(num_features * len(self.camera_keys), self.hidden_dim or 256),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim or 256, 1),
            )
            self.register_buffer(
                "_preprocessing_signature",
                torch.tensor(
                    [*self.image_size, int(self.normalize)], dtype=torch.int64
                ),
            )
            # Persist view identity to reject reordered cameras at inference.
            self.register_buffer(
                "_camera_signature",
                torch.tensor(
                    list("\0".join(self.camera_keys).encode("utf-8")), dtype=torch.uint8
                ),
            )

        elif self.hidden_dim is not None:
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.backbone.fc = nn.Linear(num_features, 1)

        # Initialize weights
        self._init_head_weights()

    def _init_head_weights(self) -> None:
        """Initialize the reward head weights."""
        for module in (self.head if self.camera_keys else self.backbone.fc).modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _load_model(self):
        model_path = self.cfg.get("model_path")
        if model_path is not None:
            self.load_from_path(model_path)

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass for training with binary classification loss.

        Args:
            input_data: Image tensor [B,C,H,W] or multi-view [B,V,H,W,C].
            labels: Binary labels (B,) where 1=success, 0=fail.

        Returns:
            Dictionary containing:
                - "loss": Binary cross entropy loss (scalar tensor).
                - "accuracy": Classification accuracy.
                - "logits": Raw model outputs (B,).
                - "probabilities": Sigmoid probabilities (B,).
        """
        logits = self._image_logits(input_data)

        # Compute probabilities
        probabilities = torch.sigmoid(logits)

        # Compute loss if labels provided
        if labels is not None:
            labels = labels.float().to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Compute accuracy
            predictions = (probabilities > 0.5).float()
            accuracy = (predictions == labels).float().mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)
            accuracy = torch.tensor(0.0, device=logits.device)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "logits": logits,
            "probabilities": probabilities,
        }

    def compute_reward(self, observations: dict[str, Any]) -> torch.Tensor:
        """Compute rewards for inference.

        Args:
            observations: Batched ``main_images`` or multi-view ``reward_images``.

        Returns:
            torch.Tensor: Reward tensor of shape [B].
        """
        key = "reward_images" if self.camera_keys else "main_images"
        images = observations.get(key)
        if images is None:
            raise ValueError(f"Missing {key} in reward observations")
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        with torch.no_grad():
            rewards = torch.sigmoid(self._image_logits(images))

        # Optional thresholding: keep consistent with prior worker behavior.
        threshold = self.cfg.get("reward_threshold", None)
        if threshold is not None:
            thr = float(threshold)
            rewards = torch.where(rewards > thr, rewards, torch.zeros_like(rewards))

        return rewards

    def _validate_camera_signature(self, state_dict: dict) -> None:
        if self.camera_keys and not torch.equal(
            state_dict.get("_camera_signature", torch.empty(0)).cpu(),
            self._camera_signature.cpu(),
        ):
            raise ValueError("Checkpoint camera order does not match camera_keys")

        if self.camera_keys and not torch.equal(
            state_dict.get("_preprocessing_signature", torch.empty(0)).cpu(),
            self._preprocessing_signature.cpu(),
        ):
            raise ValueError(
                "Checkpoint image_size/normalize do not match model configuration"
            )

    def _image_logits(self, images: torch.Tensor) -> torch.Tensor:
        parameter = next(self.parameters())
        images = images.to(parameter.device)
        if self.camera_keys:
            if images.ndim != 5 or images.shape[1] != len(self.camera_keys):
                raise ValueError("Expected images [B,V,H,W,C] or [B,V,C,H,W]")
            batch, views = images.shape[:2]
            images = images.flatten(0, 1)
        images = self.preprocess_images(images).to(parameter.dtype)
        features = self.backbone(images)
        if self.camera_keys:
            features = self.head(features.reshape(batch, views * features.shape[-1]))
        return features.squeeze(-1)

    def load_from_path(self, model_path: str) -> None:
        """Load a ResNet reward checkpoint from a file path."""
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["module.", "_orig_mod.", "model."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            # Skip mean/std buffers (they are persistent=False, auto-created)
            if new_key in ["mean", "std", "_mean", "_std"]:
                continue
            new_state_dict[new_key] = v
        self._validate_camera_signature(new_state_dict)
        self.load_state_dict(new_state_dict, strict=True)
