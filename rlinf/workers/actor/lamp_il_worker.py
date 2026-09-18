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

"""Single-GPU RLinf worker for LAMP prior, BC, and DP training."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Sampler

from rlinf.data.datasets.lamp import (
    LampMMapDataset,
    load_cache_metadata,
    load_cache_statistics,
)
from rlinf.models.embodiment.lamp.artifact_io import (
    load_training_state,
    metadata_sha256,
    save_artifact,
    save_training_state,
)
from rlinf.models.embodiment.lamp.hand_pca import fit_hand_pca
from rlinf.models.embodiment.lamp.hand_prior_artifact import (
    TorchHandPCA,
    build_prior_model,
    load_prior_artifact,
    sorted_vq_codebook,
)
from rlinf.models.embodiment.lamp.hand_vq_vae import HandVQVAE
from rlinf.models.embodiment.lamp.il_training_utils import (
    CosineSchedule,
    beta_warmup,
    configure_torch_runtime,
    episode_ids,
)
from rlinf.models.embodiment.lamp.policy_wrapper import LampPolicy, LampPolicySpec
from rlinf.models.embodiment.lamp.resnet18 import load_hf_resnet18_params
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import LAMPDiffusionPolicy
from rlinf.models.embodiment.lamp.vq_action_normalization import (
    VQ_HAND_ACTION_NORMALIZATION,
    normalize_vq_hand_action,
    vq_hand_action_bounds,
)
from rlinf.scheduler import Worker
from rlinf.utils.runner_utils import resolve_training_horizon


def _clip_gradients(
    parameters: Iterable[nn.Parameter], max_norm: float | None
) -> torch.Tensor:
    """Clip gradients when configured and otherwise report their global norm."""
    with_grad = [
        parameter
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    if max_norm is not None:
        max_norm = float(max_norm)
        if max_norm < 0.0:
            raise ValueError(f"clip_grad must be non-negative or null, got {max_norm}")
        return nn.utils.clip_grad_norm_(with_grad, max_norm)
    if not with_grad:
        return torch.zeros((), dtype=torch.float32)
    per_parameter = [
        torch.linalg.vector_norm(parameter.grad.detach().float(), ord=2)
        for parameter in with_grad
    ]
    return torch.linalg.vector_norm(torch.stack(per_parameter), ord=2)


def _nearest_vq_indices(
    physical_hand: np.ndarray, physical_codebook: np.ndarray
) -> np.ndarray:
    """Match physical hand targets to the physical DQ-RISE codebook."""

    hand = np.asarray(physical_hand, dtype=np.float32)
    codebook = np.asarray(physical_codebook, dtype=np.float32)
    if hand.shape[-1] != 16 or codebook.shape != (16, 16):
        raise ValueError(
            "VQ nearest-neighbor inputs must be [...,16] and [16,16], "
            f"got {hand.shape} and {codebook.shape}"
        )
    distances = np.square(hand[..., None, :] - codebook).sum(axis=-1)
    return np.argmin(distances, axis=-1)


def _training_contract(
    cfg: DictConfig, *, steps_per_epoch: int, max_steps: int
) -> dict[str, Any]:
    """Return the resolved numerical training recipe stored with artifacts."""
    return {
        "seed": int(cfg.actor.seed),
        "max_epochs": int(cfg.runner.max_epochs),
        "max_steps": int(max_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "global_batch_size": int(cfg.actor.global_batch_size),
        "micro_batch_size": int(cfg.actor.micro_batch_size),
        "eval_batch_size": int(cfg.actor.eval_batch_size),
        "optimizer": OmegaConf.to_container(cfg.actor.optim, resolve=True),
        "model": OmegaConf.to_container(cfg.actor.model, resolve=True),
        "torch_compile": bool(cfg.actor.get("torch_compile", True)),
        "compile_mode": str(cfg.actor.get("compile_mode", "default")),
        "validation_interval": int(cfg.actor.get("validation_interval", 0)),
        "validate_at_end": bool(cfg.actor.get("validate_at_end", False)),
        "validation_batches": int(cfg.actor.get("validation_batches", -1)),
    }


class DeterministicInfiniteBatchSampler(Sampler[list[int]]):
    """Recompute each shuffled epoch from seed and the consumed batch count."""

    def __init__(
        self, size: int, batch_size: int, seed: int, start_batch: int = 0
    ) -> None:
        if size < batch_size or batch_size < 1:
            raise ValueError("LAMP requires dataset size >= batch size >= 1")
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        self.batches_per_epoch = self.size // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        batch_index = self.start_batch
        while True:
            epoch, within = divmod(batch_index, self.batches_per_epoch)
            order = np.random.default_rng(self.seed + epoch).permutation(self.size)
            while within < self.batches_per_epoch:
                start = within * self.batch_size
                yield order[start : start + self.batch_size].tolist()
                batch_index += 1
                within += 1

    def __len__(self) -> int:
        return 2**31 - 1


class LampILWorker(Worker):
    """Train one LAMP prior or DP stage on one RLinf-managed GPU."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.stage = str(cfg.algorithm.stage)
        self._steps_per_epoch, self._max_steps = resolve_training_horizon(cfg.runner)
        self._global_step = 0
        self.device = torch.device("cpu")
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.schedule: CosineSchedule | None = None
        self._compiled_loss = None
        self._train_loader: DataLoader | None = None
        self._train_iter = None
        self._validation_loader: DataLoader | None = None
        self._accumulation_steps = 1
        self._architecture: dict[str, Any] = {}
        self._artifact_metadata: dict[str, Any] = {}
        self._policy_spec: LampPolicySpec | None = None
        self._cache_metadata: dict[str, Any] = {}
        self._statistics: dict[str, np.ndarray] = {}
        self._derived_namespace: str | None = None
        self._ema_model: nn.Module | None = None
        self._ema_settings: dict[str, Any] = {}
        self._output_dir = Path(".")

    def init_worker(self) -> None:
        if self._world_size != 1:
            raise ValueError(
                "LAMP phase-two training intentionally uses exactly one actor GPU per job"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("LAMP phase-two training requires CUDA")
        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        configure_torch_runtime()
        seed = int(self.cfg.actor.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cache_dir = Path(self.cfg.data.cache_path).expanduser().resolve()
        self._cache_metadata = load_cache_metadata(cache_dir)
        if self._cache_metadata["embodiment"] != "single":
            raise ValueError("LAMP supports single-arm datasets only")
        self._statistics = load_cache_statistics(cache_dir)
        self._output_dir = Path(
            self.cfg.runner.logger.log_path
        ).expanduser().resolve() / str(self.cfg.runner.logger.experiment_name)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self.stage == "prior":
            self._setup_prior()
            self._initialize_weights()
        elif self.stage == "dp":
            self._setup_dp()
            self._initialize_weights()
            self._setup_ema()
        else:
            raise ValueError(f"Unsupported LAMP training stage {self.stage!r}")
        self._setup_dataloaders(cache_dir)
        prior_type = str(self.cfg.actor.model.hand_prior.type)
        if self.stage == "dp" or (
            self.stage == "prior" and prior_type in {"lamplstm", "vq"}
        ):
            self._artifact_metadata["training_contract"] = _training_contract(
                self.cfg,
                steps_per_epoch=self._steps_per_epoch,
                max_steps=self._max_steps,
            )
        self._setup_optimizer()
        if (
            bool(self.cfg.actor.get("torch_compile", True))
            and self.stage != "prior_pca"
        ):
            mode = str(self.cfg.actor.get("compile_mode", "max-autotune-no-cudagraphs"))
            self._compiled_loss = torch.compile(
                self._loss, mode=mode, fullgraph=False, dynamic=False
            )
            self.log_info(f"LAMP torch.compile enabled with mode={mode!r}")
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.log_info(
            f"LAMP stage={self.stage}, parameters={parameters:,}, "
            f"trainable={trainable:,}, cache={cache_dir}"
        )

    def _initialize_weights(self) -> None:
        """Initialize a new run from deployment weights, never optimizer state."""
        path = self.cfg.actor.model.get("model_path")
        if not path:
            return
        from rlinf.models.embodiment.lamp.artifact_io import load_artifact

        metadata, state, _ = load_artifact(path)
        if metadata.get("dataset_fingerprint") != self._cache_metadata["fingerprint"]:
            raise ValueError("Initialization artifact uses different data/statistics")
        if metadata.get("architecture") != self._architecture:
            raise ValueError(
                "Initialization artifact architecture differs from this run"
            )
        if self.stage == "dp":
            if metadata.get("policy_version") != 2 or metadata.get(
                "model_type"
            ) not in ("lamp_dp", "lamp_dp_v2"):
                raise ValueError(
                    "DP initialization requires version-2 deployment weights"
                )
            if (
                metadata.get("spec", {}).get("hand_prior_type")
                != self._policy_spec.hand_prior_type
            ):
                raise ValueError("Initialization artifact prior differs from this run")
            state = {
                key.removeprefix("core."): value
                for key, value in state.items()
                if key.startswith("core.")
            }
        elif metadata.get("prior_type") != str(self.cfg.actor.model.hand_prior.type):
            raise ValueError("Initialization artifact prior differs from this run")
        self.model.load_state_dict(state, strict=True)

    def _setup_prior(self) -> None:
        prior_cfg = self.cfg.actor.model.hand_prior
        prior_type = str(prior_cfg.type)
        hand_side = str(prior_cfg.get("hand_side", "single"))
        embodiment = self._cache_metadata["embodiment"]
        valid_sides = ("single",)
        if hand_side not in valid_sides:
            raise ValueError(
                f"LAMP {embodiment} prior hand_side must be one of {valid_sides}"
            )
        architecture = _prior_architecture(prior_type, prior_cfg)
        if prior_type == "pca":
            key = _side_key(hand_side, "hand_target_norm")
            values = np.asarray(
                np.load(
                    Path(self.cfg.data.cache_path) / "train" / f"{key}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
            )
            pca = fit_hand_pca(values)
            self.model = TorchHandPCA.from_pca(pca).to(self.device)
            self.stage = "prior_pca"
            architecture = {
                "mean": pca.mean.tolist(),
                "components": pca.components.tolist(),
                "explained_variance": pca.explained_variance.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
            }
        else:
            self.model = build_prior_model(prior_type, architecture).to(self.device)
        self._architecture = architecture
        self._artifact_metadata = {
            "kind": "prior",
            "prior_type": prior_type,
            "task": self._cache_metadata["task"],
            "dataset_fingerprint": self._cache_metadata["fingerprint"],
            "hand_side": hand_side,
            "latent_dim": int(prior_cfg.get("latent_dim", 0)),
            "architecture": architecture,
        }
        if prior_type == "lamplstm":
            self._artifact_metadata.update(
                {
                    "history_length": int(prior_cfg.get("history_length", 16)),
                    "horizon": int(architecture["horizon"]),
                    "action_dim": int(architecture["action_dim"]),
                    "normalization": "cache_statistics:hand_action",
                    "condition_dropout_scope": "shared_per_sample",
                    "history_contract": self._cache_metadata["history_contract"],
                    "encoder_condition_mode": str(
                        architecture.get("condition_mode_encoder", "none")
                    ),
                    "decoder_condition_mode": str(
                        architecture.get("condition_mode_decoder", "none")
                    ),
                    "encoder_history_length": int(prior_cfg.get("history_length", 16)),
                    "decoder_history_length": int(prior_cfg.get("history_length", 16)),
                }
            )
        if prior_type == "vq":
            self._artifact_metadata["hand_action_normalization"] = (
                VQ_HAND_ACTION_NORMALIZATION
            )

    def _setup_dp(self) -> None:
        model_cfg = self.cfg.actor.model
        source = str(model_cfg.hand_prior.type)
        policy_source = "vq_codebook" if source == "vq" else source
        if policy_source not in (
            "lamplstm",
            "pca",
            "vq_codebook",
            "mlp",
        ):
            raise ValueError(f"Unsupported LAMP DP hand prior {source!r}")
        backbone_config, backbone_state, _ = load_hf_resnet18_params(
            model_cfg.resnet_path
        )
        embodiment = self._cache_metadata["embodiment"]
        priors = self._load_dp_priors(source, embodiment)
        core_stats, namespace = self._prepare_dp_targets(source, priors)
        self._derived_namespace = namespace
        architecture = _single_dp_architecture(
            backbone_config,
            policy_source,
            priors.get("single"),
            core_stats,
            self._statistics,
            model_cfg.hand_prior,
        )

        architecture["dropout_prob"] = float(model_cfg.get("dropout_prob", 0.0))
        model: nn.Module = LAMPDiffusionPolicy(**architecture)
        model.front_backbone.resnet.load_state_dict(backbone_state, strict=True)
        model.wrist_backbone.resnet.load_state_dict(backbone_state, strict=True)
        if policy_source == "lamplstm":
            model.lamplstm.load_state_dict(
                priors["single"][0].state_dict(), strict=True
            )
        latent_dims = {"single": model._hand_latent_dim()}
        core_dim = model._core_dim()
        physical_dim = 23
        self.model = model.to(self.device)
        self._architecture = architecture
        self._policy_spec = LampPolicySpec(
            task=self._cache_metadata["task"],
            policy_family="dp",
            embodiment=embodiment,
            hand_prior_type=policy_source,
            action_horizon=16,
            execution_horizon=int(model_cfg.get("execution_horizon", 4)),
            core_action_dim=core_dim,
            physical_action_dim=physical_dim,
            image_size=int(self._cache_metadata["image_size"]),
            image_keys=tuple(self._cache_metadata["image_keys"]),
            latent_dims=latent_dims,
            policy_version=2,
        )
        self._artifact_metadata = self._policy_artifact_metadata("lamp_dp")
        self._artifact_metadata["policy_version"] = 2

    def _setup_ema(self) -> None:
        """Create a frozen EMA twin for deployment-weight experiments."""

        ema_cfg = self.cfg.actor.model.get("ema", {})
        enabled = bool(ema_cfg.get("enabled", False))
        decay = float(ema_cfg.get("decay", 0.999))
        start_step = int(ema_cfg.get("start_step", 1000))
        if not enabled:
            return
        if self.stage != "dp":
            raise ValueError("LAMP model EMA is supported only for dp")
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if start_step < 0 or start_step >= self._max_steps:
            raise ValueError("EMA start_step must be within the training horizon")
        self._ema_settings = {
            "enabled": True,
            "decay": decay,
            "start_step": start_step,
        }
        self._ema_model = copy.deepcopy(self.model).to(self.device)
        self._ema_model.eval()
        for parameter in self._ema_model.parameters():
            parameter.requires_grad_(False)
        self._artifact_metadata["ema"] = dict(self._ema_settings)

    def _update_ema(self) -> None:
        if self._ema_model is None:
            return
        step = self._global_step
        start_step = int(self._ema_settings["start_step"])
        decay = float(self._ema_settings["decay"])
        if step == start_step:
            self._ema_model.load_state_dict(self.model.state_dict())
        if step < start_step:
            return
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                self._ema_model.parameters(), self.model.parameters()
            ):
                ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    def _load_dp_priors(self, source: str, embodiment: str):
        if source == "mlp":
            return {}
        prior_cfg = self.cfg.actor.model.hand_prior
        sides = ("single",)
        result = {}
        for side in sides:
            side_cfg = prior_cfg if side == "single" else prior_cfg[side]
            expected_type = source
            payload = load_prior_artifact(
                side_cfg.artifact_path,
                expected_type=expected_type,
                expected_task=self._cache_metadata["task"],
                expected_dataset_fingerprint=self._cache_metadata["fingerprint"],
                expected_hand_side=side,
                device=self.device,
            )
            if (
                source == "vq"
                and payload[1].get("hand_action_normalization")
                != VQ_HAND_ACTION_NORMALIZATION
            ):
                raise ValueError(
                    "LAMP VQ prior was not trained with the fixed Allegro "
                    "ctrlrange normalization"
                )
            if source == "vq":
                expected_low, expected_high = vq_hand_action_bounds()
                stored_low = payload[2].get("vq_hand_action_low")
                stored_high = payload[2].get("vq_hand_action_high")
                if not (
                    stored_low is not None
                    and stored_high is not None
                    and np.array_equal(stored_low, expected_low)
                    and np.array_equal(stored_high, expected_high)
                ):
                    raise ValueError(
                        "LAMP VQ prior does not contain the expected fixed "
                        "Allegro action bounds"
                    )
            configured_dim = side_cfg.get("latent_dim", None)
            artifact_dim = payload[1].get("latent_dim", None)
            if configured_dim is not None and int(configured_dim) != int(artifact_dim):
                raise ValueError(
                    f"Configured LAMP {side} latent_dim differs from the prior artifact"
                )
            result[side] = payload
        return result

    def _prepare_dp_targets(self, source: str, priors: dict[str, Any]):
        binding = {
            side: payload[1]["metadata_sha256"] for side, payload in priors.items()
        }
        namespace_payload = {
            "schema": 1,
            "policy_version": 2,
            "source": source,
            "prior_binding": binding,
            "latent_dims": OmegaConf.to_container(
                self.cfg.actor.model.hand_prior, resolve=True
            ),
        }
        namespace = hashlib.sha256(
            json.dumps(namespace_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        root = Path(self.cfg.data.cache_path) / "derived" / namespace
        metadata_path = root / "metadata.json"
        if metadata_path.is_file():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("binding") != namespace_payload:
                raise ValueError("LAMP derived DP cache binding mismatch")
            return {
                "mean": np.asarray(payload["core_action_mean"], np.float32),
                "std": np.asarray(payload["core_action_std"], np.float32),
            }, namespace
        root.mkdir(parents=True, exist_ok=True)
        raw_by_split = {}
        masks = {}
        for split in ("train", "validation"):
            raw, mask = self._dp_core_raw(split, source, priors)
            raw_by_split[split] = raw
            masks[split] = mask
        selected = raw_by_split["train"][masks["train"] > 0]
        mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(
            selected.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
        )
        for split in ("train", "validation"):
            output = (raw_by_split[split] - mean) / std
            split_dir = root / split
            split_dir.mkdir(exist_ok=True)
            np.save(split_dir / "core_norm.npy", output.astype(np.float32))
        payload = {
            "binding": namespace_payload,
            "core_action_mean": mean.tolist(),
            "core_action_std": std.tolist(),
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {"mean": mean, "std": std}, namespace

    def _dp_core_raw(self, split: str, source: str, priors: dict[str, Any]):
        split_dir = Path(self.cfg.data.cache_path) / split
        mask = np.asarray(np.load(split_dir / "mask.npy", mmap_mode="r"))
        sides = ("single",)
        chunks = []
        for side in sides:
            prefix = "" if side == "single" else f"{side}_"
            target = np.asarray(
                np.load(split_dir / f"{prefix}target_action23.npy", mmap_mode="r")
            )
            if source == "mlp":
                chunks.append(target)
                continue
            model, metadata, statistics = priors[side]
            if source == "pca":
                future = np.asarray(
                    np.load(split_dir / f"{prefix}future_hand_norm.npy", mmap_mode="r")
                )
                latent_dim = int(metadata["latent_dim"])
                latent = (future - model.mean.cpu().numpy()) @ model.components[
                    :latent_dim
                ].cpu().numpy().T
            elif source == "vq":
                codebook = np.asarray(statistics["sorted_codebook"], np.float32)
                physical_hand = target[..., 7:]
                index = _nearest_vq_indices(physical_hand, codebook).astype(np.float32)
                latent = (2.0 * index / 15.0 - 1.0)[..., None]
            elif source == "lamplstm":
                future = np.asarray(
                    np.load(split_dir / f"{prefix}future_hand_norm.npy", mmap_mode="r")
                )
                history = np.asarray(
                    np.load(
                        split_dir / f"{prefix}lamplstm_encoder_history_norm.npy",
                        mmap_mode="r",
                    )
                )
                history_mask = np.asarray(
                    np.load(
                        split_dir / f"{prefix}lamplstm_encoder_history_mask.npy",
                        mmap_mode="r",
                    )
                )
                encoder_mode = str(
                    metadata.get("encoder_condition_mode", model.condition_mode_encoder)
                )
                latent_parts = []
                model.eval()
                with torch.inference_mode():
                    for start in range(0, len(history), 512):
                        end = min(start + 512, len(history))
                        use_history = encoder_mode != "none"
                        mu, _ = model.encoder.encode(
                            torch.from_numpy(history[start:end]).to(self.device)
                            if use_history
                            else None,
                            torch.from_numpy(future[start:end]).to(self.device),
                            torch.from_numpy(history_mask[start:end]).to(self.device)
                            if use_history
                            else None,
                        )
                        latent_parts.append(mu.cpu().numpy().astype(np.float32))
                latent = np.concatenate(latent_parts)
            else:
                raise ValueError(f"Unsupported DP prior {source!r}")
            chunks.extend((target[..., :7], latent.astype(np.float32)))
        return np.concatenate(chunks, axis=-1).astype(np.float32), mask

    def _setup_optimizer(self) -> None:
        if self.stage == "prior_pca":
            return
        optim_cfg = self.cfg.actor.optim
        backbone_prefixes = (
            "front_backbone.",
            "wrist_backbone.",
            "ego_backbone.",
        )
        backbone, other = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (backbone if name.startswith(backbone_prefixes) else other).append(
                parameter
            )
        if not other:
            raise ValueError("LAMP optimizer found no trainable parameters")
        lr = float(optim_cfg.lr)
        groups: list[dict[str, Any]] = [{"params": other, "lr": lr, "lr_ratio": 1.0}]
        if backbone:
            ratio = float(optim_cfg.get("backbone_lr_ratio", 0.1))
            groups.append({"params": backbone, "lr": lr * ratio, "lr_ratio": ratio})
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=lr,
            betas=(float(optim_cfg.adam_beta1), float(optim_cfg.adam_beta2)),
            eps=float(optim_cfg.adam_eps),
            weight_decay=float(optim_cfg.weight_decay),
        )
        self.schedule = CosineSchedule(
            lr,
            self._max_steps,
            int(optim_cfg.warmup_steps),
            float(optim_cfg.min_lr),
        )

    def _setup_dataloaders(self, cache_dir: Path) -> None:
        if self.stage == "prior_pca":
            return
        keys = self._dataset_keys()
        train = LampMMapDataset(cache_dir, "train", keys)
        validation = LampMMapDataset(cache_dir, "validation", keys)
        global_batch_size = int(self.cfg.actor.global_batch_size)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        if (
            micro_batch_size < 1
            or global_batch_size < micro_batch_size
            or global_batch_size % micro_batch_size != 0
        ):
            raise ValueError(
                "LAMP global_batch_size must be a positive multiple of micro_batch_size"
            )
        self._accumulation_steps = global_batch_size // micro_batch_size
        needs_epoch_size = (
            "steps_per_epoch" in self.cfg.runner
            or int(self.cfg.runner.max_steps) < 0
            or "save_every_epochs" in self.cfg.runner
        )
        if needs_epoch_size:
            actual_steps_per_epoch = len(train) // global_batch_size
            self._steps_per_epoch, self._max_steps = resolve_training_horizon(
                self.cfg.runner, steps_per_epoch=actual_steps_per_epoch
            )
        sampler = DeterministicInfiniteBatchSampler(
            len(train),
            micro_batch_size,
            int(self.cfg.actor.seed),
            self._global_step * self._accumulation_steps,
        )
        num_workers = int(self.cfg.data.get("num_workers", 4))
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(self.cfg.data.get("pin_memory", True)),
            "persistent_workers": num_workers > 0
            and bool(self.cfg.data.get("persistent_workers", True)),
        }
        self._train_loader = DataLoader(train, batch_sampler=sampler, **loader_kwargs)
        self._train_iter = iter(self._train_loader)
        self._validation_loader = DataLoader(
            validation,
            batch_size=int(self.cfg.actor.get("eval_batch_size", micro_batch_size)),
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    def _dataset_keys(self) -> list[str]:
        hand_side = str(self.cfg.actor.model.hand_prior.get("hand_side", "single"))
        if self.stage == "prior":
            prefix = "" if hand_side == "single" else f"{hand_side}_"
            prior_type = str(self.cfg.actor.model.hand_prior.type)
            if prior_type == "lamplstm":
                history_length = int(
                    self.cfg.actor.model.hand_prior.get("history_length", 16)
                )
                return [
                    f"{prefix}hand_history{history_length}_norm",
                    f"{prefix}hand_history{history_length}_mask",
                    f"{prefix}future_hand_norm",
                    "mask",
                ]
            if prior_type == "vq":
                return [f"{prefix}target_action23"]
            return [f"{prefix}hand_target_norm"]
        core = f"derived:{self._derived_namespace}:core_norm"
        keys = [
            "front",
            "wrist",
            "arm_state_pair_norm",
            "hand_state_pair_norm",
            "target_action23",
            "mask",
            core,
        ]
        if str(self.cfg.actor.model.hand_prior.type) == "lamplstm":
            keys.extend(
                (
                    "lamplstm_decoder_history_norm",
                    "lamplstm_decoder_history_mask",
                )
            )
        return keys

    def _loss(
        self, batch: dict[str, torch.Tensor], step: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.stage == "prior":
            return self._prior_loss(batch, step)
        return self._dp_loss(batch)

    def _prior_loss(self, batch, step):
        prior_cfg = self.cfg.actor.model.hand_prior
        side = str(prior_cfg.get("hand_side", "single"))
        prefix = "" if side == "single" else f"{side}_"
        prior_type = str(prior_cfg.type)
        if prior_type == "lamplstm":
            beta = beta_warmup(
                step,
                float(prior_cfg.get("beta", 5e-4)),
                int(prior_cfg.get("beta_warmup_steps", 0)),
            )
            output = self.model(
                batch[
                    f"{prefix}hand_history{int(prior_cfg.get('history_length', 16))}_norm"
                ],
                batch[f"{prefix}future_hand_norm"],
                history_mask=batch[
                    f"{prefix}hand_history{int(prior_cfg.get('history_length', 16))}_mask"
                ],
                future_mask=batch["mask"],
                beta=beta,
                sample=self.model.training,
            )
            return {
                "total_loss": output.total_loss,
                "reconstruction_loss": output.reconstruction_loss,
                "kl_loss": output.kl_loss,
                "beta": torch.as_tensor(beta, device=self.device),
                "latent_std": output.mu.std(),
            }
        if prior_type == "vq":
            physical_hand = batch[f"{prefix}target_action23"][:, 0, 7:]
            target = normalize_vq_hand_action(physical_hand)
        else:
            target = batch[f"{prefix}hand_target_norm"]
        output = self.model(target, training=self.model.training, update_ema=False)
        return output

    def _dp_loss(self, batch):
        batch_size = batch["core_norm"].shape[0]
        timesteps = torch.randint(0, 100, (batch_size,), device=self.device)
        noise = torch.randn_like(batch["core_norm"])
        decoder_history = decoder_history_mask = None
        if str(self.cfg.actor.model.hand_prior.type) == "lamplstm":
            decoder_history = batch["lamplstm_decoder_history_norm"]
            decoder_history_mask = batch["lamplstm_decoder_history_mask"]
        return self.model.compute_loss(
            batch["front"],
            batch["wrist"],
            batch["arm_state_pair_norm"],
            batch["hand_state_pair_norm"],
            batch["core_norm"],
            batch["target_action23"],
            batch["mask"],
            timesteps=timesteps,
            noise=noise,
            decoder_history=decoder_history,
            decoder_history_mask=decoder_history_mask,
            train=self.model.training,
        )

    def run_training(self) -> dict[str, float | int]:
        max_steps = self._max_steps
        if self.stage == "prior_pca":
            self._global_step = max_steps
            metrics = self._pca_metrics()
            metrics["__global_step"] = self._global_step
            return metrics
        count = min(
            int(self.cfg.runner.local_update_steps), max_steps - self._global_step
        )
        totals: dict[str, float] = {}
        data_seconds = 0.0
        update_seconds = 0.0
        self.model.train(True)
        for _ in range(count):
            self._set_learning_rate(self._global_step)
            self.optimizer.zero_grad(set_to_none=True)
            ema_counts = None
            ema_sums = None
            for _micro_step in range(self._accumulation_steps):
                started = time.perf_counter()
                batch = self._prepare_batch(next(self._train_iter))
                data_seconds += time.perf_counter() - started
                started = time.perf_counter()
                loss_fn = self._compiled_loss or self._loss
                outputs = loss_fn(
                    batch, torch.tensor(self._global_step, device=self.device)
                )
                (outputs["total_loss"] / self._accumulation_steps).backward()
                update_seconds += time.perf_counter() - started
                if isinstance(self.model, HandVQVAE):
                    counts = outputs["ema_counts"].detach()
                    sums = outputs["ema_sums"].detach()
                    ema_counts = counts if ema_counts is None else ema_counts + counts
                    ema_sums = sums if ema_sums is None else ema_sums + sums
                for name, value in outputs.items():
                    if name in (
                        "ema_counts",
                        "ema_sums",
                        "indices",
                        "distances",
                        "reconstruction",
                    ):
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        totals[name] = (
                            totals.get(name, 0.0)
                            + float(value.detach()) / self._accumulation_steps
                        )
            started = time.perf_counter()
            grad_norm = _clip_gradients(
                self.model.parameters(), self.cfg.actor.optim.get("clip_grad")
            )
            self.optimizer.step()
            if isinstance(self.model, HandVQVAE):
                self.model.quantizer.apply_ema_updates(ema_counts, ema_sums)
            update_seconds += time.perf_counter() - started
            self._global_step += 1
            self._update_ema()
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm)
        metrics = {name: value / count for name, value in totals.items()}
        elapsed = max(data_seconds + update_seconds, 1e-9)
        metrics.update(
            {
                "lr": self._current_lr(),
                "data/samples_per_second": count
                * int(self.cfg.actor.global_batch_size)
                / elapsed,
                "data/seconds_per_step": data_seconds / count,
                "time/update_seconds_per_step": update_seconds / count,
                "gpu_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            }
        )
        validation_interval = int(self.cfg.actor.get("validation_interval", 0))
        validate_at_end = bool(self.cfg.actor.get("validate_at_end", False))
        should_validate = validate_at_end and self._global_step == max_steps
        should_validate |= (
            not validate_at_end
            and validation_interval > 0
            and self._global_step % validation_interval < count
        )
        if should_validate:
            metrics.update(
                {f"validation/{key}": value for key, value in self._validate().items()}
            )
        metrics["__global_step"] = self._global_step
        return metrics

    def _prepare_batch(self, batch: dict[str, torch.Tensor]):
        result = {}
        for name, value in batch.items():
            tensor = value.to(self.device, non_blocking=True)
            if name in ("front", "wrist"):
                tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            else:
                tensor = tensor.float()
            result[name] = tensor.contiguous()
        return result

    @torch.no_grad()
    def _ddim_validation_metrics(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Compute deployment-aligned action errors after full DDIM sampling."""
        if self.stage != "dp":
            return {}

        if str(self.cfg.actor.model.hand_prior.type) == "lamplstm":
            self.model.set_decoder_history(
                batch["lamplstm_decoder_history_norm"],
                batch["lamplstm_decoder_history_mask"],
            )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(20260916)
        predicted_action = self.model(
            batch["front"],
            batch["wrist"],
            batch["arm_state_pair_norm"],
            batch["hand_state_pair_norm"],
            generator=generator,
            train=False,
        )
        action_mse = (predicted_action - batch["target_action23"]).square()
        target_mask = batch["mask"].bool()
        valid_tokens = target_mask.sum().clamp_min(1)
        valid_elements = valid_tokens * action_mse.shape[-1]
        execution_horizon = int(self.cfg.actor.model.get("execution_horizon", 8))
        execution_mask = target_mask[:, :execution_horizon]
        execution_valid_tokens = execution_mask.sum().clamp_min(1)
        execution_valid_elements = execution_valid_tokens * action_mse.shape[-1]
        arm_dim = 7
        hand_dim = action_mse.shape[-1] - arm_dim
        return {
            "ddim_action_mse": float(action_mse[target_mask].sum() / valid_elements),
            "ddim_first_execution_horizon_mse": float(
                action_mse[:, :execution_horizon][execution_mask].sum()
                / execution_valid_elements
            ),
            "ddim_arm_mse": float(
                action_mse[..., :arm_dim][target_mask].sum() / (valid_tokens * arm_dim)
            ),
            "ddim_hand_mse": float(
                action_mse[..., arm_dim:][target_mask].sum() / (valid_tokens * hand_dim)
            ),
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        ddim_totals: dict[str, float] = {}
        ddim_token_totals: dict[str, int] = {}
        samples = 0
        max_batches = int(self.cfg.actor.get("validation_batches", -1))
        for batch_index, raw in enumerate(self._validation_loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = self._prepare_batch(raw)
            outputs = self._loss(
                batch, torch.tensor(self._global_step, device=self.device)
            )
            ddim_metrics = self._ddim_validation_metrics(batch)
            if ddim_metrics:
                execution_horizon = int(
                    self.cfg.actor.model.get("execution_horizon", 8)
                )
                execution_tokens = int(
                    batch["mask"][:, :execution_horizon].sum().item()
                )
                all_tokens = int(batch["mask"].sum().item())
                for name, value in ddim_metrics.items():
                    token_count = (
                        execution_tokens
                        if name == "ddim_first_execution_horizon_mse"
                        else all_tokens
                    )
                    ddim_totals[name] = ddim_totals.get(name, 0.0) + value * token_count
                    ddim_token_totals[name] = (
                        ddim_token_totals.get(name, 0) + token_count
                    )
            batch_size = next(iter(batch.values())).shape[0]
            samples += batch_size
            for name, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        result = (
            {name: value / samples for name, value in totals.items()} if samples else {}
        )
        if ddim_token_totals:
            result.update(
                {
                    name: value / ddim_token_totals[name]
                    for name, value in ddim_totals.items()
                }
            )
        if (
            self.stage == "prior"
            and self._artifact_metadata.get("history_contract") == "primitive_v1"
        ):
            result.update(self._validate_lstm_conditions())
        self.model.train(True)
        return result

    @torch.no_grad()
    def _validate_lstm_conditions(self) -> dict[str, float]:
        """Report paired on/off validation without consuming training RNG state."""
        directory = Path(self.cfg.data.cache_path) / "validation"
        length = int(self.cfg.actor.model.hand_prior.history_length)
        h = np.load(directory / f"hand_history{length}_norm.npy", mmap_mode="r")
        hm = np.load(directory / f"hand_history{length}_mask.npy", mmap_mode="r")
        future = np.load(directory / "future_hand_norm.npy", mmap_mode="r")
        fm = np.load(directory / "mask.npy", mmap_mode="r")
        probability = float(self.cfg.actor.model.hand_prior.condition_drop_prob)
        result = {}
        beta = float(self.cfg.actor.model.hand_prior.beta)

        def loss(rows, off, sample):
            history, target, mask, valid = [
                torch.tensor(a[rows], device=self.device) for a in (h, future, hm, fm)
            ]
            output = self.model(
                history,
                target,
                history_mask=torch.zeros_like(mask) if off else mask,
                future_mask=valid,
                beta=beta,
                sample=sample,
            )
            return float(output.total_loss)

        for off in (False, True):
            total = 0.0
            for start in range(0, len(h), 512):
                rows = np.arange(start, min(start + 512, len(h)))
                total += loss(rows, off, False) * len(rows)
            result["condition_off_loss" if off else "condition_on_loss"] = total / len(
                h
            )
        result["condition_matched_loss"] = (1 - probability) * result[
            "condition_on_loss"
        ] + probability * result["condition_off_loss"]
        if self._global_step % 5000 == 0:
            eid = episode_ids(hm, fm)
            groups = [np.flatnonzero(eid == e) for e in np.unique(eid)]
            rng = np.random.default_rng(20260909)
            rows = np.array([rng.choice(groups[i % len(groups)]) for i in range(256)])
            devices = [self.device.index] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                for off in (False, True):
                    torch.manual_seed(123)
                    result[
                        "condition_off_sample_loss"
                        if off
                        else "condition_on_sample_loss"
                    ] = sum(loss(rows, off, True) for _ in range(4)) / 4
            result["condition_matched_sample_loss"] = (1 - probability) * result[
                "condition_on_sample_loss"
            ] + probability * result["condition_off_sample_loss"]
        return result

    def _pca_metrics(self) -> dict[str, float]:
        side = str(self.cfg.actor.model.hand_prior.get("hand_side", "single"))
        key = _side_key(side, "hand_target_norm")
        values = np.array(
            np.load(
                Path(self.cfg.data.cache_path) / "validation" / f"{key}.npy",
                mmap_mode="r",
            ),
            copy=True,
        )
        pca = self.model
        latent_dim = int(self.cfg.actor.model.hand_prior.latent_dim)
        latent = pca.encode(torch.from_numpy(values).to(self.device), latent_dim)
        reconstruction = pca.decode(latent).cpu().numpy()
        error = reconstruction - values
        return {
            "total_loss": float(np.mean(np.square(error))),
            "reconstruction_mse": float(np.mean(np.square(error))),
            "reconstruction_mae": float(np.mean(np.abs(error))),
            "explained_variance_ratio": float(
                pca.explained_variance_ratio[:latent_dim].sum().cpu()
            ),
        }

    def _set_learning_rate(self, step: int) -> None:
        base = float(self.schedule(step))
        for group in self.optimizer.param_groups:
            group["lr"] = base * float(group.get("lr_ratio", 1.0))

    def _current_lr(self) -> float:
        return (
            0.0
            if self.optimizer is None
            else float(self.optimizer.param_groups[0]["lr"])
        )

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        output = Path(save_base_path).expanduser().resolve()
        save_training_state(
            output,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.schedule,
            global_step=self._global_step,
            sampler_state={
                "consumed_batches": self._global_step * self._accumulation_steps,
                "accumulation_steps": self._accumulation_steps,
            },
            metadata=self._resume_metadata(),
            ema_model=self._ema_model,
        )
        self._save_deployment_artifact(output / "artifact")
        self._save_deployment_artifact(self._output_dir / "artifact")

    def export_deployment_artifacts(self, checkpoint_base_path: str) -> None:
        """Re-export deployment artifacts from the currently loaded checkpoint."""

        checkpoint = Path(checkpoint_base_path).expanduser().resolve()
        self._save_deployment_artifact(checkpoint / "artifact")
        self._save_deployment_artifact(self._output_dir / "artifact")

    def load_checkpoint(self, load_base_path: str) -> None:
        step, sampler = load_training_state(
            load_base_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.schedule,
            expected_metadata=self._resume_metadata(),
            ema_model=self._ema_model,
        )
        checkpoint_name = Path(load_base_path).expanduser().resolve().parent.name
        if not checkpoint_name.startswith("global_step_"):
            raise ValueError("LAMP checkpoint must live below global_step_<N>/actor")
        directory_step = int(checkpoint_name.removeprefix("global_step_"))
        if directory_step != step:
            raise ValueError("LAMP checkpoint directory and saved global step differ")
        expected_sampler = {
            "consumed_batches": step * self._accumulation_steps,
            "accumulation_steps": self._accumulation_steps,
        }
        if sampler != expected_sampler:
            raise ValueError("LAMP checkpoint sampler state is inconsistent")
        self._global_step = step
        self._setup_dataloaders(Path(self.cfg.data.cache_path))

    def _save_deployment_artifact(self, output: Path) -> None:
        statistics = dict(self._statistics)
        metadata = dict(self._artifact_metadata)
        metadata["global_step"] = int(self._global_step)
        if self.stage == "prior_pca":
            pass
        elif metadata.get("prior_type") == "vq":
            statistics["sorted_codebook"] = sorted_vq_codebook(self.model)
            low, high = vq_hand_action_bounds()
            statistics["vq_hand_action_low"] = low
            statistics["vq_hand_action_high"] = high
        if metadata.get("kind") == "policy":
            wrapper_stats = _wrapper_statistics(
                self._statistics, self._policy_spec.embodiment
            )
            source_model = (
                self._ema_model if self._ema_model is not None else self.model
            )
            metadata["export_weights"] = "ema" if self._ema_model is not None else "raw"
            export_model: nn.Module = LampPolicy(
                source_model, self._policy_spec, wrapper_stats
            )
            statistics = wrapper_stats
        else:
            export_model = self.model
        metadata["statistics_keys"] = sorted(statistics)
        save_artifact(
            output, model=export_model, metadata=metadata, statistics=statistics
        )

    def _policy_artifact_metadata(self, model_type: str) -> dict[str, Any]:
        return {
            "kind": "policy",
            "policy_version": 2,
            "vq_quantization": "nearest_half_up",
            "model_type": model_type,
            "task": self._cache_metadata["task"],
            "dataset_fingerprint": self._cache_metadata["fingerprint"],
            "architecture": self._architecture,
            "spec": None if self._policy_spec is None else vars(self._policy_spec),
        }

    def _resume_metadata(self) -> dict[str, Any]:
        metadata = {
            "training_schema_version": 2,
            "stage": self.stage,
            "cache_fingerprint": self._cache_metadata["fingerprint"],
            "architecture_sha256": metadata_sha256(self._architecture),
            "artifact": self._artifact_metadata,
            "max_steps": self._max_steps,
            "global_batch_size": int(self.cfg.actor.global_batch_size),
            "micro_batch_size": int(self.cfg.actor.micro_batch_size),
            "accumulation_steps": self._accumulation_steps,
        }
        if (
            "steps_per_epoch" in self.cfg.runner
            or int(self.cfg.runner.max_steps) < 0
            or "save_every_epochs" in self.cfg.runner
        ):
            metadata["steps_per_epoch"] = self._steps_per_epoch
        return metadata


def _prior_architecture(prior_type: str, cfg: DictConfig) -> dict[str, Any]:
    latent_dim = int(cfg.latent_dim)
    if prior_type == "lamplstm":
        expected = {
            "action_dim": int(cfg.get("action_dim", 16)),
            "history_dim": int(cfg.get("history_dim", 16)),
            "history_length": int(cfg.get("history_length", 16)),
            "horizon": int(cfg.get("horizon", 16)),
        }
        if any(
            expected[name] != 16 for name in ("action_dim", "history_dim", "horizon")
        ):
            raise ValueError(
                "LAMP-LSTM prior requires action_dim=history_dim=horizon=16"
            )
        if expected["history_length"] < 1:
            raise ValueError("LAMP-LSTM history_length must be >= 1")
        return {
            "action_dim": expected["action_dim"],
            "history_dim": expected["history_dim"],
            "horizon": expected["horizon"],
            "latent_dim": latent_dim,
            "action_hidden_dim": int(cfg.get("action_hidden_dim", 256)),
            "condition_hidden_dim": int(cfg.get("condition_hidden_dim", 256)),
            "condition_mode_encoder": str(cfg.get("encoder_condition_mode", "film")),
            "condition_mode_decoder": str(cfg.get("decoder_condition_mode", "film")),
            "num_lstm_layers": int(cfg.get("num_lstm_layers", 1)),
            "beta": float(cfg.get("beta", 5e-4)),
            "condition_drop_prob": float(cfg.get("condition_drop_prob", 0.2)),
        }
    if prior_type == "vq":
        return {
            "action_dim": int(cfg.get("action_dim", 16)),
            "latent_dim": int(cfg.get("code_latent_dim", 256)),
            "hidden_dim": int(cfg.get("hidden_dim", 512)),
            "num_quantizers": int(cfg.get("num_quantizers", 2)),
            "codebook_size": int(cfg.get("codebook_size", 4)),
            "layer_num": int(cfg.get("layer_num", 5)),
            "commitment_weight": float(cfg.get("commitment_weight", 1.0)),
            "ema_decay": float(cfg.get("ema_decay", 0.8)),
            "epsilon": float(cfg.get("epsilon", 1e-5)),
            "dead_code_threshold": float(cfg.get("dead_code_threshold", 0.0)),
            "reconstruction_multiplier": float(
                cfg.get("reconstruction_multiplier", 3.0)
            ),
            "vq_multiplier": float(cfg.get("vq_multiplier", 5.0)),
        }
    if prior_type == "pca":
        if not 1 <= latent_dim <= 16:
            raise ValueError("PCA latent_dim must be in [1,16]")
        return {}
    raise ValueError(f"Unsupported prior type {prior_type!r}")


def _single_dp_architecture(
    backbone_config, source, prior, core_stats, statistics, prior_cfg
):
    kwargs: dict[str, Any] = {
        "backbone_config": backbone_config,
        "hand_prior_source": source,
        "condition_hidden_dims": [512, 256],
        "condition_dim": 256,
        "state_hidden_dims": [128, 128],
        "hand_state_window_size": 8,
        "backbone_pooling": "avg",
        "action_horizon": 16,
        "diffusion_step_embed_dim": 256,
        "down_dims": [128, 256, 512],
        "kernel_size": 5,
        "n_groups": 8,
        "num_train_timesteps": 100,
        "num_inference_steps": 16,
        "core_action_mean": core_stats["mean"].tolist(),
        "core_action_std": core_stats["std"].tolist(),
        "hand_action_mean": statistics["hand_action_mean"].tolist(),
        "hand_action_std": statistics["hand_action_std"].tolist(),
    }
    if source == "lamplstm":
        kwargs["lamplstm_model_config"] = dict(prior[1]["architecture"])
        if prior[1].get("history_contract") != "primitive_v1":
            raise ValueError(
                "LAMP prior artifact requires history_contract=primitive_v1"
            )
        kwargs["decoder_history_contract"] = "primitive_v1"
        kwargs["decoder_history_length"] = int(prior[1]["history_length"])
    elif source == "pca":
        latent_dim = int(prior_cfg.latent_dim)
        kwargs.update(
            pca_latent_dim=latent_dim,
            pca_mean=prior[0].mean.cpu().tolist(),
            pca_components=prior[0].components[:latent_dim].cpu().tolist(),
        )
    elif source == "vq_codebook":
        kwargs["vq_codebook"] = prior[2]["sorted_codebook"].tolist()
    return kwargs


def _wrapper_statistics(statistics: Mapping[str, np.ndarray], embodiment: str):
    names = (
        "arm_state_mean",
        "arm_state_std",
        "hand_history_mean",
        "hand_history_std",
        "arm_action_mean",
        "arm_action_std",
        "hand_action_mean",
        "hand_action_std",
    )
    result = {name: np.asarray(statistics[name]) for name in names}
    result["arm_state_pair_mean"] = result["arm_state_mean"]
    result["arm_state_pair_std"] = result["arm_state_std"]
    result["hand_state_pair_mean"] = result["hand_history_mean"]
    result["hand_state_pair_std"] = result["hand_history_std"]
    return result


def _tensor_stat(values: Mapping[str, np.ndarray], name: str, device):
    return torch.as_tensor(values[name], dtype=torch.float32, device=device)


def _side_key(side: str, name: str) -> str:
    return name if side == "single" else f"{side}_{name}"


__all__ = ["DeterministicInfiniteBatchSampler", "LampILWorker"]
