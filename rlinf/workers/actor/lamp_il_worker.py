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

import hashlib
import json
import time
from collections.abc import Iterator, Mapping
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
from rlinf.models.embodiment.lamp.bc_policy import BCPolicy
from rlinf.models.embodiment.lamp.bimanual_diffusion_policy import (
    LAMPBimanualDiffusionPolicy,
)
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import LAMPDiffusionPolicy
from rlinf.models.embodiment.lamp.hand_pca import fit_hand_pca
from rlinf.models.embodiment.lamp.policy_wrapper import LampPolicy, LampPolicySpec
from rlinf.models.embodiment.lamp.hand_prior_artifact import (
    TorchHandPCA,
    build_prior_model,
    load_prior_artifact,
    sorted_vq_codebook,
)
from rlinf.models.embodiment.lamp.resnet18 import load_hf_resnet18_params
from rlinf.models.embodiment.lamp.il_training_utils import (
    CosineSchedule,
    beta_warmup,
    configure_torch_runtime,
)
from rlinf.models.embodiment.lamp.hand_vq_vae import HandVQVAE
from rlinf.scheduler import Worker


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
    """Train one LAMP phase-two stage on one RLinf-managed GPU."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.stage = str(cfg.algorithm.stage)
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
        self._statistics = load_cache_statistics(cache_dir)
        self._output_dir = Path(
            self.cfg.runner.logger.log_path
        ).expanduser().resolve() / str(self.cfg.runner.logger.experiment_name)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self.stage == "prior":
            self._setup_prior()
        elif self.stage == "bc":
            self._setup_bc()
        elif self.stage == "dp":
            self._setup_dp()
        else:
            raise ValueError(f"Unsupported LAMP training stage {self.stage!r}")
        self._setup_optimizer()
        self._setup_dataloaders(cache_dir)
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

    def _setup_prior(self) -> None:
        prior_cfg = self.cfg.actor.model.hand_prior
        prior_type = str(prior_cfg.type)
        hand_side = str(prior_cfg.get("hand_side", "single"))
        embodiment = self._cache_metadata["embodiment"]
        valid_sides = ("single",) if embodiment == "single" else ("right", "left")
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

    def _setup_bc(self) -> None:
        if self._cache_metadata["embodiment"] != "single":
            raise ValueError("LAMP BC currently supports the single-arm policy only")
        model_cfg = self.cfg.actor.model
        prior_cfg = model_cfg.hand_prior
        source = str(prior_cfg.type)
        backbone_config, backbone_state, _ = load_hf_resnet18_params(
            model_cfg.resnet_path
        )
        vae_config = None
        prior_model = None
        if source == "vae":
            prior_model, prior_metadata, _ = load_prior_artifact(
                prior_cfg.artifact_path,
                expected_type="vae",
                expected_task=self._cache_metadata["task"],
                expected_dataset_fingerprint=self._cache_metadata["fingerprint"],
                expected_hand_side="single",
            )
            if int(prior_cfg.latent_dim) != int(prior_metadata["latent_dim"]):
                raise ValueError(
                    "Configured LAMP BC latent_dim differs from the VAE artifact"
                )
            vae_config = dict(prior_metadata["architecture"])
        elif source != "mlp":
            raise ValueError("LAMP BC hand prior must be 'vae' or 'mlp'")
        architecture = {
            "backbone_config": backbone_config,
            "hand_prior_source": source,
            "vae_model_config": vae_config,
            "hidden_dims": [512, 512, 256],
            "state_hidden_dims": [128, 128],
            "hand_state_window_size": 8,
            "backbone_pooling": "avg",
            "dense_init": "torch_uniform",
        }
        model = BCPolicy(**architecture)
        model.front_backbone.resnet.load_state_dict(backbone_state, strict=True)
        model.wrist_backbone.resnet.load_state_dict(backbone_state, strict=True)
        if prior_model is not None:
            model.vae.load_state_dict(prior_model.state_dict(), strict=True)
        self.model = model.to(self.device)
        self._architecture = architecture
        latent_dim = 0 if source == "mlp" else int(model.vae.latent_dim)
        self._policy_spec = LampPolicySpec(
            task=self._cache_metadata["task"],
            policy_family="bc",
            embodiment="single",
            hand_prior_type=source,
            action_horizon=1,
            execution_horizon=1,
            core_action_dim=7 + (16 if source == "mlp" else latent_dim),
            physical_action_dim=23,
            image_size=int(self._cache_metadata["image_size"]),
            image_keys=tuple(self._cache_metadata["image_keys"]),
            latent_dims={"single": latent_dim},
        )
        self._artifact_metadata = self._policy_artifact_metadata("lamp_bc")

    def _setup_dp(self) -> None:
        model_cfg = self.cfg.actor.model
        source = str(model_cfg.hand_prior.type)
        policy_source = "vq_codebook" if source == "vq" else source
        if policy_source not in ("cvae", "decoder_only", "pca", "vq_codebook", "mlp"):
            raise ValueError(f"Unsupported LAMP DP hand prior {source!r}")
        backbone_config, backbone_state, _ = load_hf_resnet18_params(
            model_cfg.resnet_path
        )
        embodiment = self._cache_metadata["embodiment"]
        priors = self._load_dp_priors(source, embodiment)
        core_stats, namespace = self._prepare_dp_targets(source, priors)
        self._derived_namespace = namespace
        if embodiment == "single":
            architecture = _single_dp_architecture(
                backbone_config,
                policy_source,
                priors.get("single"),
                core_stats,
                self._statistics,
                model_cfg.hand_prior,
            )
            model: nn.Module = LAMPDiffusionPolicy(**architecture)
            model.front_backbone.resnet.load_state_dict(backbone_state, strict=True)
            model.wrist_backbone.resnet.load_state_dict(backbone_state, strict=True)
            if policy_source in ("cvae", "decoder_only"):
                model.cvae.load_state_dict(
                    priors["single"][0].state_dict(), strict=True
                )
            latent_dims = {"single": model._hand_latent_dim()}
            core_dim = model._core_dim()
            physical_dim = 23
        else:
            architecture = _bimanual_dp_architecture(
                backbone_config,
                policy_source,
                priors,
                core_stats,
                self._statistics,
                model_cfg.hand_prior,
            )
            model = LAMPBimanualDiffusionPolicy(**architecture)
            for backbone in (
                model.ego_backbone,
                model.right_wrist_backbone,
                model.left_wrist_backbone,
            ):
                backbone.resnet.load_state_dict(backbone_state, strict=True)
            if policy_source in ("cvae", "decoder_only"):
                model.right_cvae.load_state_dict(
                    priors["right"][0].state_dict(), strict=True
                )
                model.left_cvae.load_state_dict(
                    priors["left"][0].state_dict(), strict=True
                )
            latent_dims = {side: model._latent_dim(side) for side in ("right", "left")}
            core_dim = model._core_dim()
            physical_dim = 46
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
        )
        self._artifact_metadata = self._policy_artifact_metadata("lamp_dp")

    def _load_dp_priors(self, source: str, embodiment: str):
        if source == "mlp":
            return {}
        prior_cfg = self.cfg.actor.model.hand_prior
        sides = ("single",) if embodiment == "single" else ("right", "left")
        result = {}
        for side in sides:
            side_cfg = prior_cfg if side == "single" else prior_cfg[side]
            expected_type = "cvae" if source == "decoder_only" else source
            payload = load_prior_artifact(
                side_cfg.artifact_path,
                expected_type=expected_type,
                expected_task=self._cache_metadata["task"],
                expected_dataset_fingerprint=self._cache_metadata["fingerprint"],
                expected_hand_side=side,
                device=self.device,
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
        sides = (
            ("single",)
            if self._cache_metadata["embodiment"] == "single"
            else (
                "right",
                "left",
            )
        )
        chunks = []
        for side in sides:
            prefix = "" if side == "single" else f"{side}_"
            target = np.asarray(
                np.load(split_dir / f"{prefix}target_action23.npy", mmap_mode="r")
            )
            future = np.asarray(
                np.load(split_dir / f"{prefix}future_hand_norm.npy", mmap_mode="r")
            )
            if source == "mlp":
                chunks.append(target)
                continue
            model, metadata, statistics = priors[side]
            if source == "pca":
                latent_dim = int(metadata["latent_dim"])
                latent = (future - model.mean.cpu().numpy()) @ model.components[
                    :latent_dim
                ].cpu().numpy().T
            elif source == "vq":
                codebook = np.asarray(statistics["sorted_codebook"], np.float32)
                distances = np.square(future[..., None, :] - codebook).sum(axis=-1)
                index = np.argmin(distances, axis=-1).astype(np.float32)
                latent = (2.0 * index / 15.0 - 1.0)[..., None]
            else:
                history = np.asarray(
                    np.load(split_dir / f"{prefix}hand_history_norm.npy", mmap_mode="r")
                )
                latent_parts = []
                model.eval()
                with torch.inference_mode():
                    for start in range(0, len(history), 512):
                        end = min(start + 512, len(history))
                        mu, _ = model.encode_posterior(
                            torch.from_numpy(history[start:end]).to(self.device),
                            torch.from_numpy(future[start:end]).to(self.device),
                            torch.from_numpy(mask[start:end]).to(self.device),
                        )
                        latent_parts.append(mu.cpu().numpy().astype(np.float32))
                latent = np.concatenate(latent_parts)
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
            "right_wrist_backbone.",
            "left_wrist_backbone.",
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
            int(self.cfg.runner.max_steps),
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
            if prior_type == "vae":
                return [f"{prefix}hand_history_norm", f"{prefix}hand_target_norm"]
            if prior_type == "cvae":
                return [
                    f"{prefix}hand_history_norm",
                    f"{prefix}future_hand_norm",
                    "mask",
                ]
            return [f"{prefix}hand_target_norm"]
        if self.stage == "bc":
            return [
                "front",
                "wrist",
                "arm_state_norm",
                "hand_history_norm",
                "target_action23",
                "arm_target_norm",
            ]
        core = f"derived:{self._derived_namespace}:core_norm"
        if self._cache_metadata["embodiment"] == "single":
            return [
                "front",
                "wrist",
                "arm_state_norm",
                "hand_history_norm",
                "target_action23",
                "mask",
                core,
            ]
        return [
            "ego",
            "right_wrist",
            "left_wrist",
            "right_arm_state_norm",
            "left_arm_state_norm",
            "right_hand_history_norm",
            "left_hand_history_norm",
            "right_target_action23",
            "left_target_action23",
            "mask",
            core,
        ]

    def _loss(
        self, batch: dict[str, torch.Tensor], step: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.stage == "prior":
            return self._prior_loss(batch, step)
        if self.stage == "bc":
            return self._bc_loss(batch)
        return self._dp_loss(batch)

    def _prior_loss(self, batch, step):
        prior_cfg = self.cfg.actor.model.hand_prior
        side = str(prior_cfg.get("hand_side", "single"))
        prefix = "" if side == "single" else f"{side}_"
        prior_type = str(prior_cfg.type)
        if prior_type == "vae":
            history = batch[f"{prefix}hand_history_norm"]
            target = batch[f"{prefix}hand_target_norm"]
            beta = beta_warmup(
                step, float(prior_cfg.beta), int(prior_cfg.beta_warmup_steps)
            )
            output = self.model(history, target, beta=beta)
            return {
                "total_loss": output.total_loss,
                "reconstruction_loss": output.reconstruction_loss,
                "kl_loss": output.kl_loss,
                "beta": torch.as_tensor(beta, device=self.device),
                "latent_std": output.mu.std(),
            }
        if prior_type == "cvae":
            output = self.model(
                batch[f"{prefix}hand_history_norm"],
                batch[f"{prefix}future_hand_norm"],
                target_mask=batch["mask"],
            )
            return {
                "total_loss": output.total_loss,
                "reconstruction_loss": output.reconstruction_loss,
                "posterior_kl_loss": output.posterior_kl_loss,
                "prior_kl_loss": output.prior_kl_loss,
                "weighted_kl_loss": output.weighted_kl_loss,
                "latent_std": output.mu_q.std(),
            }
        output = self.model(
            batch[f"{prefix}hand_target_norm"], training=True, update_ema=False
        )
        return output

    def _bc_loss(self, batch):
        output = self.model(
            batch["front"],
            batch["wrist"],
            batch["arm_state_norm"],
            batch["hand_history_norm"],
            train=self.model.training,
            return_aux=True,
        )
        target = batch["target_action23"][:, 0]
        arm_mean = _tensor_stat(self._statistics, "arm_action_mean", self.device)
        arm_std = _tensor_stat(self._statistics, "arm_action_std", self.device)
        hand_mean = _tensor_stat(self._statistics, "hand_action_mean", self.device)
        hand_std = _tensor_stat(self._statistics, "hand_action_std", self.device)
        pred_arm_raw = output["arm_action"] * arm_std + arm_mean
        pred_hand_raw = output["hand_action"] * hand_std + hand_mean
        no_corr_raw = output["hand_no_corr"] * hand_std + hand_mean
        xyz_loss = (
            (output["arm_action"][:, :3] - batch["arm_target_norm"][:, :3])
            .square()
            .mean()
        )
        pred_quat = pred_arm_raw[:, 3:7]
        target_quat = target[:, 3:7]
        pred_quat = pred_quat / torch.linalg.vector_norm(
            pred_quat, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        target_quat = target_quat / torch.linalg.vector_norm(
            target_quat, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        quat_loss = (1.0 - (pred_quat * target_quat).sum(dim=-1).square()).mean()
        weights = self.cfg.algorithm.bc_loss
        arm_loss = (
            float(weights.arm_xyz) * xyz_loss + float(weights.quaternion) * quat_loss
        )
        hand_loss = (pred_hand_raw - target[:, 7:]).square().mean()
        drift_loss = (pred_hand_raw - no_corr_raw).square().mean()
        total = (
            float(weights.arm) * arm_loss
            + float(weights.hand) * hand_loss
            + float(weights.drift) * drift_loss
        )
        no_corr = (no_corr_raw - target[:, 7:]).square().mean()
        return {
            "total_loss": total,
            "arm_loss": arm_loss,
            "arm_xyz_loss": xyz_loss,
            "arm_quat_loss": quat_loss,
            "hand_loss": hand_loss,
            "drift_loss": drift_loss,
            "arm_raw_mse": (pred_arm_raw - target[:, :7]).square().mean(),
            "hand_no_corr_loss": no_corr,
            "vision_gain": no_corr - hand_loss,
            "hand_mae": (pred_hand_raw - target[:, 7:]).abs().mean(),
        }

    def _dp_loss(self, batch):
        batch_size = batch["core_norm"].shape[0]
        timesteps = torch.randint(0, 100, (batch_size,), device=self.device)
        noise = torch.randn_like(batch["core_norm"])
        if self._cache_metadata["embodiment"] == "single":
            return self.model.compute_loss(
                batch["front"],
                batch["wrist"],
                batch["arm_state_norm"],
                batch["hand_history_norm"],
                batch["core_norm"],
                batch["target_action23"],
                batch["mask"],
                timesteps=timesteps,
                noise=noise,
                train=self.model.training,
            )
        return self.model.compute_loss(
            batch["ego"],
            batch["right_wrist"],
            batch["left_wrist"],
            batch["right_arm_state_norm"],
            batch["left_arm_state_norm"],
            batch["right_hand_history_norm"],
            batch["left_hand_history_norm"],
            batch["core_norm"],
            batch["right_target_action23"],
            batch["left_target_action23"],
            batch["mask"],
            timesteps=timesteps,
            noise=noise,
            train=self.model.training,
        )

    def run_training(self) -> dict[str, float | int]:
        max_steps = int(self.cfg.runner.max_steps)
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
            grad_norm = nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ],
                float(self.cfg.actor.optim.clip_grad),
            )
            self.optimizer.step()
            if isinstance(self.model, HandVQVAE):
                self.model.quantizer.apply_ema_updates(ema_counts, ema_sums)
            update_seconds += time.perf_counter() - started
            self._global_step += 1
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
        if validation_interval > 0 and self._global_step % validation_interval < count:
            metrics.update(
                {f"validation/{key}": value for key, value in self._validate().items()}
            )
        metrics["__global_step"] = self._global_step
        return metrics

    def _prepare_batch(self, batch: dict[str, torch.Tensor]):
        result = {}
        for name, value in batch.items():
            tensor = value.to(self.device, non_blocking=True)
            if name in ("front", "wrist", "ego", "right_wrist", "left_wrist"):
                tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            else:
                tensor = tensor.float()
            result[name] = tensor.contiguous()
        return result

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        samples = 0
        max_batches = int(self.cfg.actor.get("validation_batches", -1))
        for batch_index, raw in enumerate(self._validation_loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = self._prepare_batch(raw)
            outputs = self._loss(
                batch, torch.tensor(self._global_step, device=self.device)
            )
            batch_size = next(iter(batch.values())).shape[0]
            samples += batch_size
            for name, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        self.model.train(True)
        return (
            {name: value / samples for name, value in totals.items()} if samples else {}
        )

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
        )
        self._save_deployment_artifact(output / "artifact")
        self._save_deployment_artifact(self._output_dir / "artifact")

    def load_checkpoint(self, load_base_path: str) -> None:
        step, sampler = load_training_state(
            load_base_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.schedule,
            expected_metadata=self._resume_metadata(),
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
        if self.stage == "prior_pca":
            pass
        elif metadata.get("prior_type") == "vq":
            statistics["sorted_codebook"] = sorted_vq_codebook(self.model)
        if metadata.get("kind") == "policy":
            wrapper_stats = _wrapper_statistics(
                self._statistics, self._policy_spec.embodiment
            )
            export_model: nn.Module = LampPolicy(
                self.model, self._policy_spec, wrapper_stats
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
            "model_type": model_type,
            "task": self._cache_metadata["task"],
            "dataset_fingerprint": self._cache_metadata["fingerprint"],
            "architecture": self._architecture,
            "spec": None if self._policy_spec is None else vars(self._policy_spec),
        }

    def _resume_metadata(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "cache_fingerprint": self._cache_metadata["fingerprint"],
            "architecture_sha256": metadata_sha256(self._architecture),
            "artifact": self._artifact_metadata,
            "max_steps": int(self.cfg.runner.max_steps),
            "global_batch_size": int(self.cfg.actor.global_batch_size),
            "micro_batch_size": int(self.cfg.actor.micro_batch_size),
            "accumulation_steps": self._accumulation_steps,
        }


def _prior_architecture(prior_type: str, cfg: DictConfig) -> dict[str, Any]:
    latent_dim = int(cfg.latent_dim)
    if prior_type == "vae":
        return {
            "backbone": "cnn",
            "hidden_dim": int(cfg.get("hidden_dim", 512)),
            "beta": float(cfg.get("beta", 1e-4)),
            "latent_dim": latent_dim,
        }
    if prior_type == "cvae":
        return {
            "hidden_dim": int(cfg.get("hidden_dim", 1024)),
            "posterior_kl_weight": float(cfg.get("posterior_kl_weight", 1e-4)),
            "prior_kl_weight": float(cfg.get("prior_kl_weight", 1e-4)),
            "latent_dim": latent_dim,
        }
    if prior_type == "vq":
        return {
            "action_dim": 16,
            "latent_dim": int(cfg.get("code_latent_dim", 256)),
            "hidden_dim": int(cfg.get("hidden_dim", 512)),
            "num_quantizers": 2,
            "codebook_size": 4,
            "layer_num": 5,
            "commitment_weight": 1.0,
            "ema_decay": 0.8,
            "epsilon": 1e-5,
            "dead_code_threshold": 0.0,
            "reconstruction_multiplier": 3.0,
            "vq_multiplier": 5.0,
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
    if source in ("cvae", "decoder_only"):
        kwargs["cvae_model_config"] = dict(prior[1]["architecture"])
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


def _bimanual_dp_architecture(
    backbone_config, source, priors, core_stats, statistics, prior_cfg
):
    kwargs: dict[str, Any] = {
        "backbone_config": backbone_config,
        "hand_prior_source": source,
        "condition_hidden_dims": [1024, 512],
        "condition_dim": 512,
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
        "right_hand_action_mean": statistics["right_hand_action_mean"].tolist(),
        "right_hand_action_std": statistics["right_hand_action_std"].tolist(),
        "left_hand_action_mean": statistics["left_hand_action_mean"].tolist(),
        "left_hand_action_std": statistics["left_hand_action_std"].tolist(),
    }
    if source in ("cvae", "decoder_only"):
        kwargs["right_cvae_model_config"] = dict(priors["right"][1]["architecture"])
        kwargs["left_cvae_model_config"] = dict(priors["left"][1]["architecture"])
    elif source == "pca":
        for side in ("right", "left"):
            latent_dim = int(prior_cfg[side].latent_dim)
            kwargs[f"{side}_pca_latent_dim"] = latent_dim
            kwargs[f"{side}_pca_mean"] = priors[side][0].mean.cpu().tolist()
            kwargs[f"{side}_pca_components"] = (
                priors[side][0].components[:latent_dim].cpu().tolist()
            )
    elif source == "vq_codebook":
        for side in ("right", "left"):
            kwargs[f"{side}_vq_codebook"] = priors[side][2]["sorted_codebook"].tolist()
    return kwargs


def _wrapper_statistics(statistics: Mapping[str, np.ndarray], embodiment: str):
    if embodiment == "single":
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
    else:
        names = tuple(
            f"{side}_{field}_{suffix}"
            for side in ("right", "left")
            for field in ("arm_state", "hand_history", "hand_action")
            for suffix in ("mean", "std")
        )
    return {name: np.asarray(statistics[name]) for name in names}


def _tensor_stat(values: Mapping[str, np.ndarray], name: str, device):
    return torch.as_tensor(values[name], dtype=torch.float32, device=device)


def _side_key(side: str, name: str) -> str:
    return name if side == "single" else f"{side}_{name}"


__all__ = ["DeterministicInfiniteBatchSampler", "LampILWorker"]
