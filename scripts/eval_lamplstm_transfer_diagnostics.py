# Copyright 2026 The RLinf Authors.
"""Second-stage diagnostics: fixed visual probe and existing DP inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from rlinf.models.embodiment.lamp.lamplstm_prior import (
    LampLSTMPrior,
)
from rlinf.models.embodiment.lamp.resnet18 import HFResNet18Backbone
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
    LAMPDiffusionPolicy,
)
from scripts.eval_lamplstm_offline_quality import episode_ids, masked_mse, ridge_predict


def image_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    """Convert cached NHWC uint8 images to the native backbone contract."""
    return (
        torch.tensor(np.asarray(array), device=device).permute(0, 3, 1, 2).float() / 255
    )


def build_features(cache: Path, output: Path, device: str) -> None:
    """Cache a fixed pretrained ImageNet visual probe, with no DP training."""
    model = (
        HFResNet18Backbone.from_pretrained("pretrained_models/resnet-18")
        .to(device)
        .eval()
    )
    result = {}
    with torch.inference_mode():
        for split in ("train", "validation"):
            arrays = {
                k: np.load(cache / split / (k + ".npy"), mmap_mode="r")
                for k in (
                    "front",
                    "wrist",
                    "arm_state_pair_norm",
                    "hand_state_pair_norm",
                )
            }
            n = len(arrays["front"])
            idx = (
                np.linspace(0, n - 1, min(4096, n), dtype=int)
                if split == "train"
                else np.arange(n)
            )
            chunks = []
            for start in range(0, len(idx), 128):
                batch = idx[start : start + 128]
                features = [
                    model(image_tensor(arrays[k][batch], device), train=False)
                    .cpu()
                    .numpy()
                    for k in ("front", "wrist")
                ]
                features.extend(
                    arrays[k][batch].reshape(len(batch), -1)
                    for k in ("arm_state_pair_norm", "hand_state_pair_norm")
                )
                chunks.append(np.concatenate(features, -1))
            result[split] = np.concatenate(chunks)
            result[split + "_idx"] = idx
    np.savez_compressed(output, **result)
    print("features", str(output), flush=True)


def evaluate(root: Path, output: Path, features_path: Path, device: str) -> None:
    """Add pretrained visual-probe error and deployment history diagnostics."""
    prior = root / "prior/run/artifact"
    meta = json.loads((prior / "artifact.json").read_text())
    cache = root / "cache/water_plant" / meta["dataset_fingerprint"]
    st = dict(np.load(prior / "statistics.npz"))
    arrays = {
        s: {
            k: np.load(cache / s / (k + ".npy"), mmap_mode="r")
            for k in (
                "future_hand_norm",
                "mask",
                "lamplstm_encoder_history_norm",
                "lamplstm_encoder_history_mask",
            )
        }
        for s in ("train", "validation")
    }
    tr, va = arrays["train"], arrays["validation"]
    features = dict(np.load(features_path))
    model = LampLSTMPrior(**meta["architecture"]).to(device).eval()
    model.load_state_dict(load_file(str(prior / "model.safetensors")), strict=True)
    trainidx = features["train_idx"]
    h = np.asarray(va["lamplstm_encoder_history_norm"])
    hm = np.asarray(va["lamplstm_encoder_history_mask"])
    f = np.asarray(va["future_hand_norm"])
    fm = np.asarray(va["mask"])
    eid = episode_ids(hm, fm)
    starts = np.r_[0, np.flatnonzero(np.diff(eid)) + 1]
    ix = np.maximum(
        np.arange(len(h))[:, None] - 8 * np.arange(15, -1, -1), starts[eid, None]
    )
    deploy = h[ix, -1]
    deploy_mask = np.ones(deploy.shape[:2], np.float32)
    per = {}

    def decode(z, hist=h, mask=hm):
        chunks = []
        for i in range(0, len(z), 512):
            chunks.append(
                model.decode(
                    torch.tensor(z[i : i + 512], device=device),
                    torch.tensor(hist[i : i + 512], device=device),
                    torch.tensor(mask[i : i + 512], device=device),
                )
                .cpu()
                .numpy()
            )
        return np.concatenate(chunks)

    def encode(data, idx):
        chunks = []
        for i in range(0, len(idx), 512):
            ii = idx[i : i + 512]
            chunks.append(
                model.encode(
                    torch.tensor(
                        data["lamplstm_encoder_history_norm"][ii], device=device
                    ),
                    torch.tensor(data["future_hand_norm"][ii], device=device),
                    torch.tensor(
                        data["lamplstm_encoder_history_mask"][ii], device=device
                    ),
                )[0]
                .cpu()
                .numpy()
            )
        return np.concatenate(chunks)

    def measure(name, pred):
        for k in (8, 16):
            per[name + f"_mse{k}"] = masked_mse(pred[:, :k], f[:, :k], fm[:, :k])
            per[name + f"_physical_mse{k}"] = masked_mse(
                pred[:, :k] * st["hand_action_std"],
                f[:, :k] * st["hand_action_std"],
                fm[:, :k],
            )

    with torch.inference_mode():
        zt = encode(tr, trainidx)
        zv = encode(va, np.arange(len(h)))
        zp = ridge_predict(features["train"], zt, features["validation"])
        measure("visual_probe", decode(zp))
        measure("visual_probe_deploy", decode(zp, deploy, deploy_mask))
        measure("oracle_deploy", decode(zv, deploy, deploy_mask))
        measure(
            "visual_direct",
            ridge_predict(
                features["train"],
                tr["future_hand_norm"][trainidx],
                features["validation"],
            ),
        )
        # Existing DP inference on K-spaced anchors avoids overlapping executed chunks.
        dpmeta = json.loads((root / "dp/run/artifact/artifact.json").read_text())
        core = LAMPDiffusionPolicy(**dpmeta["architecture"]).to(device).eval()
        dpstate = load_file(str(root / "dp/run/artifact/model.safetensors"))
        core.load_state_dict(
            {
                k.removeprefix("core."): v
                for k, v in dpstate.items()
                if k.startswith("core.")
            },
            strict=True,
        )
        idx = np.concatenate([np.flatnonzero(eid == e)[::8] for e in np.unique(eid)])
        images = {
            k: np.load(cache / "validation" / (k + ".npy"), mmap_mode="r")
            for k in (
                "front",
                "wrist",
                "arm_state_pair_norm",
                "hand_state_pair_norm",
                "target_action23",
            )
        }
        preds = {
            name: [] for name in ("dp_matched", "dp_deploy", "dp_arm", "dp_latent")
        }
        g = torch.Generator(device=device).manual_seed(123)
        for start in range(0, len(idx), 64):
            ii = idx[start : start + 64]
            inputs = [image_tensor(images[k][ii], device) for k in ("front", "wrist")]
            inputs.extend(
                torch.tensor(images[k][ii], device=device)
                for k in ("arm_state_pair_norm", "hand_state_pair_norm")
            )
            core.set_decoder_history(
                torch.tensor(h[ii], device=device), torch.tensor(hm[ii], device=device)
            )
            result = core(*inputs, return_aux=True, train=False, generator=g)
            preds["dp_matched"].append(result["hand_action_norm"].cpu().numpy())
            preds["dp_arm"].append(result["pred"][..., :7].cpu().numpy())
            preds["dp_latent"].append(result["latent_action"].cpu().numpy())
            core.set_decoder_history(
                torch.tensor(deploy[ii], device=device),
                torch.tensor(deploy_mask[ii], device=device),
            )
            _, aux = core._decode_core(result["core_action_norm"])
            preds["dp_deploy"].append(aux["hand_action_norm"].cpu().numpy())
        preds = {k: np.concatenate(v) for k, v in preds.items()}
        for name in ("dp_matched", "dp_deploy"):
            per[name + "_mse8"] = masked_mse(
                preds[name][:, :8], f[idx, :8], fm[idx, :8]
            )
        per["dp_arm_normalized_mse8"] = masked_mse(
            preds["dp_arm"][:, :8] / st["arm_action_std"],
            images["target_action23"][idx, :8, :7] / st["arm_action_std"],
            fm[idx, :8],
        )
        scale = core.core_action_std[-model.latent_dim :].cpu().numpy()
        per["dp_latent_nmse8"] = masked_mse(
            preds["dp_latent"][:, :8] / scale, zv[idx, :8] / scale, fm[idx, :8]
        )
    result = {
        "id": root.name,
        "metrics": {k: float(v.mean()) for k, v in per.items()},
        "dp_windows": len(idx),
        "dp_num_inference_steps": core.num_inference_steps,
        "deployment_history_length": 16,
        "probe_source": "pretrained_models/resnet-18; fixed ImageNet backbone",
        "seed": 123,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / (root.name + ".npz"), episode=eid, dp_indices=idx, **per
    )
    (output / (root.name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(root.name, result["metrics"], flush=True)


def main() -> None:
    """Build shared features or evaluate a shard of existing models."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/lamp_sweep/water_plant"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lamp_sweep/transfer_quality_20260909"),
    )
    p.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/lamp_sweep/offline_visual_features_20260909.npz"),
    )
    p.add_argument("--build-features", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(2)
    roots = sorted(a.root.glob("lstm_*"))
    if a.build_features:
        meta = json.loads((roots[0] / "prior/run/artifact/artifact.json").read_text())
        build_features(
            roots[0] / "cache/water_plant" / meta["dataset_fingerprint"],
            a.features,
            a.device,
        )
        return
    roots = [r for r in roots if (r / "eval/result.json").exists()][a.shard :: a.shards]
    if a.limit:
        roots = roots[: a.limit]
    for root in roots:
        if not (a.output / (root.name + ".json")).exists():
            evaluate(root, a.output, a.features, a.device)


if __name__ == "__main__":
    main()
