# Copyright 2026 The RLinf Authors.
"""Separate reconstruction, DP latent error and decoder error amplification."""

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
from scripts.eval_lamplstm_offline_quality import episode_ids, masked_mse
from scripts.eval_lamplstm_transfer_diagnostics import image_tensor
from scripts.lamplstm_analysis_utils import atomic_json


def matched_direction(delta, rng):
    """Match each window's normalized latent L2 norm without changing its scale."""
    noise = rng.standard_normal(delta.shape).astype(np.float32)
    axes = tuple(range(1, delta.ndim))
    norm = np.sqrt(np.sum(delta**2, axis=axes, keepdims=True))
    denom = np.sqrt(np.sum(noise**2, axis=axes, keepdims=True)).clip(1e-12)
    return noise * norm / denom


def evaluate(prior, cache, output, device, dp=None):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    meta = json.loads((prior / "artifact.json").read_text())
    cm = json.loads((cache / "metadata.json").read_text())
    if (
        meta["dataset_fingerprint"] != cm["fingerprint"]
        or meta["history_contract"] != "primitive_v1"
    ):
        raise ValueError("History/cache contract mismatch")
    model = LampLSTMPrior(**meta["architecture"]).to(device).eval()
    model.load_state_dict(load_file(str(prior / "model.safetensors")), strict=True)
    keys = (
        "future_hand_norm",
        "mask",
        "lamplstm_encoder_history_norm",
        "lamplstm_encoder_history_mask",
    )
    data = {
        s: {k: np.load(cache / s / f"{k}.npy", mmap_mode="r") for k in keys}
        for s in ("train", "validation")
    }
    va = data["validation"]
    eid = episode_ids(va[keys[3]], va[keys[1]])
    idx = np.concatenate([np.flatnonzero(eid == e)[::8] for e in np.unique(eid)])
    f, fm, h, hm = [np.asarray(va[k][idx]) for k in keys]

    def tensor(x):
        return torch.tensor(np.asarray(x), device=device, dtype=torch.float32)

    @torch.inference_mode()
    def encode(d, indices, off=False):
        chunks = []
        for start in range(0, len(indices), 256):
            ii = indices[start : start + 256]
            mask = tensor(d[keys[3]][ii])
            z, _ = model.encode(
                tensor(d[keys[2]][ii]),
                tensor(d[keys[0]][ii]),
                torch.zeros_like(mask) if off else mask,
            )
            chunks.append(z.cpu().numpy())
        return np.concatenate(chunks)

    @torch.inference_mode()
    def decode(z, history=h, masks=hm):
        return np.concatenate(
            [
                model.decode(
                    tensor(z[i : i + 256]),
                    tensor(history[i : i + 256]),
                    tensor(masks[i : i + 256]),
                )
                .cpu()
                .numpy()
                for i in range(0, len(z), 256)
            ]
        )

    mu = encode(va, idx)
    tr = data["train"]
    train_mu = encode(tr, np.arange(len(tr["mask"])))
    valid = train_mu[np.asarray(tr["mask"]) > 0]
    mean = valid.mean(0, dtype=np.float64).astype(np.float32)
    scale = valid.std(0, dtype=np.float64).astype(np.float32).clip(1e-6)
    recon = decode(mu)
    per = {}

    def measure(name, pred, target=f):
        per[name] = masked_mse(pred[:, :8], target[:, :8], fm[:, :8])

    measure("oracle_hand_mse8", recon)
    off = encode(va, idx, off=True)
    off_pred = decode(off, masks=np.zeros_like(hm))
    measure("both_off_mse8", off_pred)
    p = meta["architecture"]["condition_drop_prob"]
    # Same Bernoulli event for both ends; inference is otherwise deterministic.
    keep = np.random.default_rng(123).random(len(mu)) >= p
    measure(
        "shared_dropout_validation_mse8", np.where(keep[:, None, None], recon, off_pred)
    )
    per["shared_dropout_expected_mse8"] = (1 - p) * per["oracle_hand_mse8"] + p * per[
        "both_off_mse8"
    ]
    rng = np.random.default_rng(123)
    for alpha in (0.05, 0.1, 0.5):
        delta = alpha * np.tanh(rng.standard_normal(mu.shape)).astype(np.float32)
        pred = decode(mu + scale * delta)
        measure(f"random_{alpha:g}_gt_mse8", pred)
        measure(f"random_{alpha:g}_decoder_change8", pred, recon)
        per[f"random_{alpha:g}_input_energy"] = masked_mse(
            delta, np.zeros_like(delta), fm
        )

    arrays = {
        "indices": idx,
        "episode_id": eid[idx],
        "oracle_latent": mu,
        "oracle_hand": recon,
        "gt_hand": f,
        "mask": fm,
        "history": h,
        "history_mask": hm,
        "latent_mean": mean,
        "latent_std": scale,
    }
    if dp is not None:
        from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
            LAMPDiffusionPolicy,
        )

        dm = json.loads((dp / "artifact.json").read_text())
        core = LAMPDiffusionPolicy(**dm["architecture"]).to(device).eval()
        state = load_file(str(dp / "model.safetensors"))
        core.load_state_dict(
            {
                k.removeprefix("core."): v
                for k, v in state.items()
                if k.startswith("core.")
            },
            strict=True,
        )
        core.num_inference_steps = 16
        np.testing.assert_allclose(
            core.core_action_std[-mu.shape[-1] :].cpu().numpy(),
            scale,
            rtol=1e-4,
            atol=1e-6,
        )
        images = {
            k: np.load(cache / "validation" / f"{k}.npy", mmap_mode="r")
            for k in (
                "front",
                "wrist",
                "arm_state_pair_norm",
                "hand_state_pair_norm",
                "target_action23",
            )
        }
        for seed in (123, 456, 789):
            generator = torch.Generator(device=device).manual_seed(seed)
            zs, arms, hands = [], [], []
            with torch.inference_mode():
                for start in range(0, len(idx), 32):
                    ii = idx[start : start + 32]
                    inputs = [
                        image_tensor(images[k][ii], device) for k in ("front", "wrist")
                    ]
                    inputs += [
                        tensor(images[k][ii])
                        for k in ("arm_state_pair_norm", "hand_state_pair_norm")
                    ]
                    core.set_decoder_history(
                        tensor(h[start : start + 32]), tensor(hm[start : start + 32])
                    )
                    result = core(
                        *inputs, return_aux=True, train=False, generator=generator
                    )
                    zs.append(result["latent_action"].cpu().numpy())
                    arms.append(result["pred"][..., :7].cpu().numpy())
                    hands.append(result["hand_action_norm"].cpu().numpy())
            z = np.concatenate(zs)
            pred = decode(z)
            np.testing.assert_allclose(
                pred, np.concatenate(hands), rtol=1e-4, atol=1e-5
            )
            delta = (z - mu) / scale
            per[f"dp_{seed}_latent_nmse16"] = masked_mse(
                delta, np.zeros_like(delta), fm
            )
            measure(f"dp_{seed}_hand_mse8", pred)
            measure(f"dp_{seed}_decoder_change8", pred, recon)
            stats = dict(np.load(prior / "statistics.npz"))
            per[f"dp_{seed}_arm_nmse8"] = masked_mse(
                np.concatenate(arms)[:, :8] / stats["arm_action_std"],
                images["target_action23"][idx, :8, :7] / stats["arm_action_std"],
                fm[:, :8],
            )
            matched = matched_direction(delta, np.random.default_rng(seed))
            for fraction in (0.0, 0.25, 0.5, 1.0):
                measure(
                    f"dp_{seed}_actual_direction_{fraction:g}",
                    decode(mu + fraction * scale * delta),
                )
                measure(
                    f"dp_{seed}_random_direction_{fraction:g}",
                    decode(mu + fraction * scale * matched),
                )
            eps = np.tanh(np.random.default_rng(seed).standard_normal(z.shape)).astype(
                np.float32
            )
            for alpha in (0.05, 0.1, 0.5):
                measure(
                    f"dp_{seed}_exploration_{alpha:g}", decode(z + alpha * scale * eps)
                )
            arrays[f"dp_{seed}_latent"] = z
            arrays[f"dp_{seed}_hand"] = pred
            arrays[f"dp_{seed}_arm"] = np.concatenate(arms)
    # Per-coordinate / execution-position summaries use only valid action targets.
    squared = (recon[:, :8] - f[:, :8]) ** 2
    mask = fm[:, :8, None]
    summary = {
        "oracle_mse_by_step": (
            (squared * mask).sum((0, 2))
            / (mask.sum((0, 2)) * squared.shape[-1]).clip(1)
        ).tolist(),
        "oracle_mse_by_joint": (
            (squared * mask).sum((0, 1)) / mask.sum().clip(1)
        ).tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **arrays, **per)
    atomic_json(
        output,
        {
            "prior": str(prior),
            "dp": str(dp) if dp else None,
            "checkpoint": meta["global_step"],
            "windows": len(idx),
            "metrics": {k: float(v.mean()) for k, v in per.items()},
            "p95": {k: float(np.quantile(v, 0.95)) for k, v in per.items()},
            **summary,
            "note": "Demo-state diagnosis; not online ground-truth actions.",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dp", type=Path)
    a = p.parse_args()
    evaluate(a.prior, a.cache, a.output, a.device, a.dp)


if __name__ == "__main__":
    main()
