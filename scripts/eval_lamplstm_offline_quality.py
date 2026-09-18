# Copyright 2026 The RLinf Authors.
"""Evaluate frozen LSTM action codecs without training a diffusion policy.

Run from the repository root with PYTHONPATH=. and the project Python.
All perturbations are oracle-anchored diagnostics, not SAC success estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from rlinf.models.embodiment.lamp.lamplstm_prior import (
    LampLSTMPrior,
)


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return per-window errors, excluding padded future frames."""
    return (((pred - target) ** 2).mean(-1) * mask).sum(-1) / mask.sum(-1).clip(1)


def episode_ids(history_mask: np.ndarray, future_mask: np.ndarray) -> np.ndarray:
    """Recover episode boundaries from valid-future tails; validate history resets."""
    ends = np.flatnonzero(future_mask.sum(1) == 1)
    starts = np.r_[0, ends[:-1] + 1]
    assert ends[-1] == len(future_mask) - 1
    assert np.all(history_mask[starts].sum(1) <= 1)
    return np.repeat(np.arange(len(ends)), ends - starts + 1)


def ridge_predict(x: np.ndarray, y: np.ndarray, xv: np.ndarray) -> np.ndarray:
    """Fixed train-only ridge probe; alpha is not tuned to validation or SR."""
    mean, std = x.mean(0), x.std(0).clip(1e-5)
    x = np.c_[((x - mean) / std), np.ones(len(x))].astype(np.float64)
    xv = np.c_[((xv - mean) / std), np.ones(len(xv))].astype(np.float64)
    penalty = np.eye(x.shape[1]) * len(x) * 0.01
    penalty[-1, -1] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y.reshape(len(y), -1))
    return (xv @ weights).reshape((len(xv),) + y.shape[1:]).astype(np.float32)


def evaluate(
    root: Path,
    output: Path,
    device: str,
    seed: int,
    train_samples: int,
    cache_dir: Path | None = None,
) -> None:
    """Measure one native artifact against its held-out episode cache."""
    artifact = root / "prior/run/artifact"
    meta = json.loads((artifact / "artifact.json").read_text())
    state = load_file(str(artifact / "model.safetensors"))
    model = LampLSTMPrior(**meta["architecture"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    cache = cache_dir or root / "cache/water_plant" / meta["dataset_fingerprint"]
    cm = json.loads((cache / "metadata.json").read_text())
    assert cm["fingerprint"] == meta["dataset_fingerprint"]
    assert not set(cm["train_episodes"]) & set(cm["validation_episodes"])
    stats = dict(np.load(artifact / "statistics.npz"))
    for key, val in dict(np.load(cache / "statistics.npz")).items():
        np.testing.assert_array_equal(stats[key], val)
    dpfile = root / "dp/run/artifact/model.safetensors"
    if dpfile.exists():
        with safe_open(str(dpfile), framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if k.startswith("core.lamplstm.")]
            assert len(keys) == len(state), (len(keys), len(state))
            for key in keys:
                torch.testing.assert_close(
                    f.get_tensor(key),
                    state[key.removeprefix("core.lamplstm.")],
                    rtol=0,
                    atol=0,
                )
    arrays = {}
    for split in ("train", "validation"):
        names = [
            "future_hand_norm",
            "mask",
            "lamplstm_encoder_history_norm",
            "lamplstm_encoder_history_mask",
            "arm_state_pair_norm",
            "hand_state_pair_norm",
        ]
        arrays[split] = {
            k: np.load(cache / split / (k + ".npy"), mmap_mode="r") for k in names
        }
    tr, va = arrays["train"], arrays["validation"]
    eid = episode_ids(va["lamplstm_encoder_history_mask"], va["mask"])
    assert len(np.unique(eid)) == len(cm["validation_episodes"])
    train_idx = np.linspace(
        0, len(tr["mask"]) - 1, min(train_samples, len(tr["mask"])), dtype=int
    )
    future = np.asarray(va["future_hand_norm"])
    h = np.asarray(va["lamplstm_encoder_history_norm"])
    hm = np.asarray(va["lamplstm_encoder_history_mask"])
    fm = np.asarray(va["mask"])
    rng = np.random.default_rng(seed)

    def encode(a, history, mask):
        parts, logs = [], []
        for i in range(0, len(a), 512):
            ts = [
                torch.tensor(v[i : i + 512], device=device) for v in (history, a, mask)
            ]
            mu, lv = model.encode(*ts)
            parts.append(mu.cpu().numpy())
            logs.append(lv.cpu().numpy())
        return np.concatenate(parts), np.concatenate(logs)

    def decode(z, history=h, mask=hm):
        parts = []
        for i in range(0, len(z), 512):
            ts = [
                torch.tensor(v[i : i + 512], device=device) for v in (z, history, mask)
            ]
            parts.append(model.decode(*ts).cpu().numpy())
        return np.concatenate(parts)

    with torch.inference_mode():
        train_mu, _ = encode(
            tr["future_hand_norm"][train_idx],
            tr["lamplstm_encoder_history_norm"][train_idx],
            tr["lamplstm_encoder_history_mask"][train_idx],
        )
        mu, lv = encode(future, h, hm)
        off_mu, _ = encode(future, h, np.zeros_like(hm))
        scale = train_mu.reshape(-1, model.latent_dim).std(0).clip(1e-5)
        reconstruction = decode(mu)
        per_window = {}
        ast = stats["hand_action_std"]
        gtphysical = future * ast + stats["hand_action_mean"]
        lastphysical = h[:, -1] * stats["hand_history_std"] + stats["hand_history_mean"]
        hold = np.broadcast_to(
            ((lastphysical - stats["hand_action_mean"]) / ast)[:, None], future.shape
        )
        dynamics = np.mean(np.diff(gtphysical[:, :8], axis=1) ** 2, axis=(1, 2))
        dynamic_cut = np.quantile(dynamics, 0.75)

        def measure(name, pred):
            for k in (8, 16):
                per_window[f"{name}_mse{k}"] = masked_mse(
                    pred[:, :k], future[:, :k], fm[:, :k]
                )
                per_window[f"{name}_physical_mse{k}"] = masked_mse(
                    pred[:, :k] * ast, future[:, :k] * ast, fm[:, :k]
                )
            per_window[f"{name}_deviation8"] = masked_mse(
                pred[:, :8] * ast, reconstruction[:, :8] * ast, fm[:, :8]
            )
            physical = pred * ast + stats["hand_action_mean"]
            per_window[f"{name}_boundary"] = (
                (physical[:, 0] - lastphysical) ** 2
            ).mean(-1)
            for order, label in ((1, "velocity"), (3, "jerk")):
                valid = np.stack(
                    [fm[:, j : 8 - order + j] for j in range(order + 1)]
                ).prod(0)
                d = np.diff(physical[:, :8], n=order, axis=1)
                dg = np.diff(gtphysical[:, :8], n=order, axis=1)
                per_window[f"{name}_{label}_error"] = masked_mse(d, dg, valid)
                per_window[f"{name}_{label}_power"] = masked_mse(
                    d, np.zeros_like(d), valid
                )

        measure("recon", reconstruction)
        measure("hold", hold)
        measure("encoder_off", decode(off_mu))
        measure("decoder_off", decode(mu, h, np.zeros_like(hm)))
        measure("both_off", decode(off_mu, h, np.zeros_like(hm)))
        measure("zero", decode(np.zeros_like(mu)))
        perm = np.array([rng.choice(np.flatnonzero(eid != e)) for e in eid])
        measure("shuffle", decode(mu[perm]))
        measure(
            "posterior_sample",
            decode(
                mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)
            ),
        )
        noise = rng.standard_normal(mu.shape).astype(np.float32)
        for alpha in (0.1, 0.5, 1.0):
            for kind, eps in [("gauss", noise), ("tanh", np.tanh(noise))]:
                measure(f"{kind}{alpha:g}", decode(mu + alpha * scale * eps))
        # Same perturbation in history-normalized coordinates across models.
        noisy = h + 0.1 * rng.standard_normal(h.shape).astype(np.float32)
        measure("history_noise", decode(mu, noisy))
        measure(
            "history_repeat",
            decode(mu, np.repeat(h[:, -1:], h.shape[1], axis=1), np.ones_like(hm)),
        )
        # Match each of K possible policy-grid offsets, averaged across windows.
        starts = np.r_[0, np.flatnonzero(np.diff(eid)) + 1]
        start_for_row = starts[eid]
        indices = np.maximum(
            np.arange(len(h))[:, None] - 8 * np.arange(h.shape[1] - 1, -1, -1),
            start_for_row[:, None],
        )
        stride_history = h[indices, -1]
        measure("history_stride8", decode(mu, stride_history, np.ones_like(hm)))
        newmu, _ = encode(future, stride_history, np.ones_like(hm))
        measure("both_stride8", decode(newmu, stride_history, np.ones_like(hm)))

        def features(data, idx):
            return np.concatenate(
                [
                    data[k][idx].reshape(len(data[k][idx]), -1)
                    for k in ("arm_state_pair_norm", "hand_state_pair_norm")
                ],
                -1,
            )

        xt, xv = features(tr, train_idx), features(va, slice(None))
        probe = ridge_predict(xt, train_mu, xv)
        measure("probe", decode(probe))
        measure("probe_stride8", decode(probe, stride_history, np.ones_like(hm)))
        measure(
            "direct_probe", ridge_predict(xt, tr["future_hand_norm"][train_idx], xv)
        )
        per_window["probe_latent_nmse"] = masked_mse(probe / scale, mu / scale, fm)
        per_window["kl"] = ((0.5 * (np.exp(lv) + mu**2 - 1 - lv)).mean(-1) * fm).sum(
            -1
        ) / fm.sum(-1)
        per_window["posterior_std"] = (np.exp(0.5 * lv).mean(-1) * fm).sum(-1) / fm.sum(
            -1
        )
        per_window["latent_energy"] = (np.square(mu).mean(-1) * fm).sum(-1) / fm.sum(-1)
        for variant in (
            "zero",
            "shuffle",
            "history_stride8",
            "decoder_off",
            "probe",
            "tanh1",
        ):
            per_window[variant + "_excess8"] = (
                per_window[variant + "_mse8"] - per_window["recon_mse8"]
            )
        metrics = {k: float(v.mean()) for k, v in per_window.items()}
        for name in ("recon", "probe", "tanh1", "history_stride8", "zero", "shuffle"):
            key = name + "_mse8"
            metrics[key + "_p95"] = float(np.quantile(per_window[key], 0.95))
            metrics[key + "_dynamic"] = float(
                per_window[key][dynamics >= dynamic_cut].mean()
            )
            metrics[key + "_episode_mean"] = float(
                np.mean([per_window[key][eid == e].mean() for e in np.unique(eid)])
            )
        metrics["latent_train_std"] = float(scale.mean())
        metrics["latent_utilization"] = 1 - metrics["recon_mse8"] / max(
            metrics["shuffle_mse8"], 1e-12
        )
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / (root.name + ".npz"), episode=eid, **per_window)
        result_path = root / "eval/result.json"
        sr = json.loads(result_path.read_text()) if result_path.exists() else {}
        payload = {
            "id": root.name,
            "seed": seed,
            "metrics": metrics,
            "sr": sr.get("eval/success_once"),
            "n_trajectories": sr.get("eval/num_trajectories"),
            "prior_model_sha256": hashlib.sha256(
                (artifact / "model.safetensors").read_bytes()
            ).hexdigest(),
            "dp_prior_weights_identical": True if dpfile.exists() else None,
            "legacy_independent_dropout": meta.get("condition_dropout_scope")
            != "shared_per_sample",
            "cache": str(cache),
            "cache_metadata": cm,
            "architecture": meta["architecture"],
            "actual_history_length": h.shape[1],
            "metadata_encoder_history_length": meta.get("encoder_history_length"),
            "train_probe_samples": len(train_idx),
            "validation_windows": len(h),
        }
        (output / (root.name + ".json")).write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        print(
            root.name,
            "sr",
            payload["sr"],
            "recon",
            round(metrics["recon_mse8"], 5),
            "probe",
            round(metrics["probe_mse8"], 5),
            flush=True,
        )


def main() -> None:
    """Evaluate one shard of the sweep, optionally resuming completed models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/lamp_sweep/water_plant")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lamp_sweep/offline_quality_20260909"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--cache", type=Path, help="Existing cache directory for a prior-only run."
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--experiment",
        type=Path,
        help="Evaluate one run, including prior-only runs without a DP artifact.",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    roots = sorted(args.root.glob("lstm_*"))[args.shard :: args.shards]
    if args.experiment is not None:
        roots = [args.experiment]
    if args.limit:
        roots = roots[: args.limit]
    for root in roots:
        if (args.output / (root.name + ".json")).exists():
            continue
        evaluate(
            root, args.output, args.device, args.seed, args.train_samples, args.cache
        )


if __name__ == "__main__":
    main()
