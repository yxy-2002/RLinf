# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the three self-conditioned LAMP-LSTM SAC exploration diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.datasets.lamp.action_windows import ActionWindowDataset
from rlinf.models.embodiment.lamp.lamplstm_prior import LampLSTMPrior


def build_model(cfg: dict[str, Any]) -> LampLSTMPrior:
    d, m = cfg["data"], cfg["model"]
    return LampLSTMPrior(
        action_dim=int(d["action_dim"]),
        history_dim=int(d["action_dim"]),
        horizon=int(d["horizon"]),
        latent_dim=int(m["latent_dim"]),
        action_hidden_dim=int(m["action_hidden_dim"]),
        condition_hidden_dim=int(m["condition_hidden_dim"]),
        condition_mode_encoder=str(m["encoder_condition_mode"]),
        condition_mode_decoder=str(m["decoder_condition_mode"]),
        num_lstm_layers=int(m["num_lstm_layers"]),
        beta=float(cfg["loss"]["beta"]),
        condition_drop_prob=0.0,
    )


def summary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "p95": float(np.nanpercentile(x, 95)),
        "max": float(np.nanmax(x)),
    }


def dynamics(actions: np.ndarray) -> dict[str, Any]:
    vel = np.diff(actions, axis=0)
    acc = np.diff(actions, n=2, axis=0)
    jerk = np.diff(actions, n=3, axis=0)
    return {
        "velocity": summary(np.linalg.norm(vel, axis=-1)),
        "acceleration": summary(np.linalg.norm(acc, axis=-1)),
        "jerk": summary(np.linalg.norm(jerk, axis=-1)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-artifact", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fixed-samples", type=int, default=100)
    p.add_argument(
        "--max-anchors",
        type=int,
        default=None,
        help="Evaluate only the first N anchors (useful for large LeRobot artifacts).",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.01, 0.1, 0.2, 0.5, 1.0]
    )
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    artifact = Path(args.data_artifact or ckpt_path.parent).expanduser().resolve()
    out = Path(args.output_dir or artifact / "evaluation" / "sac_exploration").resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    ds = ActionWindowDataset(artifact, "all")
    n, horizon = len(ds), model.horizon
    if args.max_anchors is not None:
        n = min(n, int(args.max_anchors))
    alphas = [float(a) for a in args.alphas]
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    # Experiment 1: each alpha has an independent self-generated history.
    rollouts: dict[float, np.ndarray] = {}
    gains: dict[float, list[np.ndarray]] = {a: [] for a in alphas}
    histories: dict[float, list[np.ndarray]] = {a: [] for a in alphas}
    with torch.inference_mode():
        for alpha in alphas:
            item0 = ds[0]
            history = item0["history"].unsqueeze(0).to(device)
            mask = item0["history_mask"].unsqueeze(0).to(device)
            chunks, refs = [], []
            for t in range(n):
                item = ds[t]
                future = item["future_actions"].unsqueeze(0).to(device)
                mu, _ = model.encode(history, future, mask)
                noise = torch.randn(
                    mu.shape, generator=rng, device=device, dtype=mu.dtype
                )
                z = mu + alpha * noise
                ref = model.decode(mu, history, mask)
                pred = model.decode(z, history, mask)
                chunks.append(pred.cpu().numpy()[0])
                refs.append(ref.cpu().numpy()[0])
                histories[alpha].append(history.cpu().numpy()[0])
                if alpha > 0:
                    delta = alpha * noise
                    gains[alpha].append(
                        (pred - ref).norm(dim=-1).cpu().numpy()
                        / delta.norm(dim=-1).clamp_min(1e-12).cpu().numpy()
                    )
                history = torch.cat((history[:, 1:], pred[:, :1]), dim=1)
                mask = torch.cat((mask[:, 1:], torch.ones_like(mask[:, :1])), dim=1)
            rollouts[alpha] = np.asarray(chunks, dtype=np.float32)
            if alpha == 0.0:
                reference = np.asarray(refs, dtype=np.float32)
    dev = {a: rollouts[a] - reference for a in alphas}
    e1 = {
        "alphas": alphas,
        "deviation": {
            str(a): {
                "l2": summary(np.linalg.norm(dev[a], axis=-1)),
                "mse": summary(dev[a] ** 2),
                "per_dim_mean": np.mean(np.abs(dev[a]), axis=(0, 1)).tolist(),
            }
            for a in alphas
        },
        "gain": {
            str(a): summary(np.concatenate(gains[a])) if gains[a] else None
            for a in alphas
        },
    }
    np.savez_compressed(
        out / "experiment_1_noise_sweep.npz",
        reference=reference,
        **{f"alpha_{a:g}": rollouts[a] for a in alphas},
        **{f"deviation_{a:g}": dev[a] for a in alphas},
    )

    # Experiment 2: fixed alpha=0 condition, many latent samples, no history update.
    sample_alpha = [a for a in (0.1, 0.5, 1.0) if a in alphas] or [0.1, 0.5, 1.0]
    fixed = {a: [] for a in sample_alpha}
    fixed_std = {a: [] for a in sample_alpha}
    with torch.inference_mode():
        for t in range(n):
            item = ds[t]
            hist = torch.from_numpy(histories[0.0][t]).unsqueeze(0).to(device)
            mask = torch.ones((1, hist.shape[1]), device=device)
            future = item["future_actions"].unsqueeze(0).to(device)
            mu, _ = model.encode(hist, future, mask)
            for alpha in sample_alpha:
                z = mu.expand(args.fixed_samples, -1, -1) + alpha * torch.randn(
                    (args.fixed_samples, horizon, model.latent_dim),
                    generator=rng,
                    device=device,
                )
                acts = (
                    model.decode(
                        z,
                        hist.expand(args.fixed_samples, -1, -1),
                        mask.expand(args.fixed_samples, -1),
                    )
                    .cpu()
                    .numpy()
                )
                fixed[alpha].append(acts)
                fixed_std[alpha].append(acts.std(axis=0))
    e2 = {
        "samples": args.fixed_samples,
        "alphas": sample_alpha,
        "action_std": {
            str(a): {
                "mean": summary(np.asarray(fixed_std[a])),
                "p95": float(np.percentile(np.asarray(fixed_std[a]), 95)),
                "max": float(np.max(np.asarray(fixed_std[a]))),
            }
            for a in sample_alpha
        },
    }
    np.savez_compressed(
        out / "experiment_2_fixed_condition.npz",
        **{
            f"alpha_{a:g}": np.asarray(fixed[a], dtype=np.float32) for a in sample_alpha
        },
    )

    # Experiment 3: long self-conditioned rollouts and surrogate envelope checks.
    seq = np.load(artifact / "sequence.npy").astype(np.float32)
    lo, hi = seq.min(0), seq.max(0)
    e3 = {"alphas": alphas, "rollout": {}}
    for alpha in alphas:
        a = rollouts[alpha]
        envelope = (a < lo[None, None, :]) | (a > hi[None, None, :])
        e3["rollout"][str(alpha)] = {
            "action": summary(a),
            "out_of_envelope_fraction": float(envelope.mean()),
            "final_mse": float(np.mean((a[-1] - seq[-1]) ** 2)),
            "nan_or_inf": bool(~np.isfinite(a).all()),
            "dynamics": dynamics(a),
            "cumulative_deviation_l2": float(
                np.linalg.norm(a - reference, axis=-1).sum()
            ),
        }
    np.savez_compressed(
        out / "experiment_3_autoregressive.npz",
        **{f"alpha_{a:g}": rollouts[a] for a in alphas},
    )
    result = {
        "checkpoint": str(ckpt_path),
        "seed": args.seed,
        "condition_mode": f"encoder-{ckpt['config']['model']['encoder_condition_mode']}_decoder-{ckpt['config']['model']['decoder_condition_mode']}",
        "condition_drop_prob": float(
            ckpt["config"]["model"].get("condition_drop_prob", 0.0)
        ),
        "alphas": alphas,
        "experiment_1": e1,
        "experiment_2": e2,
        "experiment_3": e3,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
