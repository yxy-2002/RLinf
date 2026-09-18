# Copyright 2026 The RLinf Authors.
"""Train a small fixed-budget latent probe, never a diffusion policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from rlinf.models.embodiment.lamp.lamplstm_prior import (
    LampLSTMPrior,
)
from scripts.eval_lamplstm_offline_quality import episode_ids, masked_mse


def evaluate(root: Path, output: Path, feature_path: Path, device: str) -> None:
    """Fit only training examples and decode predictions on validation episodes."""
    torch.manual_seed(42)
    feature = dict(np.load(feature_path))
    prior = root / "prior/run/artifact"
    meta = json.loads((prior / "artifact.json").read_text())
    cache = root / "cache/water_plant" / meta["dataset_fingerprint"]
    model = LampLSTMPrior(**meta["architecture"]).to(device).eval()
    model.load_state_dict(load_file(str(prior / "model.safetensors")), strict=True)
    data = {}
    latent = {}
    with torch.no_grad():
        for split in ("train", "validation"):
            idx = feature[split + "_idx"]
            data[split] = {
                k: np.load(cache / split / (k + ".npy"), mmap_mode="r")[idx]
                for k in (
                    "future_hand_norm",
                    "mask",
                    "lamplstm_encoder_history_norm",
                    "lamplstm_encoder_history_mask",
                )
            }
            d = data[split]
            zs = []
            for i in range(0, len(idx), 512):
                zs.append(
                    model.encode(
                        torch.tensor(
                            d["lamplstm_encoder_history_norm"][i : i + 512],
                            device=device,
                        ),
                        torch.tensor(d["future_hand_norm"][i : i + 512], device=device),
                        torch.tensor(
                            d["lamplstm_encoder_history_mask"][i : i + 512],
                            device=device,
                        ),
                    )[0]
                    .cpu()
                    .numpy()
                )
            latent[split] = np.concatenate(zs)
    mean = latent["train"].reshape(-1, model.latent_dim).mean(0)
    std = latent["train"].reshape(-1, model.latent_dim).std(0).clip(1e-5)
    xm, xs = feature["train"].mean(0), feature["train"].std(0).clip(1e-4)
    xt = torch.tensor((feature["train"] - xm) / xs, device=device)
    xv = torch.tensor((feature["validation"] - xm) / xs, device=device)
    yt = torch.tensor((latent["train"] - mean) / std, device=device)
    mask = torch.tensor(data["train"]["mask"], device=device)
    probe = nn.Sequential(
        nn.Linear(xt.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 32),
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(123)
    for _ in range(1000):
        ix = torch.randint(len(xt), (256,), device=device, generator=generator)
        pred = probe(xt[ix]).reshape(-1, 16, 2)
        loss = (((pred - yt[ix]) ** 2).mean(-1) * mask[ix]).sum() / mask[ix].sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        trainloss = (
            ((probe(xt).reshape(-1, 16, 2) - yt) ** 2).mean(-1) * mask
        ).sum() / mask.sum()
        z = probe(xv).reshape(-1, 16, 2).cpu().numpy() * std + mean
        va = data["validation"]
        h = va["lamplstm_encoder_history_norm"]
        hm = va["lamplstm_encoder_history_mask"]
        fm = va["mask"]
        f = va["future_hand_norm"]
        eid = episode_ids(hm, fm)
        starts = np.r_[0, np.flatnonzero(np.diff(eid)) + 1]
        ix = np.maximum(
            np.arange(len(h))[:, None] - 8 * np.arange(15, -1, -1), starts[eid, None]
        )
        dh = h[ix, -1]
        arrays = {}
        for name, history, hmval in [
            ("mlp_probe", h, hm),
            ("mlp_probe_deploy", dh, np.ones(dh.shape[:2], np.float32)),
        ]:
            chunks = []
            for i in range(0, len(z), 512):
                chunks.append(
                    model.decode(
                        torch.tensor(z[i : i + 512], device=device),
                        torch.tensor(history[i : i + 512], device=device),
                        torch.tensor(hmval[i : i + 512], device=device),
                    )
                    .cpu()
                    .numpy()
                )
            pred = np.concatenate(chunks)
            arrays[name + "_mse8"] = masked_mse(pred[:, :8], f[:, :8], fm[:, :8])
        arrays["mlp_probe_latent_nmse"] = masked_mse(
            z / std, latent["validation"] / std, fm
        )
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": root.name,
        "steps": 1000,
        "train_samples": len(xt),
        "metrics": {k: float(v.mean()) for k, v in arrays.items()},
        "train_loss": float(trainloss),
    }
    np.savez_compressed(output / (root.name + ".npz"), episode=eid, **arrays)
    (output / (root.name + ".json")).write_text(json.dumps(payload, indent=2) + "\n")
    print(payload, flush=True)


def main() -> None:
    """Run one shard with a fixed feature source and optimization budget."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/lamp_sweep/water_plant"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lamp_sweep/nonlinear_quality_20260909"),
    )
    p.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/lamp_sweep/offline_visual_features_20260909.npz"),
    )
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(2)
    roots = [
        r for r in sorted(a.root.glob("lstm_*")) if (r / "eval/result.json").exists()
    ][a.shard :: a.shards]
    for root in roots:
        if not (a.output / (root.name + ".json")).exists():
            evaluate(root, a.output, a.features, a.device)


if __name__ == "__main__":
    main()
