# Copyright 2026 The RLinf Authors.
"""Decompose posterior KL and audit the deterministic targets learned by DP."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import kurtosis, norm, skew

from rlinf.models.embodiment.lamp.lamplstm_prior import (
    LampLSTMPrior,
)
from scripts.lamplstm_analysis_utils import atomic_json, digest


def distribution(x):
    """Descriptive marginal shape and cross-coordinate dependence, not joint KL."""
    q = norm.ppf((np.arange(len(x)) + 0.5) / len(x))
    centered = (x - x.mean(0)) / x.std(0).clip(1e-12)
    return {
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "skew": skew(x, axis=0).tolist(),
        "excess_kurtosis": kurtosis(x, axis=0).tolist(),
        "correlation": np.corrcoef(x.T).tolist(),
        "marginal_shape_w2_to_normal": np.sqrt(
            ((np.sort(centered, axis=0) - q[:, None]) ** 2).mean(0)
        ).tolist(),
        "tail_abs_gt3_fraction": (np.abs(x) > 3).mean(0).tolist(),
    }


def kl_components(mu, lv):
    """Return raw per-coordinate KL components before beta weighting."""
    mu, lv = np.asarray(mu, dtype=np.float64), np.asarray(lv, dtype=np.float64)
    mean_term = 0.5 * mu**2
    variance_term = 0.5 * (np.exp(lv) - 1 - lv)
    return mean_term, variance_term


def analyze(args):
    torch.set_num_threads(2)
    name = f"h8_b0.0005_lr5e-05_p{args.dropout:g}_{args.mode}_d256_n1_s42"
    base = Path("outputs/lamplstm_followup_96_40/runs").resolve() / name / "prior/run"
    artifact = base / (
        "artifact"
        if args.step == 40000
        else f"checkpoints/global_step_{args.step}/actor/artifact"
    )
    meta = json.loads((artifact / "artifact.json").read_text())
    cache = (
        Path("outputs/lamplstm_shared_sweep_40k/cache/water_plant").resolve()
        / meta["dataset_fingerprint"]
    )
    model = LampLSTMPrior(**meta["architecture"]).to(args.device).eval()
    model.load_state_dict(load_file(str(artifact / "model.safetensors")), strict=True)
    output = {}
    arrays = {}
    for split in ("train", "validation"):
        data = {
            k: np.load(cache / split / f"{k}.npy", mmap_mode="r")
            for k in (
                "future_hand_norm",
                "mask",
                "lamplstm_encoder_history_norm",
                "lamplstm_encoder_history_mask",
            )
        }
        mus = []
        lvs = []
        with torch.inference_mode():
            for start in range(0, len(data["mask"]), 512):
                tensors = [
                    torch.tensor(
                        np.asarray(data[k][start : start + 512]), device=args.device
                    )
                    for k in (
                        "lamplstm_encoder_history_norm",
                        "future_hand_norm",
                        "lamplstm_encoder_history_mask",
                    )
                ]
                mu, lv = model.encode(*tensors)
                mus.append(mu.cpu().numpy())
                lvs.append(lv.cpu().numpy())
        mu, lv = np.concatenate(mus), np.concatenate(lvs)
        valid = np.asarray(data["mask"]) > 0
        selected = mu[valid].astype(np.float64)
        selected_lv = lv[valid].astype(np.float64)
        a, b = kl_components(selected, selected_lv)
        if split == "train":
            center = selected.mean(0)
            scale = selected.std(0).clip(1e-6)
        z = (selected - center) / scale
        per_step = []
        for t in range(mu.shape[1]):
            aa, bb = kl_components(mu[valid[:, t], t], lv[valid[:, t], t])
            per_step.append(
                {"mean_term": aa.mean(0).tolist(), "variance_term": bb.mean(0).tolist()}
            )
        output[split] = {
            "valid_tokens": len(selected),
            "kl": float((a + b).mean()),
            "kl_mean_term": float(a.mean()),
            "kl_variance_term": float(b.mean()),
            "kl_per_dim": (a + b).mean(0).tolist(),
            "mean_term_per_dim": a.mean(0).tolist(),
            "variance_term_per_dim": b.mean(0).tolist(),
            "mu_variance_per_dim": selected.var(0).tolist(),
            "mu_offset_contribution_per_dim": (0.5 * selected.mean(0) ** 2).tolist(),
            "posterior_sigma_median": np.median(
                np.exp(0.5 * selected_lv), axis=0
            ).tolist(),
            "standardized_mu": distribution(z),
            "per_timestep": per_step,
        }
        arrays[split + "_mu"] = mu
        arrays[split + "_logvar"] = lv
        arrays[split + "_mask"] = valid
        print(
            args.mode, args.dropout, args.step, split, output[split]["kl"], flush=True
        )
    job = f"{name}_ck{args.step}_dpseed42"
    diag_path = Path("outputs/lamplstm_followup_96_40/dp_diagnostics") / f"{job}.json"
    diag = json.loads(diag_path.read_text())
    metrics = diag["metrics"]
    output["dp_diagnostics"] = {
        k: float(np.mean([metrics[f"dp_{s}_{k}"] for s in (123, 456, 789)]))
        for k in ("latent_nmse16", "hand_mse8", "decoder_change8", "arm_nmse8")
    }
    output["oracle_hand_mse8"] = metrics["oracle_hand_mse8"]
    episodes = (
        Path("outputs/lamplstm_followup_96_40/runs") / job / "eval/episodes.jsonl"
    )
    es = list(map(json.loads, episodes.read_text().splitlines()))
    output["sr"] = sum(e["success_once"] for e in es) / len(es)
    output.update(
        name=job,
        mode=args.mode,
        dropout=args.dropout,
        step=args.step,
        artifact=str(artifact.resolve()),
        weights_sha=digest(artifact / "model.safetensors"),
        statistics_sha=digest(artifact / "statistics.npz"),
        dp_diagnostics_source=str(diag_path.resolve()),
        train_mean=center.tolist(),
        train_std=scale.tolist(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{job}.npz", **arrays)
    atomic_json(args.output / f"{job}.json", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mode", choices=("concat", "film"), required=True)
    p.add_argument("--dropout", type=float, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/lamplstm_kl_pilot"))
    analyze(p.parse_args())
