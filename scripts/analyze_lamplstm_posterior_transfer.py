# Copyright 2026 The RLinf Authors.
"""Compare posterior sampling with cached DP errors for frozen LSTM codecs.

Run from the repository root. No training or diffusion inference is performed.
The primary endpoint is normalized hand MSE over the first eight valid steps.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from rlinf.models.embodiment.lamp.lamplstm_prior import (
    LampLSTMPrior,
)
from scripts.eval_lamplstm_offline_quality import masked_mse
from scripts.lamplstm_analysis_utils import atomic_json, digest

LOGGER = logging.getLogger(__name__)
DP_SEEDS = (123, 456, 789)


def numerical_agreement(
    actual: np.ndarray, cached: np.ndarray, scale: np.ndarray | float = 1.0
) -> dict:
    """Bound CPU/GPU discrepancies in data-standardized units, including zeros."""
    error = (actual.astype(np.float64) - cached) / scale
    result = {
        "rms": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
    }
    if result["rms"] > 1e-3 or result["max_abs"] > 1e-2:
        raise ValueError(f"Cached inference differs materially: {result}")
    return result


def match_window_energy(
    noise: np.ndarray, delta: np.ndarray, mask: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Match valid K8 energy in DP-standardized coordinates per window.

    Args:
        noise: Random displacement in raw posterior coordinates, [N, H, D].
        delta: Actual displacement in the same coordinates and shape.
        mask: Valid future tokens, [N, H]. Only the first eight are matched.
        scale: Training-set standard deviations used by DP, [D].

    Returns:
        Rescaled noise with matching masked K8 energy. Padding cannot set scale.
    """
    weights = mask[:, :8, None]
    energy = (((delta[:, :8] / scale) ** 2) * weights).sum((1, 2))
    noise_energy = (((noise[:, :8] / scale) ** 2) * weights).sum((1, 2))
    if np.any((noise_energy == 0) & (energy > 0)):
        raise ValueError("Cannot match nonzero DP error with zero random direction")
    factor = np.sqrt(energy / noise_energy.clip(1e-30))
    return noise * factor[:, None, None]


def cluster_interval(
    values: np.ndarray, episodes: np.ndarray, draws: int = 2000
) -> dict:
    """Bootstrap whole demo episodes, preserving window-weighted means.

    Values contain one number per anchor; Monte Carlo seeds must be averaged
    before calling. This estimates demo uncertainty, not training-seed variance.
    """
    unique = np.unique(episodes)
    sums = np.array([values[episodes == e].sum() for e in unique])
    counts = np.array([(episodes == e).sum() for e in unique])
    indices = np.random.default_rng(20260914).integers(
        0, len(unique), size=(draws, len(unique))
    )
    boot = sums[indices].sum(1) / counts[indices].sum(1)
    return {
        "mean": float(values.mean()),
        "episode_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "per_episode": (sums / counts).tolist(),
        "window_p95": float(np.quantile(values, 0.95)),
    }


def ratio_summary(ratio: np.ndarray, mask: np.ndarray) -> dict:
    """Describe signed DP errors in posterior-sigma units, [S, N, H, D]."""
    selected = ratio[:, mask > 0].reshape(-1, ratio.shape[-1])
    absolute = np.abs(selected)
    return {
        "rms": float(np.sqrt(np.mean(selected**2))),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "fraction_abs_gt1": float((absolute > 1).mean()),
        "fraction_abs_gt3": float((absolute > 3).mean()),
        "signed_mean_per_dim": selected.mean(0).tolist(),
        "rms_per_dim": np.sqrt((selected**2).mean(0)).tolist(),
        "p95_abs_per_dim": np.quantile(absolute, 0.95, axis=0).tolist(),
        "fraction_abs_gt3_per_dim": (absolute > 3).mean(0).tolist(),
    }


def lag_alignment(delta: np.ndarray, mask: np.ndarray, scale: np.ndarray) -> float:
    """Return pooled uncentered lag-one alignment, not a causal mechanism test."""
    z = delta / scale
    valid = (mask[:, 1:] * mask[:, :-1]) > 0
    left, right = z[:, :-1][valid], z[:, 1:][valid]
    denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
    return float(np.sum(left * right) / max(float(denom), 1e-30))


def verify_receipt(diag_path: Path, prior: Path, dp: Path) -> dict:
    """Verify cached diagnostics and the exact artifacts that generated them."""
    receipt_path = diag_path.with_suffix(".complete.json")
    receipt = json.loads(receipt_path.read_text())
    paths = {
        "prior": prior / "model.safetensors",
        "metadata": prior / "artifact.json",
        "statistics": prior / "statistics.npz",
        "dp": dp / "model.safetensors",
        "dp_metadata": dp / "artifact.json",
    }
    hashes = {key: digest(path) for key, path in paths.items()}
    for key, value in hashes.items():
        if value != receipt["inputs"][key]:
            raise ValueError(f"Diagnostic artifact changed: {key}: {paths[key]}")
    for path in (diag_path, diag_path.with_suffix(".npz")):
        matching = [
            sha
            for name, sha in receipt["outputs"].items()
            if Path(name).resolve() == path.resolve()
        ]
        if matching != [digest(path)]:
            raise ValueError(f"Diagnostic output hash mismatch: {path}")
    return hashes


def analyze(pilot_path: Path, args: argparse.Namespace) -> None:
    """Evaluate one of the previously selected six checkpoints and save arrays."""
    info = json.loads(pilot_path.read_text())
    name = info["name"]
    LOGGER.info("Starting %s", name)
    prior = Path(info["artifact"])
    diag_path = Path(info["dp_diagnostics_source"])
    diag_info = json.loads(diag_path.read_text())
    dp = Path(diag_info["dp"])
    hashes = verify_receipt(diag_path, prior, dp)
    if hashes["prior"] != info["weights_sha"]:
        raise ValueError("KL pilot and DP diagnostic refer to different priors")
    if hashes["statistics"] != info["statistics_sha"]:
        raise ValueError("KL pilot statistics changed")
    data = dict(np.load(diag_path.with_suffix(".npz")))
    meta = json.loads((prior / "artifact.json").read_text())
    model = LampLSTMPrior(**meta["architecture"]).to(args.device).eval()
    state = load_file(str(prior / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    with safe_open(str(dp / "model.safetensors"), framework="pt") as weights:
        for key, value in state.items():
            torch.testing.assert_close(
                weights.get_tensor("core.lamplstm." + key), value, rtol=0, atol=0
            )
        scale = weights.get_tensor("core.core_action_std").numpy()[-model.latent_dim :]
        center = weights.get_tensor("core.core_action_mean").numpy()[
            -model.latent_dim :
        ]
    np.testing.assert_allclose(scale, data["latent_std"], rtol=3e-4, atol=1e-6)
    # Calculations below use the DP buffer itself, not the reestimated scale.
    history, history_mask = data["history"], data["history_mask"]
    target, mask, episodes = data["gt_hand"], data["mask"], data["episode_id"]

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, device=args.device, dtype=torch.float32)

    @torch.inference_mode()
    def decode(z: np.ndarray) -> np.ndarray:
        chunks = []
        for i in range(0, len(z), args.batch_size):
            end = i + args.batch_size
            pred = model.decode(
                tensor(z[i:end]), tensor(history[i:end]), tensor(history_mask[i:end])
            )
            chunks.append(pred.cpu().numpy())
        return np.concatenate(chunks)

    with torch.inference_mode():
        mu, logvar = model.encode(tensor(history), tensor(target), tensor(history_mask))
    mu, logvar = mu.cpu().numpy(), logvar.cpu().numpy()
    sigma = np.exp(0.5 * logvar)
    agreement = {
        "encoder_mu_dp_standardized": numerical_agreement(
            mu, data["oracle_latent"], scale
        )
    }
    with np.load(pilot_path.with_suffix(".npz")) as pilot:
        np.testing.assert_allclose(
            logvar, pilot["validation_logvar"][data["indices"]], rtol=1e-4, atol=1e-5
        )
    oracle = decode(mu)
    agreement["oracle_hand_normalized"] = numerical_agreement(
        oracle, data["oracle_hand"]
    )
    dp_latent = np.stack([data[f"dp_{seed}_latent"] for seed in DP_SEEDS])
    dp_hands = np.stack([decode(z) for z in dp_latent])
    agreement["dp_hand_normalized"] = numerical_agreement(
        dp_hands,
        np.stack([data[f"dp_{seed}_hand"] for seed in DP_SEEDS]),
    )
    delta = dp_latent - mu
    ratio = delta / sigma
    arrays = {
        "indices": data["indices"],
        "episode_id": episodes,
        "mask": mask,
        "mu": mu,
        "logvar": logvar,
        "sigma": sigma,
        "dp_latent": dp_latent,
        "dp_error_sigma_units": ratio,
        "dp_latent_std": scale,
        "dp_latent_mean": center,
        "oracle_hand": oracle,
        "gt_hand": target,
        "dp_hand": dp_hands,
    }
    predictions = {"mu": oracle[None], "dp": dp_hands}
    rng = np.random.default_rng(args.seed)
    posterior_noise = rng.standard_normal(
        (args.posterior_draws // 2,) + mu.shape
    ).astype(np.float32)
    # Antithetic pairs reduce Monte Carlo variance, with unchanged marginals.
    posterior_noise = np.concatenate([posterior_noise, -posterior_noise])
    posterior = []
    for i, noise in enumerate(posterior_noise):
        posterior.append(decode(mu + sigma * noise))
        if (i + 1) % 8 == 0:
            LOGGER.info("%s posterior %s/%s", name, i + 1, len(posterior_noise))
    predictions["posterior"] = np.stack(posterior)

    # Match energy in the executed prefix; decoder is causal in latent time.
    controls = {"posterior_matched": [], "isotropic_matched": []}
    control_lags = {key: [] for key in controls}
    match_max_error = 0.0
    for s, actual in enumerate(delta):
        control_rng = np.random.default_rng(args.seed + 1000 + DP_SEEDS[s])
        for draw in range(args.control_draws):
            noise = control_rng.standard_normal(mu.shape).astype(np.float32)
            for key, raw in (
                ("posterior_matched", sigma * noise),
                ("isotropic_matched", scale * noise),
            ):
                matched = match_window_energy(raw, actual, mask, scale)
                actual_energy = masked_mse(
                    actual[:, :8] / scale, np.zeros_like(actual[:, :8]), mask[:, :8]
                )
                matched_energy = masked_mse(
                    matched[:, :8] / scale, np.zeros_like(matched[:, :8]), mask[:, :8]
                )
                np.testing.assert_allclose(
                    matched_energy, actual_energy, rtol=1e-5, atol=1e-6
                )
                match_max_error = max(
                    match_max_error,
                    float(np.max(np.abs(matched_energy - actual_energy))),
                )
                controls[key].append(decode(mu + matched))
                control_lags[key].append(
                    lag_alignment(matched[:, :8], mask[:, :8], scale)
                )
        LOGGER.info("%s matched controls DP seed %s complete", name, DP_SEEDS[s])
    predictions.update({key: np.stack(value) for key, value in controls.items()})
    stats = dict(np.load(prior / "statistics.npz"))
    action_scale = stats["hand_action_std"]
    metrics = {}
    for key, preds in predictions.items():
        arrays[key + "_hand_samples"] = preds
        metrics[key] = {}
        for horizon in (8, 16):
            for label, reference, unit in (
                ("gt_mse", target, 1.0),
                ("decoder_change", oracle, 1.0),
                ("physical_gt_mse", target, action_scale),
            ):
                values = np.stack(
                    [
                        masked_mse(
                            pred[:, :horizon] * unit,
                            reference[:, :horizon] * unit,
                            mask[:, :horizon],
                        )
                        for pred in preds
                    ]
                )
                metric = f"{label}{horizon}"
                arrays[f"{key}_{metric}"] = values
                metrics[key][metric] = cluster_interval(values.mean(0), episodes)
                metrics[key][metric]["per_draw_mean"] = values.mean(1).tolist()
                metrics[key][metric]["draw_window_p95"] = float(
                    np.quantile(values, 0.95)
                )
        squared = ((preds[:, :, :8] - target[:, :8]) ** 2).mean(0)
        metrics[key]["gt_mse_by_step8"] = (
            (squared * mask[:, :8, None]).sum((0, 2))
            / (mask[:, :8].sum(0) * target.shape[-1]).clip(1)
        ).tolist()
        metrics[key]["gt_mse_by_joint8"] = (
            (squared * mask[:, :8, None]).sum((0, 1)) / mask[:, :8].sum().clip(1)
        ).tolist()

    contrasts = {}
    for left, right in (
        ("posterior", "mu"),
        ("dp", "mu"),
        ("dp", "posterior"),
        ("dp", "posterior_matched"),
        ("dp", "isotropic_matched"),
    ):
        for metric in ("gt_mse8", "decoder_change8"):
            difference = arrays[f"{left}_{metric}"].mean(0) - arrays[
                f"{right}_{metric}"
            ].mean(0)
            contrasts[f"{left}_minus_{right}_{metric}"] = cluster_interval(
                difference, episodes
            )
    posterior_mean = predictions["posterior"].mean(0)
    mean_shift = masked_mse(posterior_mean[:, :8], oracle[:, :8], mask[:, :8])
    posterior_mc = {}
    half = args.posterior_draws // 2
    for count in (half // 2, half):
        # Include both signs of each selected noise draw.
        chosen = np.r_[np.arange(count), half + np.arange(count)]
        posterior_mc[str(2 * count)] = float(arrays["posterior_gt_mse8"][chosen].mean())
    motion = masked_mse(target[:, 1:8], target[:, :7], mask[:, 1:8] * mask[:, :7])
    high_motion = motion >= np.quantile(motion, 0.75)
    arrays["high_motion"] = high_motion
    strata = {}
    for label, selected in (("high_motion_top_quartile", high_motion),):
        strata[label] = {
            "windows": int(selected.sum()),
            "metrics": {
                key: float(arrays[f"{key}_gt_mse8"][:, selected].mean())
                for key in predictions
            },
            "dp_error_sigma_units": ratio_summary(
                ratio[:, selected, :8], mask[selected, :8]
            ),
        }
    result = {
        "name": name,
        "mode": info["mode"],
        "dropout": info["dropout"],
        "step": info["step"],
        "sr": info["sr"],
        "windows": len(mu),
        "demo_episodes": len(np.unique(episodes)),
        "protocol": {
            "posterior_draws": args.posterior_draws,
            "posterior_antithetic": True,
            "control_draws_per_dp_seed": args.control_draws,
            "dp_noise_seeds": list(DP_SEEDS),
            "seed": args.seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "condition": "both enabled, matched demonstration history",
            "main_horizon": 8,
            "full_horizon": 16,
            "matched_energy": "per-window valid K8, DP-standardized coordinates",
            "aggregation": "equal window weight, draws averaged first",
            "ci": "2000 whole-episode bootstrap draws; exploratory, 10 episodes",
            "limitations": "demo-state diagnostic; no new online SR or training",
        },
        "provenance": {
            "pilot": str(pilot_path.resolve()),
            "pilot_sha": digest(pilot_path),
            "diagnostic": str(diag_path),
            "diagnostic_sha": digest(diag_path.with_suffix(".npz")),
            "script_sha": digest(Path(__file__)),
            **hashes,
        },
        "metrics": metrics,
        "contrasts": contrasts,
        "sigma_median_per_dim8": np.median(sigma[:, :8][mask[:, :8] > 0], 0).tolist(),
        "sigma_over_dp_scale_median_per_dim8": np.median(
            (sigma[:, :8] / scale)[mask[:, :8] > 0], 0
        ).tolist(),
        "dp_error_sigma_units8": ratio_summary(ratio[:, :, :8], mask[:, :8]),
        "dp_error_sigma_units16": ratio_summary(ratio, mask),
        "dp_error_sigma_units_by_step": [
            ratio_summary(ratio[:, :, t : t + 1], mask[:, t : t + 1])
            for t in range(mask.shape[1])
        ],
        "lag_one_alignment8": {
            "dp": [lag_alignment(x[:, :8], mask[:, :8], scale) for x in delta],
            "posterior": [
                lag_alignment((sigma * x)[:, :8], mask[:, :8], scale)
                for x in posterior_noise
            ],
            **control_lags,
        },
        "posterior_mean_prediction_shift8": cluster_interval(mean_shift, episodes),
        "posterior_mc_convergence": posterior_mc,
        "matched_energy_max_absolute_error": match_max_error,
        "cached_inference_agreement": agreement,
        "strata": strata,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{name}.npz", **arrays)
    atomic_json(args.output / f"{name}.json", result)
    LOGGER.info(
        "Finished %s: mu=%.6f posterior=%.6f DP=%.6f matched=%.6f",
        name,
        metrics["mu"]["gt_mse8"]["mean"],
        metrics["posterior"]["gt_mse8"]["mean"],
        metrics["dp"]["gt_mse8"]["mean"],
        metrics["posterior_matched"]["gt_mse8"]["mean"],
    )


def main() -> None:
    """Run the six-checkpoint pilot, optionally selecting one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("outputs/lamplstm_kl_pilot"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/lamplstm_posterior_transfer_pilot")
    )
    parser.add_argument("--name", help="Exact pilot checkpoint stem; default all six")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--posterior-draws", type=int, default=32)
    parser.add_argument("--control-draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.posterior_draws < 4 or args.posterior_draws % 4 or args.control_draws < 1:
        parser.error("posterior-draws must be a positive multiple of 4; controls >= 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    paths = sorted(args.pilot.glob(f"{args.name}.json" if args.name else "h*.json"))
    if not paths:
        parser.error("No matching pilot checkpoint metadata")
    for path in paths:
        analyze(path, args)


if __name__ == "__main__":
    main()
