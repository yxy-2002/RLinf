# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Evaluate a LAMP-LSTM prior and latent-noise sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.datasets.lamp.action_windows import ActionWindowDataset
from rlinf.models.embodiment.lamp.lamplstm_prior import LampLSTMPrior


def _build_model(config: dict[str, Any]) -> LampLSTMPrior:
    data_cfg = config["data"]
    model_cfg = config["model"]
    return LampLSTMPrior(
        action_dim=int(data_cfg["action_dim"]),
        history_dim=int(data_cfg["action_dim"]),
        horizon=int(data_cfg["horizon"]),
        latent_dim=int(model_cfg["latent_dim"]),
        action_hidden_dim=int(model_cfg["action_hidden_dim"]),
        condition_hidden_dim=int(model_cfg["condition_hidden_dim"]),
        condition_mode_encoder=str(model_cfg["encoder_condition_mode"]),
        condition_mode_decoder=str(model_cfg["decoder_condition_mode"]),
        num_lstm_layers=int(model_cfg["num_lstm_layers"]),
        beta=float(config["loss"]["beta"]),
        condition_drop_prob=0.0,
    )


def _stats(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = mask[..., None].astype(np.float64)
    denominator = weights.sum(axis=(0, 1)).clip(min=1.0)
    mean = (values.astype(np.float64) * weights).sum(axis=(0, 1)) / denominator
    variance = (((values.astype(np.float64) - mean[None, None, :]) ** 2) * weights).sum(
        axis=(0, 1)
    ) / denominator
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _metrics(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    weights = mask[..., None].astype(np.float64)
    denominator = max(float(weights.sum() * target.shape[-1]), 1.0)
    mse = float((((prediction - target) ** 2) * weights).sum() / denominator)
    target_mean, target_std = _stats(target, mask)
    prediction_mean, prediction_std = _stats(prediction, mask)
    return {
        "mse": mse,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "prediction_mean": prediction_mean.tolist(),
        "prediction_std": prediction_std.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-artifact",
        default=None,
        help="Directory containing the persisted DexJoCo action windows; defaults to checkpoint parent.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--latent-mode",
        choices=("posterior", "random"),
        default="posterior",
        help="Use posterior latent or replace it with N(0,I) at every anchor.",
    )
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 0.2, 0.5, 1.0],
    )
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    artifact_dir = (
        checkpoint_path.parent
        if args.data_artifact is None
        else Path(args.data_artifact).expanduser().resolve()
    )
    output_dir = (
        artifact_dir / "evaluation"
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = ActionWindowDataset(artifact_dir, "all")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    targets, masks, baseline_predictions = [], [], []
    posterior_mu, posterior_log_var, sampled_latents = [], [], []
    noisy_predictions = {float(std): [] for std in args.noise_stds}
    # Roll out sequentially along the time axis. The model prediction at the
    # current anchor becomes the newest history frame for the next anchor.
    current_history = dataset[0]["history"].unsqueeze(0).to(device)
    current_mask = dataset[0]["history_mask"].unsqueeze(0).to(device)
    with torch.inference_mode():
        for anchor in range(len(dataset)):
            sample_item = dataset[anchor]
            future = sample_item["future_actions"].unsqueeze(0).to(device)
            future_mask = sample_item["future_mask"].unsqueeze(0).to(device)
            mu, log_var = model.encode(current_history, future, current_mask)
            if args.latent_mode == "random":
                latent = torch.randn(
                    mu.shape, generator=generator, device=device, dtype=mu.dtype
                )
            else:
                latent = mu + torch.exp(0.5 * log_var) * torch.randn(
                    mu.shape, generator=generator, device=device, dtype=mu.dtype
                )
            baseline = model.decode(latent, current_history, current_mask)
            targets.append(future.cpu().numpy())
            masks.append(future_mask.cpu().numpy())
            posterior_mu.append(mu.cpu().numpy())
            posterior_log_var.append(log_var.cpu().numpy())
            sampled_latents.append(latent.cpu().numpy())
            baseline_predictions.append(baseline.cpu().numpy())
            for std in args.noise_stds:
                noisy_latent = latent + float(std) * torch.randn(
                    latent.shape, generator=generator, device=device, dtype=latent.dtype
                )
                noisy_predictions[float(std)].append(
                    model.decode(noisy_latent, current_history, current_mask)
                    .cpu()
                    .numpy()
                )
            current_history = torch.cat(
                (current_history[:, 1:], baseline[:, :1]), dim=1
            )
            current_mask = torch.cat(
                (current_mask[:, 1:], torch.ones_like(current_mask[:, :1])), dim=1
            )
    target_array = np.concatenate(targets, axis=0)
    mask_array = np.concatenate(masks, axis=0)
    baseline_array = np.concatenate(baseline_predictions, axis=0)
    mu_array = np.concatenate(posterior_mu, axis=0)
    log_var_array = np.concatenate(posterior_log_var, axis=0)
    latent_array = np.concatenate(sampled_latents, axis=0)
    metrics = {"baseline": _metrics(target_array, baseline_array, mask_array)}
    arrays = {
        "target": target_array,
        "future_mask": mask_array,
        "baseline": baseline_array,
    }
    for std, chunks in noisy_predictions.items():
        prediction = np.concatenate(chunks, axis=0)
        key = f"noise_{std:g}"
        metrics[key] = _metrics(target_array, prediction, mask_array)
        arrays[key] = prediction
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "reconstructions.npz", **arrays)
    np.savez_compressed(
        output_dir / "latent_samples.npz",
        mu=mu_array,
        log_var=log_var_array,
        sampled_latent=latent_array,
    )
    feature_statistics = {}
    for name, prediction in [("target", target_array), ("baseline", baseline_array)]:
        mean, std_values = _stats(prediction, mask_array)
        feature_statistics[f"{name}_mean"] = mean
        feature_statistics[f"{name}_std"] = std_values
    for std, chunks in noisy_predictions.items():
        prediction = np.concatenate(chunks, axis=0)
        mean, std_values = _stats(prediction, mask_array)
        key = f"noise_{std:g}"
        feature_statistics[f"{key}_mean"] = mean
        feature_statistics[f"{key}_std"] = std_values
    np.savez_compressed(output_dir / "feature_statistics.npz", **feature_statistics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
