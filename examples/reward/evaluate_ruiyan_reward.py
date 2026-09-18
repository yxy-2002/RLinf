# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline evaluation of trusted local reward checkpoints and labeled images."""

import argparse
import csv
import html
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel


def classification_metrics(labels, probabilities, threshold):
    pairs = list(zip(labels, probabilities))
    tp = sum(y == 1 and p > threshold for y, p in pairs)
    tn = sum(y == 0 and p <= threshold for y, p in pairs)
    fp = sum(y == 0 and p > threshold for y, p in pairs)
    fn = sum(y == 1 and p <= threshold for y, p in pairs)
    return {
        "total": len(pairs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(pairs),
        "success_recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "success_precision": tp / (tp + fp) if tp + fp else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config/reward_training.yaml",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size < 1:
        parser.error("threshold must be in [0,1] and batch size positive")
    if args.output.exists():
        parser.error("Output already exists; use a new directory to preserve reports")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    if "success_images" in data:
        images = data["success_images"] + data["failure_images"]
        labels = [1] * len(data["success_images"]) + [0] * len(data["failure_images"])
    else:
        images, labels = data["images"], list(map(int, data["labels"]))
    if not len(images) or len(images) != len(labels) or not set(labels) <= {0, 1}:
        raise ValueError("Expected nonempty matched images and binary labels")
    cfg = OmegaConf.load(args.config).actor.model
    keys = list(cfg.get("image_keys", []))
    if len(keys) > 1 and data.get("metadata", {}).get("image_keys") != keys:
        raise ValueError("Dataset camera order does not match model image_keys")
    for image in images:
        if (
            image.dtype != torch.uint8
            or tuple(image.shape)[-1] != 3
            or image.ndim != (4 if len(keys) > 1 else 3)
        ):
            raise ValueError("Expected HWC RGB uint8 images from reward collector")
    cfg = OmegaConf.load(args.config).actor.model
    cfg.pretrained = False  # Complete checkpoint supplies all weights; no download.
    cfg.model_path = str(args.checkpoint.resolve())
    torch.set_num_threads(4)
    model = ResNetRewardModel(cfg).to(args.device).eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            batch = torch.stack(images[start : start + args.batch_size]).to(args.device)
            probabilities.extend(model(batch)["probabilities"].cpu().tolist())
    if not all(0 <= p <= 1 for p in probabilities):
        raise ValueError("Nonfinite or invalid model probabilities")
    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    metrics = classification_metrics(labels, probabilities, args.threshold)
    metrics.update(
        threshold=args.threshold,
        checkpoint=str(args.checkpoint.resolve()),
        data=str(args.data.resolve()),
        metadata=data.get("metadata"),
        model_config=OmegaConf.to_container(cfg, resolve=True),
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cards, errors = [], []
    with (args.output / "predictions.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["index", "label", "success_probability", "prediction", "correct", "image"]
        )
        for i, (im, y, p) in enumerate(zip(images, labels, probabilities)):
            pred = int(p > args.threshold)
            name = f"images/{i:04d}.png"
            preview = torch.cat(list(im), dim=1) if im.ndim == 4 else im
            Image.fromarray(preview.numpy()).save(args.output / name)
            writer.writerow([i, y, p, pred, pred == y, name])
            card = f'<figure><img width="256" src="{name}"><figcaption>#{i} label={y} p={p:.4f} pred={pred} {"OK" if pred == y else "ERROR"}</figcaption></figure>'
            cards.append(card)
            if pred != y:
                errors.append(card)
    for name, content in [("index.html", cards), ("errors.html", errors)]:
        (args.output / name).write_text(
            '<meta charset="utf-8"><style>body{font-family:sans-serif}figure{display:inline-block;margin:8px}</style><h1>Reward evaluation</h1><pre>'
            + html.escape(
                json.dumps(
                    {k: v for k, v in metrics.items() if k != "model_config"}, indent=2
                )
            )
            + "</pre>"
            + "".join(content or ["<p>No misclassified frames.</p>"])
        )
    print(
        json.dumps({k: v for k, v in metrics.items() if k != "model_config"}, indent=2)
    )


if __name__ == "__main__":
    main()
