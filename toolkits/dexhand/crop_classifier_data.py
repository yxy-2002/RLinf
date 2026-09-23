# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Crop native reward and demo datasets, preserving image shapes and metadata."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.reward_model import RewardDatasetPayload
from toolkits.dexhand.review_classifier_data import image_views, validate_payload


def validate_region(value: list[float]) -> tuple[float, float, float, float]:
    """Validate normalized top/left/bottom/right bounds."""
    if len(value) != 4:
        raise ValueError("Crop bounds must be [top, left, bottom, right]")
    top, left, bottom, right = map(float, value)
    if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
        raise ValueError(f"Invalid normalized crop bounds: {value}")
    return top, left, bottom, right


def load_crops(path: Path) -> tuple[dict, dict[str, tuple]]:
    """Convert explicit stored-image or raw-camera coordinates to stored bounds."""
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    space = config.get("coordinates", "stored")
    if space not in ("stored", "raw") or not config.get("cameras"):
        raise ValueError("Set coordinates to stored/raw and provide cameras")
    crops = {}
    for camera, settings in config["cameras"].items():
        target = validate_region(settings["crop"])
        if space == "raw":
            if "source_region" not in settings:
                raise ValueError(
                    f"{camera}: raw coordinates require the source_region represented by stored images"
                )
            source = validate_region(settings["source_region"])
            top, left, bottom, right = target
            st, sl, sb, sr = source
            if top < st or left < sl or bottom > sb or right > sr:
                raise ValueError(
                    f"{camera}: requested crop includes pixels absent from stored images"
                )
            target = (
                (top - st) / (sb - st),
                (left - sl) / (sr - sl),
                (bottom - st) / (sb - st),
                (right - sl) / (sr - sl),
            )
        crops[camera] = target
    return config, crops


def crop_images(
    images: np.ndarray | torch.Tensor, bounds: tuple
) -> np.ndarray | torch.Tensor:
    """Crop HWC batches and resize back using the online environment's OpenCV interpolation."""
    tensor = isinstance(images, torch.Tensor)
    arr = images.detach().cpu().numpy() if tensor else np.asarray(images)
    if arr.ndim < 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 RGB [...,H,W,3], got {arr.shape}, {arr.dtype}"
        )
    height, width = arr.shape[-3:-1]
    top, left, bottom, right = validate_region(bounds)
    y0, x0, y1, x1 = (
        int(height * top),
        int(width * left),
        int(height * bottom),
        int(width * right),
    )
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Crop is empty at stored resolution {width}x{height}")
    flat = arr.reshape(-1, height, width, 3)
    output = np.empty_like(flat)
    for i, frame in enumerate(flat):
        output[i] = cv2.resize(
            frame[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_LINEAR
        )
    output = output.reshape(arr.shape)
    return torch.from_numpy(output).to(images.device) if tensor else output


def crop_reward(data: dict, crops: dict, config: dict) -> dict:
    """Crop named reward views while retaining labels, IDs and camera ordering."""
    payload = RewardDatasetPayload.from_dict(data)
    keys = validate_payload(payload)
    missing = set(crops) - set(keys)
    if missing:
        raise ValueError(f"Crop cameras missing from reward payload: {sorted(missing)}")
    images = []
    for image in payload.images:
        views = image_views(image).clone()
        for name, bounds in crops.items():
            idx = keys.index(name)
            views[idx] = crop_images(views[idx], bounds)
        if image.ndim == 3:
            views = views[0]
            if image.shape[-1] != 3:
                views = views.permute(2, 0, 1)
        images.append(views.contiguous())
    metadata = copy.deepcopy(payload.metadata)
    preprocessing = metadata.setdefault("preprocessing", {})
    preprocessing.setdefault("offline_crops", []).append(copy.deepcopy(config))
    return RewardDatasetPayload(images, payload.labels, metadata).to_dict()


def crop_observation(obs: dict, crops: dict, camera_keys: list[str]) -> dict:
    """Map the main camera and alphabetically ordered extra views by name."""
    if not camera_keys or len(camera_keys) != len(set(camera_keys)):
        raise ValueError(
            "Demo cropping requires unique --camera-keys (main camera first)"
        )
    if set(crops) - set(camera_keys):
        raise ValueError("Crop cameras are absent from --camera-keys")
    result = dict(obs)
    extras = sorted(camera_keys[1:])
    for name, bounds in crops.items():
        if name == camera_keys[0]:
            if "main_images" not in obs:
                raise ValueError("Missing main_images in demo observation")
            result["main_images"] = crop_images(obs["main_images"], bounds)
        else:
            value = result.get("extra_view_images")
            if value is None or value.ndim < 4 or value.shape[-4] != len(extras):
                raise ValueError(
                    "extra_view_images view count does not match --camera-keys"
                )
            value = value.clone() if isinstance(value, torch.Tensor) else value.copy()
            index = extras.index(name)
            value[..., index, :, :, :] = crop_images(value[..., index, :, :, :], bounds)
            result["extra_view_images"] = value
    return result


def crop_data(
    data: dict, crops: dict, config: dict, camera_keys: list[str] | None
) -> tuple[dict, str]:
    """Dispatch native reward, replay-buffer and CollectEpisode payloads."""
    if not isinstance(data, dict):
        raise ValueError(
            "Expected a native RLinf dataset dictionary, not infra transition pickles"
        )
    if "images" in data and "labels" in data:
        return crop_reward(data, crops, config), "reward"
    result = dict(data)
    if "curr_obs" in data and "next_obs" in data:
        for key in ("curr_obs", "next_obs"):
            result[key] = crop_observation(data[key], crops, camera_keys)
        return result, "trajectory"
    if isinstance(data.get("observations"), list) and "actions" in data:
        result["observations"] = [
            crop_observation(obs, crops, camera_keys) for obs in data["observations"]
        ]
        return result, "episode"
    raise ValueError(
        "Unsupported payload: expected reward images/labels, curr_obs/next_obs, or episode observations"
    )


def crop_directory(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    camera_keys: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Transform a dataset tree into a fresh directory, retaining non-image files."""
    source, destination = input_dir.resolve(), output_dir.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("Input and output directories must not overlap")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"Output must be absent or empty: {destination}")
    config, crops = load_crops(config_path)
    paths = sorted(p for p in source.rglob("*") if p.is_file())
    if not any(p.suffix in (".pt", ".pkl") for p in paths):
        raise ValueError("No .pt/.pkl datasets found")
    manifest = {
        "source": str(source),
        "crop_config": config,
        "camera_keys": camera_keys,
        "files": [],
    }
    staging = None
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".crop-", dir=destination.parent))
    try:
        for path in paths:
            relative = path.relative_to(source)
            if path.suffix in (".pt", ".pkl"):
                if path.suffix == ".pt":
                    data = torch.load(path, map_location="cpu", weights_only=False)
                else:
                    with path.open("rb") as stream:
                        data = pickle.load(stream)
                try:
                    transformed, kind = crop_data(data, crops, config, camera_keys)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{relative}: {exc}") from exc
                manifest["files"].append({"path": str(relative), "kind": kind})
                print(f"[{kind}] {relative}")
                if staging is not None:
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix == ".pt":
                        torch.save(transformed, target)
                    else:
                        with target.open("wb") as stream:
                            pickle.dump(transformed, stream)
            elif staging is not None:
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        if staging is not None:
            (staging / "crop_manifest.json").write_text(json.dumps(manifest, indent=2))
            if destination.exists():
                destination.rmdir()
            staging.replace(destination)
        print(
            f"{'Validated (no writes)' if dry_run else 'Saved'} {len(manifest['files'])} datasets: {destination}"
        )
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return manifest


def main() -> None:
    """Read crop configuration without initializing cameras, Ray or a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--crop-config", required=True, type=Path)
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        help="Required for demos: main camera first, then other camera names",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    crop_directory(
        args.data_dir, args.output_dir, args.crop_config, args.camera_keys, args.dry_run
    )


if __name__ == "__main__":
    main()
