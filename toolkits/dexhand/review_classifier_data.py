# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Review native RLinf reward payloads without importing robot environments."""

from __future__ import annotations

import argparse
import copy
import datetime
import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from rlinf.data.datasets.reward_model import RewardDatasetPayload


def find_inputs(values: list[str]) -> list[Path]:
    """Expand files, directories and glob patterns, rejecting missing inputs."""
    paths = []
    for value in values:
        matches = sorted(glob.glob(value))
        if not matches:
            raise FileNotFoundError(value)
        for match in matches:
            path = Path(match)
            paths.extend(sorted(path.glob("*.pt")) if path.is_dir() else [path])
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not paths:
        raise ValueError("No .pt reward files found")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Input filenames must be unique; review each run separately")
    return paths


def image_views(image: torch.Tensor) -> torch.Tensor:
    """Return uint8 RGB views as VHWC, including legacy single-view data."""
    image = torch.as_tensor(image)
    if image.ndim == 3:
        if image.shape[-1] != 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != torch.uint8:
        raise ValueError(
            f"Expected uint8 RGB HWC/CHW or VHWC, got {image.shape}, {image.dtype}"
        )
    return image


def validate_payload(payload: RewardDatasetPayload) -> list[str]:
    """Validate view order, binary labels and aligned sample identifiers."""
    if any(label not in (0, 1) for label in payload.labels):
        raise ValueError("Reward labels must be binary")
    for key in ("step_ids", "episode_ids"):
        if key in payload.metadata and len(payload.metadata[key]) != len(
            payload.images
        ):
            raise ValueError(f"metadata.{key} must align with images")
    keys = payload.metadata.get("camera_keys")
    if not keys:
        if payload.images and image_views(payload.images[0]).shape[0] != 1:
            raise ValueError("Multi-view payloads require metadata.camera_keys")
        keys = ["image"]
    if len(keys) != len(set(keys)) or not all(isinstance(key, str) for key in keys):
        raise ValueError("camera_keys must contain unique names")
    for image in payload.images:
        if image_views(image).shape[0] != len(keys):
            raise ValueError("Image view count differs from camera_keys")
    return list(keys)


def filter_payload(
    payload: RewardDatasetPayload, indices: list[int]
) -> RewardDatasetPayload:
    """Filter samples and their identifiers without changing labels or layout."""
    metadata = copy.deepcopy(payload.metadata)
    for key in ("step_ids", "episode_ids"):
        if key in metadata:
            metadata[key] = [metadata[key][i] for i in indices]
    return RewardDatasetPayload(
        [payload.images[i] for i in indices],
        [payload.labels[i] for i in indices],
        metadata,
    )


class ReviewSession:
    """Keep review decisions separate from immutable source payloads."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.payloads = []
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(data, dict) or not {"images", "labels"} <= data.keys():
                raise ValueError(f"{path}: expected a reward images/labels payload")
            self.payloads.append(RewardDatasetPayload.from_dict(data, str(path)))
        self.camera_keys = [validate_payload(payload) for payload in self.payloads]
        self.samples = [
            (f, i)
            for f, payload in enumerate(self.payloads)
            for i in range(len(payload.images))
        ]
        if not self.samples:
            raise ValueError("No frames to review")
        self.decisions: list[bool | None] = [None] * len(self.samples)
        self.dirty = False

    def visible(self, label: int | None) -> list[int]:
        """Return global frame indices matching the current label filter."""
        return [
            j
            for j, (f, i) in enumerate(self.samples)
            if label is None or self.payloads[f].labels[i] == label
        ]

    def mark(self, index: int, keep: bool) -> None:
        """Mark one frame without removing it from the navigation order."""
        self.decisions[index] = keep
        self.dirty = True

    def save(
        self,
        output_dir: Path | None,
        replace_inputs: bool = False,
        export_images: bool = True,
    ) -> Path:
        """Write a snapshot; back up every input before explicit replacement."""
        if not replace_inputs and output_dir is None:
            raise ValueError("Specify output_dir or replace_inputs")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        root = self.paths[0].parent if replace_inputs else output_dir
        destination = root / (
            f"review_backup_{stamp}" if replace_inputs else f"review_{stamp}"
        )
        destination.mkdir(parents=True, exist_ok=False)
        if replace_inputs:
            for path in self.paths:
                shutil.copy2(path, destination / path.name)
        kept = [[] for _ in self.paths]
        manifest = []
        for (file_idx, sample_idx), decision in zip(
            self.samples, self.decisions, strict=True
        ):
            manifest.append({
                "source": str(self.paths[file_idx]),
                "sample_index": sample_idx,
                "keep": decision,
            })
            if decision is not False:
                kept[file_idx].append(sample_idx)
        for file_idx, (path, payload) in enumerate(
            zip(self.paths, self.payloads, strict=True)
        ):
            cleaned = filter_payload(payload, kept[file_idx])
            target = path if replace_inputs else destination / path.name
            temporary = target.with_suffix(".pt.tmp")
            cleaned.save(str(temporary))
            temporary.replace(target)
            if export_images:
                for sample_idx in kept[file_idx]:
                    label = "success" if payload.labels[sample_idx] else "failure"
                    for view_idx, view in enumerate(
                        image_views(payload.images[sample_idx])
                    ):
                        # Camera index is filename-safe; names are recorded in the manifest.
                        folder = destination / "images" / label / f"view_{view_idx}"
                        folder.mkdir(parents=True, exist_ok=True)
                        image_path = folder / f"{file_idx:04d}_{sample_idx:08d}.png"
                        if not cv2.imwrite(str(image_path), view.numpy()[..., ::-1]):
                            raise OSError(f"Cannot write {image_path}")
        (destination / "review.json").write_text(
            json.dumps({"camera_keys": self.camera_keys, "samples": manifest}, indent=2)
        )
        self.dirty = False
        print(
            f"Saved {sum(map(len, kept))}/{len(self.samples)} frames; {'backup' if replace_inputs else 'output'}: {destination}"
        )
        return destination


def confirm(message: str) -> bool:
    """Ask for a decision inside the existing OpenCV display session."""
    window = "Confirm"
    canvas = np.zeros((150, 900, 3), dtype=np.uint8)
    cv2.putText(
        canvas, message, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )
    cv2.putText(
        canvas,
        "Y: yes   N / Esc: no",
        (15, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 255, 100),
        1,
    )
    cv2.imshow(window, canvas)
    try:
        while True:
            key = cv2.waitKeyEx(100)
            if key == ord("y"):
                return True
            if (
                key in (ord("n"), 27)
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            ):
                return False
    finally:
        cv2.destroyWindow(window)


def render(
    session: ReviewSession, visible: list[int], cursor: int, height: int
) -> np.ndarray:
    """Compose all camera views and review statistics in BGR for OpenCV."""
    parts = []
    if visible:
        index = visible[cursor]
        file_idx, sample_idx = session.samples[index]
        payload = session.payloads[file_idx]
        for name, image in zip(
            session.camera_keys[file_idx],
            image_views(payload.images[sample_idx]),
            strict=True,
        ):
            arr = image.numpy()[..., ::-1]
            width = max(1, round(arr.shape[1] * height / arr.shape[0]))
            part = np.zeros((height + 28, width, 3), dtype=np.uint8)
            part[28:] = cv2.resize(arr, (width, height))
            cv2.putText(
                part, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 180), 1
            )
            parts.append(part)
        tag = {None: "UNREVIEWED", True: "KEEP", False: "DISCARD"}[
            session.decisions[index]
        ]
        heading = f"{cursor + 1}/{len(visible)} label={payload.labels[sample_idx]} {tag} {session.paths[file_idx].name}:{sample_idx}"
    else:
        heading = "No samples in this filter"
    width = max(1000, sum(part.shape[1] for part in parts))
    canvas = np.zeros((height + 125, width, 3), dtype=np.uint8)
    x = 0
    for part in parts:
        canvas[95 : 95 + part.shape[0], x : x + part.shape[1]] = part
        x += part.shape[1]
    reviewed = sum(d is not None for d in session.decisions)
    discarded = sum(d is False for d in session.decisions)
    lines = [
        heading,
        f"reviewed={reviewed}/{len(session.samples)} keep={len(session.samples) - discarded} discard={discarded}",
        "n/p: next/prev | g/b: keep/discard | 1/2/0: positive/negative/all | s: save | q: quit",
    ]
    for y, line in zip((22, 48, 74), lines, strict=True):
        cv2.putText(
            canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1
        )
    return canvas


def run_ui(session: ReviewSession, args: argparse.Namespace) -> None:
    """Run interactive review, keeping unsaved changes on cancelled saves."""
    if os.name == "posix" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "A graphical desktop is required; use --dry-run for headless inspection"
        )
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")
    window = "Review classifier data"
    visible, cursor = session.visible(None), 0

    def save() -> bool:
        if not confirm(
            "Back up and replace inputs?"
            if args.replace_inputs
            else "Save reviewed snapshot? Unreviewed frames are kept."
        ):
            return False
        session.save(args.output_dir, args.replace_inputs, not args.no_export_images)
        return True

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            cv2.imshow(window, render(session, visible, cursor, args.display_height))
            key = cv2.waitKeyEx(100)
            if key in (ord("n"), 83, 65363, 2555904):
                cursor = min(cursor + 1, max(0, len(visible) - 1))
            elif key in (ord("p"), 81, 65361, 2424832):
                cursor = max(0, cursor - 1)
            elif key in (ord("g"), ord("b")) and visible:
                session.mark(visible[cursor], key == ord("g"))
                cursor = min(cursor + 1, len(visible) - 1)
            elif key in (ord("0"), ord("1"), ord("2")):
                visible = session.visible(
                    {ord("0"): None, ord("1"): 1, ord("2"): 0}[key]
                )
                cursor = 0
            elif key == ord("s"):
                save()
            elif (
                key in (ord("q"), 27)
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            ):
                if session.dirty and confirm("Unsaved changes. Save before quitting?"):
                    if not save():
                        continue
                break
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    """Parse command-line options and review native reward datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Reward .pt files, directories or quoted globs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reviewed_reward_data"))
    parser.add_argument(
        "--replace-inputs",
        action="store_true",
        help="Back up originals before replacing them",
    )
    parser.add_argument("--no-export-images", action="store_true")
    parser.add_argument("--display-height", type=int, default=360)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.display_height < 1:
        parser.error("--display-height must be positive")
    session = ReviewSession(find_inputs(args.input))
    positives = sum(sum(payload.labels) for payload in session.payloads)
    print(
        f"Loaded {len(session.samples)} frames: success={positives}, failure={len(session.samples) - positives}; cameras={session.camera_keys}"
    )
    if not args.dry_run:
        run_ui(session, args)


if __name__ == "__main__":
    main()
