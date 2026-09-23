# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline reward review and crop tests; no cameras or GUI backend required."""

import argparse
import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.reward_model import RewardBinaryDataset, RewardDatasetPayload
from rlinf.data.reward_collection import split_reward_episodes
from toolkits.dexhand import crop_classifier_data as crop
from toolkits.dexhand import review_classifier_data as review

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["wrist_1", "global"]


def make_payload(path, episode=0, split=False):
    image = torch.arange(8 * 10 * 3, dtype=torch.uint8).reshape(8, 10, 3)
    images = [torch.stack([image, image.flip(0)]) for _ in range(4)]
    metadata = {
        "camera_keys": KEYS,
        "preprocessing": {"color_space": "RGB"},
        "step_ids": [10, 20, 30, 40],
    }
    metadata["episode_ids" if split else "episode_id"] = (
        [episode] * 4 if split else episode
    )
    payload = RewardDatasetPayload(images, [0, 1, 0, 1], metadata)
    payload.save(str(path))
    return payload


@pytest.mark.parametrize("split", [False, True])
def test_review_filters_identifiers_and_exports_unique_images(tmp_path, split):
    paths = [tmp_path / f"episode_{i:06d}.pt" for i in range(2)]
    original = [make_payload(path, i, split) for i, path in enumerate(paths)]
    session = review.ReviewSession(paths)
    assert session.visible(1) == [1, 3, 5, 7]
    assert session.visible(0) == [0, 2, 4, 6]
    session.mark(1, False)
    session.mark(2, True)
    destination = session.save(tmp_path / "clean")
    cleaned = RewardDatasetPayload.load(str(destination / paths[0].name))
    assert cleaned.labels == [0, 0, 1]
    assert cleaned.metadata["step_ids"] == [10, 30, 40]
    if split:
        assert cleaned.metadata["episode_ids"] == [0, 0, 0]
    assert cleaned.metadata["camera_keys"] == KEYS
    assert len(list((destination / "images").rglob("*.png"))) == 14
    image_path = destination / "images/failure/view_0/0000_00000000.png"
    np.testing.assert_array_equal(
        cv2.imread(str(image_path)), original[0].images[0][0].numpy()[..., ::-1]
    )
    assert RewardDatasetPayload.load(str(paths[0])).labels == original[0].labels
    assert not session.dirty
    assert (
        json.loads((destination / "review.json").read_text())["samples"][1]["keep"]
        is False
    )


def test_replace_inputs_creates_exact_backup(tmp_path):
    path = tmp_path / "episode_000000.pt"
    make_payload(path)
    before = path.read_bytes()
    session = review.ReviewSession([path])
    session.mark(0, False)
    backup = session.save(None, replace_inputs=True, export_images=False)
    assert (backup / path.name).read_bytes() == before
    assert RewardDatasetPayload.load(str(path)).labels == [1, 0, 1]


def test_review_render_and_controls(tmp_path, monkeypatch):
    path = tmp_path / "episode_000000.pt"
    make_payload(path)
    session = review.ReviewSession([path])
    assert review.render(session, [], 0, 40).shape == (165, 1000, 3)
    monkeypatch.setenv("DISPLAY", ":test")
    for method in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, method, lambda *args: None)
    keys = iter([ord("1"), ord("b"), ord("0"), ord("g"), ord("s"), ord("q")])
    monkeypatch.setattr(cv2, "waitKeyEx", lambda _: next(keys))
    monkeypatch.setattr(review, "confirm", lambda _: True)
    args = argparse.Namespace(
        output_dir=tmp_path / "out",
        replace_inputs=False,
        no_export_images=True,
        display_height=40,
    )
    review.run_ui(session, args)
    assert session.decisions == [True, False, None, None]
    assert len(list(args.output_dir.glob("review_*/*.pt"))) == 1


def test_find_inputs_and_invalid_payloads(tmp_path):
    path = tmp_path / "data.pt"
    make_payload(path)
    assert review.find_inputs([str(tmp_path), str(path)]) == [path]
    with pytest.raises(FileNotFoundError):
        review.find_inputs([str(tmp_path / "missing*")])
    torch.save({"curr_obs": {}}, path)
    with pytest.raises(ValueError, match="images/labels"):
        review.ReviewSession([path])


@pytest.mark.parametrize("tensor", [False, True])
def test_crop_batches_matches_online_resize(tensor):
    arr = np.arange(2 * 8 * 10 * 3, dtype=np.uint8).reshape(2, 8, 10, 3)
    value = torch.from_numpy(arr) if tensor else arr
    out = crop.crop_images(value, (0.25, 0.2, 1.0, 0.8))
    expected = cv2.resize(arr[0, 2:8, 2:8], (10, 8), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(np.asarray(out)[0], expected)
    assert out.shape == value.shape and out.dtype == value.dtype
    np.testing.assert_array_equal(value, arr)


def test_raw_coordinates_and_unrecoverable_region(tmp_path):
    config = {
        "coordinates": "raw",
        "cameras": {
            "wrist_1": {
                "source_region": [0.0, 0.125, 1.0, 0.875],
                "crop": [0.25, 0.3125, 0.75, 0.6875],
            }
        },
    }
    path = tmp_path / "crop.yaml"
    OmegaConf.save(config, path)
    _, regions = crop.load_crops(path)
    assert regions["wrist_1"] == (0.25, 0.25, 0.75, 0.75)
    config["cameras"]["wrist_1"]["crop"] = [0, 0, 1, 1]
    OmegaConf.save(config, path)
    with pytest.raises(ValueError, match="absent"):
        crop.load_crops(path)


def test_reward_crop_preserves_order_labels_and_split(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    for i in range(2):
        make_payload(source / f"episode_{i:06d}.pt", i)
    config = tmp_path / "crop.yaml"
    OmegaConf.save({"cameras": {"global": {"crop": [0.25, 0.2, 1.0, 0.8]}}}, config)
    out = tmp_path / "cropped"
    crop.crop_directory(source, out, config)
    original = RewardDatasetPayload.load(str(source / "episode_000000.pt"))
    result = RewardDatasetPayload.load(str(out / "episode_000000.pt"))
    assert result.labels == original.labels
    assert result.metadata["step_ids"] == original.metadata["step_ids"]
    torch.testing.assert_close(result.images[0][0], original.images[0][0])
    assert not torch.equal(result.images[0][1], original.images[0][1])
    split_reward_episodes(str(out), str(tmp_path / "split"), val_split=0.5)
    assert len(RewardBinaryDataset(str(tmp_path / "split/train.pt"), KEYS)) == 4


def test_demo_both_storage_formats_and_index_copy(tmp_path):
    source = tmp_path / "demos"
    source.mkdir()
    image = torch.arange(240, dtype=torch.uint8).reshape(8, 10, 3)
    obs = {
        "main_images": image.unsqueeze(0),
        "extra_view_images": image.unsqueeze(0).unsqueeze(0),
        "states": torch.ones(1, 12),
    }
    trajectory = {
        "curr_obs": obs,
        "next_obs": copy.deepcopy(obs),
        "actions": torch.ones(1, 12),
        "dones": torch.tensor([True]),
    }
    episode = {
        "observations": [obs, copy.deepcopy(obs)],
        "actions": [np.ones(12)],
        "success": True,
    }
    torch.save(trajectory, source / "trajectory_0.pt")
    with (source / "episode.pkl").open("wb") as f:
        pickle.dump(episode, f)
    (source / "trajectory_index.json").write_text('{"trajectory_id_list": [0]}')
    config = tmp_path / "crop.yaml"
    OmegaConf.save({"cameras": {"wrist_1": {"crop": [0, 0, 0.5, 0.5]}}}, config)
    out = tmp_path / "output"
    crop.crop_directory(source, out, config, KEYS, dry_run=True)
    assert not out.exists()
    crop.crop_directory(source, out, config, KEYS)
    changed = torch.load(out / "trajectory_0.pt", weights_only=False)
    for key in ("curr_obs", "next_obs"):
        assert not torch.equal(changed[key]["main_images"], obs["main_images"])
        torch.testing.assert_close(
            changed[key]["extra_view_images"], obs["extra_view_images"]
        )
        torch.testing.assert_close(changed[key]["states"], obs["states"])
    torch.testing.assert_close(changed["actions"], trajectory["actions"])
    assert (out / "trajectory_index.json").read_bytes() == (
        source / "trajectory_index.json"
    ).read_bytes()
    with (out / "episode.pkl").open("rb") as f:
        changed_episode = pickle.load(f)
    assert changed_episode["success"] and len(changed_episode["observations"]) == 2
    assert not torch.equal(
        changed_episode["observations"][-1]["main_images"], obs["main_images"]
    )
    with pytest.raises(FileExistsError):
        crop.crop_directory(source, out, config, KEYS)


def test_failure_does_not_publish_partial_output(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    make_payload(source / "a.pt")
    torch.save({"unknown": 1}, source / "z.pt")
    config = tmp_path / "crop.yaml"
    OmegaConf.save({"cameras": {"wrist_1": {"crop": [0, 0, 1, 1]}}}, config)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="z.pt"):
        crop.crop_directory(source, out, config)
    assert not out.exists() and not list(tmp_path.glob(".crop-*"))
    with pytest.raises(ValueError, match="overlap"):
        crop.crop_directory(source, source / "out", config)


@pytest.mark.parametrize("module", ["review_classifier_data", "crop_classifier_data"])
def test_cli_help_does_not_import_hardware(module):
    result = subprocess.run(
        [sys.executable, "-m", f"toolkits.dexhand.{module}", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout


@pytest.mark.parametrize("chw", [False, True])
def test_legacy_single_view_crop_and_review(tmp_path, chw):
    image = torch.arange(240, dtype=torch.uint8).reshape(8, 10, 3)
    if chw:
        image = image.permute(2, 0, 1)
    data = RewardDatasetPayload([image], [1], {}).to_dict()
    changed = crop.crop_reward(data, {"image": (0, 0, 0.5, 0.5)}, {})
    assert changed["images"][0].shape == image.shape
    path = tmp_path / "legacy.pt"
    torch.save(changed, path)
    assert review.ReviewSession([path]).camera_keys == [["image"]]


def test_demo_extra_camera_mapping_and_missing_camera():
    main = torch.zeros(2, 8, 10, 3, dtype=torch.uint8)
    extras = torch.arange(2 * 2 * 8 * 10 * 3).to(torch.uint8).reshape(2, 2, 8, 10, 3)
    obs = {"main_images": main, "extra_view_images": extras}
    changed = crop.crop_observation(obs, {"z": (0, 0, 0.5, 0.5)}, ["main", "z", "a"])
    torch.testing.assert_close(changed["extra_view_images"][:, 0], extras[:, 0])
    assert not torch.equal(changed["extra_view_images"][:, 1], extras[:, 1])
    with pytest.raises(ValueError, match="camera-keys"):
        crop.crop_observation(obs, {"unknown": (0, 0, 1, 1)}, ["main", "a", "z"])


def test_review_cli_dry_run(tmp_path):
    path = tmp_path / "episode_000000.pt"
    make_payload(path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolkits.dexhand.review_classifier_data",
            "--input",
            str(path),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "success=2, failure=2" in result.stdout
    assert list(tmp_path.iterdir()) == [path]
