# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the read-only trajectory player and its HTTP contract."""

import base64
import io
import json
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pytest
import torch
from PIL import Image

from toolkits.replay_buffer.web_visualizer import (
    Trajectory,
    Workspace,
    encode_image,
    make_server,
)


@pytest.fixture
def recording(tmp_path):
    """Build distinguishable frames and batches for alignment assertions."""
    images = torch.zeros(4, 2, 8, 10, 3, dtype=torch.uint8)
    for frame in range(4):
        for batch in range(2):
            images[frame, batch, ..., 0] = frame * 40 + batch * 10
    data = {
        "actions": torch.arange(24).reshape(4, 2, 3).float(),
        "rewards": torch.arange(8).reshape(4, 2, 1).float(),
        "dones": torch.zeros(4, 2, 1, dtype=torch.bool),
        "curr_obs": {
            "main_images": images,
            "extra_view_images": images.unsqueeze(2).expand(-1, -1, 2, -1, -1, -1),
            "states": torch.arange(40).reshape(4, 2, 5).float(),
        },
        "next_obs": {"main_images": images + 1},
        "model_weights_id": "test-model",
    }
    folder = tmp_path / "demo"
    folder.mkdir()
    path = folder / "trajectory_2_test.pt"
    torch.save(data, path)
    return path, data


def test_frame_batch_and_camera_alignment(recording):
    path, data = recording
    trajectory = Trajectory(path)
    assert trajectory.info()["length"] == 4
    assert trajectory.info()["batches"] == 2
    frame = trajectory.frame(2, 1, "curr_obs")
    assert frame["values"]["actions"] == data["actions"][2, 1].tolist()
    assert (
        frame["values"]["curr_obs/states"] == data["curr_obs"]["states"][2, 1].tolist()
    )
    assert frame["metadata"]["model_weights_id"] == "test-model"
    assert len(frame["images"]) == 3
    encoded = frame["images"]["curr_obs/main_images"].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert image.size == (10, 8)
    assert image.format == "PNG"
    np.testing.assert_array_equal(
        np.asarray(image), data["curr_obs"]["main_images"][2, 1].numpy()
    )
    assert list(trajectory.frame(2, 1, "next_obs")["images"]) == [
        "next_obs/main_images"
    ]
    assert trajectory.series("actions", 1, 2)["points"] == [
        [i, data["actions"][i, 1, 2].item()] for i in range(4)
    ]
    assert trajectory.series("actions", 1, 2)["dimensions"] == 3


@pytest.mark.parametrize(
    "channels,chw", [(3, False), (3, True), (4, False), (1, False)]
)
def test_full_resolution_png_preserves_all_pixels(channels, chw):
    """Catch both lossy encoding and the former 960x720 thumbnail limit."""
    pixels = torch.randint(
        0,
        256,
        (800, 1280, channels),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(42),
    )
    source = pixels.permute(2, 0, 1) if chw else pixels
    encoded = encode_image(source)
    assert encoded.startswith("data:image/png;base64,")
    image = Image.open(io.BytesIO(base64.b64decode(encoded.split(",")[1])))
    assert image.format == "PNG"
    assert image.size == (1280, 800)
    expected = pixels.numpy()
    if channels == 1:
        expected = expected[..., 0]
    np.testing.assert_array_equal(np.asarray(image), expected)


@pytest.mark.parametrize(
    "index,batch,observation",
    [(-1, 0, "curr_obs"), (4, 0, "curr_obs"), (0, 2, "curr_obs"), (0, 0, "unknown")],
)
def test_invalid_selection(recording, index, batch, observation):
    with pytest.raises(ValueError):
        Trajectory(recording[0]).frame(index, batch, observation)


def test_unbatched_chw_float_and_nonfinite(tmp_path):
    path = tmp_path / "trajectory_0.pt"
    images = torch.ones(3, 3, 8, 10)
    torch.save(
        {
            "actions": torch.zeros(3, 2),
            "curr_obs": {"rgb": images},
            "rewards": torch.tensor([float("nan"), float("inf"), 1.0]),
        },
        path,
    )
    trajectory = Trajectory(path)
    frame = trajectory.frame(0, 0, "curr_obs")
    assert trajectory.batches == 1
    assert frame["values"]["rewards"] == "nan"
    image = Image.open(
        io.BytesIO(base64.b64decode(frame["images"]["curr_obs/rgb"].split(",")[1]))
    )
    assert image.size == (10, 8)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    json.dumps(frame, allow_nan=False)


def test_scan_deduplicates_and_sorts_naturally(recording, tmp_path):
    path, data = recording
    torch.save(data, path.with_name("trajectory_10_test.pt"))
    (tmp_path / "empty").mkdir()
    workspace = Workspace(tmp_path)
    assert workspace.browse(".")["directories"] == ["demo", "empty"]
    found = workspace.scan(["demo", ".", str(path.parent)])["trajectories"]
    assert [item["name"] for item in found] == [
        "trajectory_2_test.pt",
        "trajectory_10_test.pt",
    ]
    assert workspace.scan(["empty"])["trajectories"] == []
    with pytest.raises(ValueError, match="at least one"):
        workspace.scan([])


def test_workspace_rejects_escape_and_symlinks(recording, tmp_path):
    root = tmp_path / "restricted"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path / "demo", target_is_directory=True)
    (root / "trajectory_external.pt").symlink_to(recording[0])
    workspace = Workspace(root)
    for path in ("..", "outside", "trajectory_external.pt", str(recording[0])):
        with pytest.raises(PermissionError):
            workspace.resolve(path)
    assert workspace.browse(".")["directories"] == []
    assert workspace.scan(["."])["trajectories"] == []


def test_cache_reload_and_invalid_files(recording, tmp_path):
    path, data = recording
    workspace = Workspace(tmp_path)
    first = workspace.load(str(path))
    assert workspace.load(str(path)) is first
    data["actions"] = torch.ones(4, 2, 6)
    torch.save(data, path)
    assert workspace.load(str(path)).info()["fields"]["actions"]["shape"] == [4, 2, 6]
    bad = tmp_path / "trajectory_bad.pt"
    torch.save({"weights": torch.ones(3)}, bad)
    with pytest.raises(ValueError, match="actions"):
        workspace.load(str(bad))
    assert workspace.load(str(path)).length == 4


def test_http_api_assets_errors_and_read_only(recording, tmp_path):
    server = make_server(Workspace(tmp_path), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            assert response.status == 200
            assert "真机数据播放器" in response.read().decode()
        for asset in ("app.js", "style.css"):
            with urlopen(f"{base}/{asset}") as response:
                assert response.status == 200
        query = urlencode({"path": str(recording[0]), "index": 3, "batch": 1})
        with urlopen(f"{base}/api/frame?{query}") as response:
            frame = json.load(response)
            assert frame["index"] == 3
            assert frame["values"]["actions"] == [21, 22, 23]
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base}/api/browse?path=..")
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base}/api/frame?{query}&observation=bad")
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base}/api/unknown")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_numeric_only_large_values_and_plot_bounds(tmp_path):
    path = tmp_path / "trajectory_numeric.pt"
    torch.save(
        {
            "actions": torch.arange(5001).reshape(5001, 1, 1),
            "states": torch.zeros(5001, 1, 1050),
        },
        path,
    )
    trajectory = Trajectory(path)
    frame = trajectory.frame(5000, 0, "curr_obs")
    assert frame["images"] == {}
    assert frame["values"]["states"]["truncated"]
    assert len(frame["values"]["states"]["preview"]) == 1024
    points = trajectory.series("actions", 0, 0)["points"]
    assert len(points) == 4000
    assert points[0] == [0, 0]
    assert points[-1] == [5000, 5000]
    with pytest.raises(ValueError, match="Dimension"):
        trajectory.series("actions", 0, 1)
    with pytest.raises(ValueError, match="Batch"):
        trajectory.series("actions", 1, 0)
    with pytest.raises(ValueError, match="numeric"):
        trajectory.series("missing", 0, 0)


def test_corrupt_recording_does_not_prevent_other_loads(recording, tmp_path):
    broken = tmp_path / "trajectory_broken.pt"
    broken.write_bytes(b"incomplete recording")
    workspace = Workspace(tmp_path)
    with pytest.raises(Exception):
        workspace.load(str(broken))
    assert workspace.load(str(recording[0])).length == 4
