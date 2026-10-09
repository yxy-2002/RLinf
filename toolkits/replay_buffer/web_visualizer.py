# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Serve a local, read-only player for recorded RLinf trajectories."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)
ASSETS = Path(__file__).with_name("web")
MAX_FILES = 10000


def flatten(data: dict, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield nested dictionary leaves with slash-separated names."""
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from flatten(value, name)
        else:
            yield name, value


def json_value(value: Any) -> Any:
    """Convert tensors to bounded, strictly JSON-compatible display values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() > 1024:
            return {
                "shape": list(value.shape),
                "preview": json_value(value.reshape(-1)[:1024]),
                "truncated": True,
            }
        return json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value[:1024]]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def is_image(name: str, value: Any) -> bool:
    """Recognize recorded image tensors without confusing state vectors."""
    return (
        isinstance(value, torch.Tensor)
        and value.ndim >= 4
        and any(token in name.lower() for token in ("image", "img", "rgb", "camera"))
        and (value.shape[-1] in (1, 3, 4) or value.shape[-3] in (1, 3, 4))
    )


def encode_image(value: torch.Tensor) -> str:
    """Encode a full-resolution PNG, preserving every uint8 pixel exactly."""
    pixels = value.detach().cpu()
    if pixels.shape[-1] not in (1, 3, 4):
        pixels = pixels.permute(1, 2, 0)
    if pixels.dtype == torch.uint8:
        array = pixels.numpy()
    else:
        array = pixels.float().numpy()
        if pixels.is_floating_point() and array.size and np.nanmax(array) <= 1:
            array = array * 255
        array = np.nan_to_num(array).clip(0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    image = Image.fromarray(array)
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class Trajectory:
    """Read one tensor dictionary, keeping modern torch archives memory mapped."""

    def __init__(self, path: Path):
        try:
            self.data = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except RuntimeError as exc:
            if "mmap" not in str(exc):
                raise
            self.data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(self.data, dict):
            raise ValueError("Expected a trajectory tensor dictionary")
        actions = self.data.get("actions")
        if not isinstance(actions, torch.Tensor) or actions.ndim < 2:
            raise ValueError("Missing actions tensor with shape [T,B,...] or [T,A]")
        self.length = actions.shape[0]
        self.batched = actions.ndim >= 3
        self.batches = actions.shape[1] if self.batched else 1
        if not self.length or not self.batches:
            raise ValueError("Trajectory has no frames or batches")
        self.leaves = dict(flatten(self.data))
        self.images = {k for k, v in self.leaves.items() if is_image(k, v)}

    def _temporal(self, value: Any) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] in (self.length, self.length + 1)
        )

    def _sample(self, value: Any, frame: int, batch: int) -> Any:
        if self._temporal(value):
            if self.batched and value.ndim > 1 and value.shape[1] == self.batches:
                return value[frame, batch]
            return value[frame]
        return value

    def info(self) -> dict:
        """Describe the trajectory and numeric fields available for plotting."""
        return {
            "length": self.length,
            "batches": self.batches,
            "fields": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in self.leaves.items()
                if isinstance(value, torch.Tensor)
            },
            "plot_fields": [
                key
                for key, value in self.leaves.items()
                if key not in self.images and self._temporal(value)
            ],
            "image_fields": sorted(self.images),
        }

    def frame(self, index: int, batch: int, observation: str) -> dict:
        """Return aligned camera frames, numeric data, and static metadata."""
        if not 0 <= index < self.length or not 0 <= batch < self.batches:
            raise ValueError("Frame or batch is outside this trajectory")
        if observation not in ("curr_obs", "next_obs"):
            raise ValueError("Observation must be curr_obs or next_obs")
        images, values, metadata = {}, {}, {}
        for key, value in self.leaves.items():
            sample = self._sample(value, index, batch)
            if key in self.images:
                if key.startswith(("curr_obs/", "next_obs/")) and not key.startswith(
                    observation + "/"
                ):
                    continue
                if sample.ndim < 3:
                    continue
                views = sample.reshape(-1, *sample.shape[-3:])
                for view, pixels in enumerate(views):
                    name = key if sample.ndim == 3 else f"{key}[{view}]"
                    images[name] = encode_image(pixels)
            elif self._temporal(value):
                values[key] = json_value(sample)
            else:
                metadata[key] = json_value(sample)
        return {
            "index": index,
            "images": images,
            "values": values,
            "metadata": metadata,
        }

    def series(self, field: str, batch: int, dimension: int) -> dict:
        """Return one scalar dimension over time, bounded to 4000 plot points."""
        if field not in self.info()["plot_fields"]:
            raise ValueError("Choose a numeric time-series field")
        if not 0 <= batch < self.batches:
            raise ValueError("Batch is outside this trajectory")
        value = self.leaves[field][: self.length]
        if self.batched and value.ndim > 1 and value.shape[1] == self.batches:
            value = value[:, batch]
        value = value.reshape(self.length, -1)
        if not 0 <= dimension < value.shape[1]:
            raise ValueError("Dimension is outside this field")
        indices = np.unique(
            np.linspace(0, self.length - 1, min(self.length, 4000)).astype(int)
        )
        return {
            "dimensions": value.shape[1],
            "points": [[int(i), json_value(value[i, dimension])] for i in indices],
        }


class Workspace:
    """Restrict browsing to one workspace and cache only the current trajectory."""

    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory")
        self.lock = threading.RLock()
        self.cached: tuple | None = None
        self.trajectory: Trajectory | None = None

    def resolve(self, name: str) -> Path:
        """Resolve a user-selected path without allowing workspace escapes."""
        path = (self.root / name).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise PermissionError("Path is outside the configured workspace")
        return path

    def browse(self, name: str) -> dict:
        """List immediate child directories, including hidden recording folders."""
        path = self.resolve(name)
        directories = []
        for child in path.iterdir():
            if child.is_dir() and child.resolve().is_relative_to(self.root):
                directories.append(str(child.relative_to(self.root)))
        return {
            "root": str(self.root),
            "path": str(path.relative_to(self.root)),
            "parent": str(path.parent.relative_to(self.root))
            if path != self.root
            else None,
            "directories": sorted(directories, key=str.casefold),
        }

    def scan(self, names: list[str]) -> dict:
        """Discover trajectory archives recursively, deduplicating overlapping roots."""
        if not names:
            raise ValueError("Select at least one directory")
        found: dict[str, dict] = {}
        warnings = []
        visited = set()
        for name in names:
            start = self.resolve(name)
            if not start.is_dir():
                raise ValueError("Select directories, not files")
            for directory, dirs, files in os.walk(
                start, onerror=lambda exc: warnings.append(str(exc)), followlinks=False
            ):
                if directory in visited:
                    dirs[:] = []
                    continue
                visited.add(directory)
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in {".git", "node_modules", ".venv", "__pycache__"}
                ]
                for filename in files:
                    if not filename.startswith("trajectory_") or not filename.endswith(
                        ".pt"
                    ):
                        continue
                    path = Path(directory) / filename
                    if not path.resolve().is_relative_to(self.root):
                        continue
                    relative = str(path.relative_to(self.root))
                    found[relative] = {
                        "path": relative,
                        "directory": str(path.parent.relative_to(self.root)),
                        "name": filename,
                        "bytes": path.stat().st_size,
                    }
                    if len(found) >= MAX_FILES:
                        warnings.append(
                            f"Showing the first {MAX_FILES} files; select narrower directories."
                        )
                        return {
                            "trajectories": list(found.values()),
                            "warnings": warnings,
                        }

        def natural(item: dict) -> list:
            return [
                int(p) if p.isdigit() else p for p in re.split(r"(\d+)", item["path"])
            ]

        return {
            "trajectories": sorted(found.values(), key=natural),
            "warnings": warnings,
        }

    def load(self, name: str) -> Trajectory:
        """Load a trajectory on demand; callers hold the workspace lock."""
        path = self.resolve(name)
        if not path.name.startswith("trajectory_") or path.suffix != ".pt":
            raise ValueError("Select a trajectory_*.pt recording")
        stat = path.stat()
        key = (path, stat.st_mtime_ns, stat.st_size)
        if key != self.cached:
            self.trajectory = None
            self.cached = None
            self.trajectory = Trajectory(path)
            self.cached = key
        return self.trajectory


def make_server(workspace: Workspace, host: str, port: int) -> ThreadingHTTPServer:
    """Create the local HTTP server without starting its event loop."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Serve bundled assets and the read-only data API."""
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)

                def arg(name: str, default: str = "") -> str:
                    return query.get(name, [default])[0]

                if parsed.path in ("/", "/app.js", "/style.css"):
                    filename, mime = {
                        "/": ("index.html", "text/html"),
                        "/app.js": ("app.js", "text/javascript"),
                        "/style.css": ("style.css", "text/css"),
                    }[parsed.path]
                    self.respond(200, (ASSETS / filename).read_bytes(), mime)
                    return
                if parsed.path == "/api/browse":
                    data = workspace.browse(arg("path", "."))
                elif parsed.path == "/api/scan":
                    data = workspace.scan(query.get("path", []))
                elif parsed.path in ("/api/info", "/api/frame", "/api/series"):
                    with workspace.lock:
                        trajectory = workspace.load(arg("path"))
                        if parsed.path == "/api/info":
                            data = trajectory.info()
                        elif parsed.path == "/api/frame":
                            data = trajectory.frame(
                                int(arg("index", "0")),
                                int(arg("batch", "0")),
                                arg("observation", "curr_obs"),
                            )
                        else:
                            data = trajectory.series(
                                arg("field"),
                                int(arg("batch", "0")),
                                int(arg("dimension", "0")),
                            )
                else:
                    self.respond(404, b'{"error":"Not found"}', "application/json")
                    return
                self.respond(
                    200,
                    json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
                    "application/json",
                )
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                LOGGER.warning("Viewer request failed: %s", exc)
                status = 403 if isinstance(exc, PermissionError) else 400
                self.respond(
                    status,
                    json.dumps({"error": str(exc)}, ensure_ascii=False).encode(),
                    "application/json",
                )

        def respond(self, status: int, payload: bytes, mime: str) -> None:
            """Write a complete response without permitting cross-origin reads."""
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.debug(format, *args)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    """Run the viewer with no Ray, GPU, or frontend build step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Browsable workspace root (default: current directory)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: localhost)"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    server = make_server(Workspace(args.root), args.host, args.port)
    LOGGER.info(
        "Trajectory viewer: http://%s:%s — workspace: %s",
        args.host,
        server.server_port,
        args.root.resolve(),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping trajectory viewer")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
