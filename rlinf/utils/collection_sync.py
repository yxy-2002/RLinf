# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Replicate finalized collection files between Ray nodes without SSH."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

CHUNK_BYTES = 8 * 1024 * 1024


def _collection_files_type() -> type:
    """Define a self-contained actor that Ray can ship to older remote checkouts."""

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()

    class CollectionFiles:
        """Serve or receive files restricted to the declared output directories.

        A receiver writes one file at a time. Only a complete, verified temporary
        file is published; existing different files are never overwritten.
        """

        def __init__(self, directories: list[str]):
            if not directories or any(not Path(p).is_absolute() for p in directories):
                raise ValueError("Collection sync requires absolute output directories")
            self.roots = [Path(p).resolve() for p in directories]
            self._pending = None

        def _path(self, root: int, relative: str) -> Path:
            path = self.roots[root] / relative
            if not path.resolve().is_relative_to(self.roots[root]):
                raise ValueError(f"File escapes collection directory: {relative}")
            return path

        def manifest(self) -> list[dict[str, Any]]:
            """Return sizes and SHA-256 hashes after the producer has closed writers."""
            result = []
            for index, root in enumerate(self.roots):
                if not root.is_dir():
                    raise FileNotFoundError(f"Collection directory is missing: {root}")
                for path in sorted(root.rglob("*")):
                    if path.is_symlink():
                        raise ValueError(
                            f"Collection sync does not follow symlinks: {path}"
                        )
                    if path.is_file():
                        result.append(
                            {
                                "root": index,
                                "relative": str(path.relative_to(root)),
                                "size": path.stat().st_size,
                                "sha256": _sha256(path),
                            }
                        )
            return result

        def read_chunk(self, entry: dict, offset: int) -> bytes:
            """Read one bounded chunk from a finalized source file."""
            with self._path(entry["root"], entry["relative"]).open("rb") as stream:
                stream.seek(offset)
                return stream.read(CHUNK_BYTES)

        def begin(self, entry: dict) -> bool:
            """Prepare a receive; return false when an identical file already exists."""
            if self._pending is not None:
                raise RuntimeError("Previous collection transfer has not finished")
            destination = self._path(entry["root"], entry["relative"])
            if destination.exists():
                if (
                    destination.stat().st_size == entry["size"]
                    and _sha256(destination) == entry["sha256"]
                ):
                    return False
                raise FileExistsError(
                    f"Collection sync destination differs: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=".collection-sync-",
                delete=False,
            )
            self._pending = (stream, destination, dict(entry), hashlib.sha256())
            return True

        def append(self, offset: int, chunk: bytes) -> None:
            """Write an ordered chunk, rejecting gaps and excess bytes."""
            if self._pending is None:
                raise RuntimeError("No collection transfer is active")
            stream, _, entry, digest = self._pending
            if stream.tell() != offset or len(chunk) != min(
                CHUNK_BYTES, entry["size"] - offset
            ):
                raise ValueError("Invalid collection chunk offset or length")
            stream.write(chunk)
            digest.update(chunk)

        def finish(self) -> None:
            """Publish the verified file atomically without replacing an existing file."""
            if self._pending is None:
                raise RuntimeError("No collection transfer is active")
            stream, destination, entry, digest = self._pending
            try:
                if (
                    stream.tell() != entry["size"]
                    or digest.hexdigest() != entry["sha256"]
                ):
                    raise ValueError(f"Collection checksum mismatch: {destination}")
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                # Both names are on the same filesystem; link fails if another
                # writer created the destination while we were transferring.
                os.link(stream.name, destination)
            finally:
                self.abort()

        def abort(self) -> None:
            """Remove any incomplete temporary file, leaving finalized data intact."""
            if self._pending is not None:
                stream = self._pending[0]
                stream.close()
                Path(stream.name).unlink(missing_ok=True)
                self._pending = None

    return CollectionFiles


CollectionFiles = _collection_files_type()


def sync_collection_data(outputs: list[dict], cluster: Any) -> None:
    """Copy closed collector outputs to the same absolute paths on cluster nodes.

    Args:
        outputs: Each collector's Ray node ID and absolute output directories.
        cluster: Initialized RLinf cluster, used for node IDs and interpreters.

    Raises:
        Exception: Transfer, checksum or conflicting-file errors. Source files
            are never removed, so a failed transfer can be retried.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from rlinf.utils.logging import get_logger

    logger = get_logger()
    nodes = [cluster.get_node_info(rank) for rank in range(cluster.num_nodes)]
    remote_files = ray.remote(CollectionFiles)

    def endpoint(node, paths):
        return remote_files.options(
            num_cpus=0,
            max_restarts=0,
            runtime_env={"py_executable": node.python_interpreter_path},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node.ray_id, soft=False),
        ).remote(paths)

    def wait(reference):
        return ray.get(reference, timeout=300)

    for output in outputs:
        source_node = next(n for n in nodes if n.ray_id == output["node_id"])
        targets = [n for n in nodes if n.ray_id != source_node.ray_id]
        if not targets:
            continue
        source = endpoint(source_node, output["directories"])
        try:
            files = wait(source.manifest.remote())
            for node in targets:
                target = endpoint(node, output["directories"])
                copied = skipped = 0
                try:
                    logger.info(
                        "[collection sync] Node %s -> %s: %s files, same paths %s",
                        source_node.node_rank,
                        node.node_rank,
                        len(files),
                        output["directories"],
                    )
                    for entry in files:
                        if not wait(target.begin.remote(entry)):
                            skipped += 1
                            continue
                        for offset in range(0, entry["size"], CHUNK_BYTES):
                            chunk = source.read_chunk.remote(entry, offset)
                            wait(target.append.remote(offset, chunk))
                        wait(target.finish.remote())
                        copied += 1
                    logger.info(
                        "[collection sync] Node %s complete: %s copied, %s identical; SHA-256 verified",
                        node.node_rank,
                        copied,
                        skipped,
                    )
                finally:
                    try:
                        ray.get(target.abort.remote(), timeout=10)
                    finally:
                        ray.kill(target, no_restart=True)
        finally:
            ray.kill(source, no_restart=True)
