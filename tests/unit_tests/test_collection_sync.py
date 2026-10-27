# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Verify collection replication preserves complete, checksummed files."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rlinf.utils import collection_sync
from rlinf.utils.collection_sync import CollectionFiles


def transfer(source, target, entry):
    if target.begin(entry):
        for offset in range(0, entry["size"], collection_sync.CHUNK_BYTES):
            target.append(offset, source.read_chunk(entry, offset))
        target.finish()


def test_chunked_sync_idempotence_and_existing_file_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(collection_sync, "CHUNK_BYTES", 7)
    source_dir, target_dir = tmp_path / "source", tmp_path / "target"
    source_dir.mkdir()
    (source_dir / "trajectory.pt").write_bytes(bytes(range(100)))
    (source_dir / "empty").touch()
    source, target = (
        CollectionFiles([str(source_dir)]),
        CollectionFiles([str(target_dir)]),
    )
    entries = source.manifest()
    for entry in entries:
        transfer(source, target, entry)
        assert not target.begin(entry)
        assert (source_dir / entry["relative"]).read_bytes() == (
            target_dir / entry["relative"]
        ).read_bytes()
    (target_dir / "trajectory.pt").write_bytes(b"existing data")
    with pytest.raises(FileExistsError, match="differs"):
        target.begin(next(e for e in entries if e["relative"] == "trajectory.pt"))
    assert (target_dir / "trajectory.pt").read_bytes() == b"existing data"


def test_incomplete_or_corrupt_files_never_published(tmp_path):
    target = CollectionFiles([str(tmp_path)])
    entry = {
        "root": 0,
        "relative": "demo.pt",
        "size": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }
    target.begin(entry)
    with pytest.raises(ValueError, match="offset or length"):
        target.append(1, b"abc")
    target.abort()
    target.begin(entry)
    target.append(0, b"bad")
    with pytest.raises(ValueError, match="checksum"):
        target.finish()
    assert list(tmp_path.iterdir()) == []
    target.begin(entry)
    with pytest.raises(ValueError, match="checksum"):
        target.finish()
    assert list(tmp_path.iterdir()) == []


def test_path_escape_and_shared_filesystem(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_bytes(b"demo")
    endpoint = CollectionFiles([str(source)])
    assert not endpoint.begin(endpoint.manifest()[0])
    with pytest.raises(ValueError, match="escapes"):
        endpoint.begin({"root": 0, "relative": "../other", "size": 0, "sha256": ""})
    (source / "link").symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="symlinks"):
        endpoint.manifest()


def test_single_node_sync_needs_no_transfer(monkeypatch):
    import ray

    remote = Mock()
    monkeypatch.setattr(ray, "remote", remote)
    node = SimpleNamespace(ray_id="local")
    cluster = SimpleNamespace(num_nodes=1, get_node_info=lambda _: node)
    collection_sync.sync_collection_data(
        [{"node_id": "local", "directories": ["/tmp/demo"]}], cluster
    )
    remote.return_value.options.assert_not_called()


@pytest.mark.parametrize("enabled,fail", [(False, False), (True, False), (True, True)])
def test_driver_sync_runs_after_collection_and_before_worker_cleanup(
    monkeypatch, enabled, fail
):
    from omegaconf import OmegaConf

    script = (
        Path(__file__).resolve().parents[2] / "examples/embodiment/collect_real_data.py"
    )
    spec = importlib.util.spec_from_file_location("collection_sync_entrypoint", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    from contextlib import contextmanager

    @contextmanager
    def exit_cleanup(**kwargs):
        resources = []
        try:
            yield resources
        finally:
            assert collector in resources
            events.append("worker_closed")

    monkeypatch.setattr(module, "training_exit_cleanup", exit_cleanup)
    collector = Mock()
    collector.run.return_value.wait.side_effect = lambda: events.append("run_finished")
    collector.collection_outputs.return_value.wait.return_value = [
        {"node_id": "control", "directories": ["/tmp/demos"]}
    ]
    collector._close.side_effect = lambda: events.append("worker_closed")
    worker = Mock()
    worker.create_group.return_value = collector
    monkeypatch.setattr(module, "DataCollector", worker)
    monkeypatch.setattr(module, "Cluster", Mock())
    monkeypatch.setattr(module, "ComponentPlacement", Mock())

    def sync(outputs, cluster):
        assert events == ["run_finished"]
        assert outputs[0]["node_id"] == "control"
        events.append("sync")
        if fail:
            raise OSError("transfer failed")

    monkeypatch.setattr(collection_sync, "sync_collection_data", sync)
    cfg = OmegaConf.create(
        {
            "runner": {"sync_data_to_nodes": enabled},
            "cluster": {},
            "env": {"eval": {}, "group_name": "env"},
        }
    )
    if fail:
        with pytest.raises(OSError, match="transfer failed"):
            module.main.__wrapped__(cfg)
    else:
        module.main.__wrapped__(cfg)
    assert events == ["run_finished", *(["sync"] if enabled else []), "worker_closed"]
