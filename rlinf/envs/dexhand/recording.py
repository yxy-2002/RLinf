# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Streaming episode records; failed episodes remain explicitly aborted."""

import hashlib
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path


class EpisodeWriter:
    def __init__(self, directory, config, spec):
        self.path = (
            Path(directory) / f"episode-{time.time_ns()}-{uuid.uuid4().hex[:8]}.jsonl"
        )
        self.config = config
        self.spec = spec
        self.count = 0

    def __enter__(self):
        digest = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()
        artifacts = {}
        for section in self.config.values():
            if isinstance(section, dict):
                for key, value in section.items():
                    if (key.endswith("_file") or key.endswith("_urdf")) and value:
                        artifacts[str(value)] = hashlib.sha256(
                            Path(value).read_bytes()
                        ).hexdigest()
        if self.config.get("replay_file"):
            replay = Path(self.config["replay_file"])
            artifacts[str(replay)] = hashlib.sha256(replay.read_bytes()).hexdigest()
        from rlinf_dexhand.retargeting.kinematics import ASSETS

        for asset in sorted(ASSETS.rglob("*")):
            if asset.is_file() and asset.suffix in (".urdf", ".yaml", ".yml"):
                artifacts["bundled:" + str(asset.relative_to(ASSETS))] = hashlib.sha256(
                    asset.read_bytes()
                ).hexdigest()
        self.file = self.path.open("x")
        self.file.write(
            json.dumps(
                {
                    "type": "metadata",
                    "schema_version": 1,
                    "package_version": "0.2.0",
                    "config": self.config,
                    "config_sha256": digest,
                    "artifacts": artifacts,
                    "hand_spec": asdict(self.spec),
                }
            )
            + "\n"
        )
        self.file.flush()
        return self

    def append(self, record):
        if not record["state"]["valid"] or not record["sample"]["valid"]:
            raise ValueError("Cannot record invalid data as a valid transition")
        self.file.write(json.dumps({"type": "step", **record}, allow_nan=False) + "\n")
        self.file.flush()
        self.count += 1

    def __exit__(self, kind, error, tb):
        self.file.write(
            json.dumps(
                {
                    "type": "end",
                    "status": "aborted" if kind else "complete",
                    "error": str(error) if kind else None,
                    "steps": self.count,
                }
            )
            + "\n"
        )
        self.file.close()
