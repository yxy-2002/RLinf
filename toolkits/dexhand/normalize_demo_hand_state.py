# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Add measured normalized Wuji posture to existing native demo observations."""

import argparse
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from rlinf.utils.logging import get_logger
from rlinf.utils.wuji_observation import normalize_wuji_hand_state

FIELD = "hand_state_normalized"


def add_normalized_state(observation: dict, side: str) -> bool:
    """Add the posture field in place; reject conflicting existing values."""
    states = observation["states"]
    values = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != 38:
        raise ValueError("Expected flattened Wuji Euler observations [..., 38]")
    normalized = normalize_wuji_hand_state(values[..., :20], side)
    result = (
        torch.from_numpy(normalized).to(states.device)
        if isinstance(states, torch.Tensor)
        else normalized
    )
    if FIELD in observation:
        assert_equal(observation[FIELD], result)
        return False
    observation[FIELD] = result
    return True


def assert_equal(before: object, after: object) -> None:
    """Check exact preservation of all original nested payload values."""
    if type(before) is not type(after):
        raise ValueError("Payload type changed")
    if isinstance(before, torch.Tensor):
        equal = before.dtype == after.dtype and torch.equal(before, after)
    elif isinstance(before, np.ndarray):
        equal = before.dtype == after.dtype and np.array_equal(before, after)
    elif isinstance(before, dict):
        if before.keys() != after.keys():
            raise ValueError("Payload keys changed")
        for key in before:
            assert_equal(before[key], after[key])
        return
    elif isinstance(before, (list, tuple)):
        if len(before) != len(after):
            raise ValueError("Payload length changed")
        for left, right in zip(before, after):
            assert_equal(left, right)
        return
    else:
        equal = before == after
    if not equal:
        raise ValueError("Payload value changed")


def load_payload(path: Path) -> dict:
    """Read a native trajectory or trusted local collection pickle."""
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    with path.open("rb") as handle:
        return pickle.load(handle)


def observations(payload: dict, suffix: str) -> list[dict]:
    """Return the observation containers for each native recording format."""
    if suffix == ".pt":
        return [payload["curr_obs"], payload["next_obs"]]
    return payload["observations"]


def migrate_file(path: Path, side: str) -> int:
    """Atomically replace one file after checking new and original fields."""
    payload = load_payload(path)
    added = [
        add_normalized_state(obs, side) for obs in observations(payload, path.suffix)
    ]
    if not any(added):
        return 0
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            if path.suffix == ".pt":
                torch.save(payload, handle)
            else:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        # Use the destination format even though the temporary suffix differs.
        if path.suffix == ".pt":
            checked = torch.load(temporary, map_location="cpu", weights_only=True)
        else:
            with temporary.open("rb") as handle:
                checked = pickle.load(handle)
        assert_equal(payload, checked)
        for obs, was_added in zip(observations(checked, path.suffix), added):
            if was_added:
                del obs[FIELD]
        assert_equal(load_payload(path), checked)
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sum(added)


def main() -> None:
    """Back up a run directory and migrate its demos and pickle episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    args = parser.parse_args()
    root, backup = args.data_dir.resolve(), args.backup_dir.resolve()
    if backup == root or root in backup.parents:
        parser.error("Backup must be outside the source directory")
    paths = sorted((root / "demos").glob("trajectory_*.pt")) + sorted(
        (root / "collected_data").glob("*.pkl")
    )
    if not paths:
        parser.error("No demos/trajectory_*.pt or collected_data/*.pkl found")
    # Refuse existing backups, so a repeated invocation cannot overwrite originals.
    shutil.copytree(root, backup)
    logger = get_logger()
    logger.info("Original dataset backed up to %s", backup)
    count = 0
    for path in paths:
        count += migrate_file(path, args.side)
        logger.info("Verified %s", path.relative_to(root))
    logger.info(
        "Finished: %d files, %d observation containers updated", len(paths), count
    )


if __name__ == "__main__":
    main()
