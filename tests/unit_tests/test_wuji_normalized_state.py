# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Measured posture conversion and lossless demo migration without hardware."""

import pickle

import numpy as np
import pytest
import torch
from rlinf_dexhand.wuji_spec import to_normalized, wuji_spec

from rlinf.utils.wuji_observation import normalize_wuji_hand_state
from toolkits.dexhand.normalize_demo_hand_state import (
    FIELD,
    add_normalized_state,
    assert_equal,
    load_payload,
    migrate_file,
    observations,
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_normalization_matches_controller_and_clips(side):
    spec = wuji_spec(side)
    low, high = np.asarray(spec.lower), np.asarray(spec.upper)
    poses = np.stack([low, (low + high) / 2, high, low - 0.01, high + 0.01])
    original = poses.copy()
    result = normalize_wuji_hand_state(poses[:, None], side)
    expected = np.stack([to_normalized(spec, np.clip(q, low, high)) for q in poses])
    np.testing.assert_allclose(result[:, 0], expected, atol=1e-7)
    np.testing.assert_array_equal(poses, original)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[1], 0.5)


@pytest.mark.parametrize("positions", [np.zeros(19), np.full(20, np.nan)])
def test_invalid_feedback_is_rejected(positions):
    with pytest.raises(ValueError):
        normalize_wuji_hand_state(positions)


@pytest.mark.parametrize("suffix", [".pt", ".pkl"])
def test_migration_preserves_payload_and_is_idempotent(tmp_path, suffix):
    spec = wuji_spec("left")
    states = torch.zeros(3, 1, 38)
    states[..., :20] = torch.tensor((np.array(spec.lower) + spec.upper) / 2)
    obs = {
        "states": states,
        "main_images": torch.zeros(3, 1, 2, 2, 3, dtype=torch.uint8),
    }
    actions = torch.rand(2, 1, 26)
    if suffix == ".pt":
        payload = {
            "curr_obs": {k: v[:-1] for k, v in obs.items()},
            "next_obs": {k: v[1:] for k, v in obs.items()},
            "actions": actions,
            "forward_inputs": {"action": actions.clone()},
        }
    else:
        payload = {
            "observations": [{k: v[t, 0] for k, v in obs.items()} for t in range(3)],
            "actions": [row.numpy() for row in actions],
            "infos": [{"label": "demo", "time": 1.0}],
        }
    path = tmp_path / f"episode{suffix}"
    if suffix == ".pt":
        torch.save(payload, path)
    else:
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
    assert migrate_file(path, "left") == (2 if suffix == ".pt" else 3)
    checked = load_payload(path)
    if suffix == ".pt":
        assert torch.equal(
            checked["curr_obs"][FIELD][1:], checked["next_obs"][FIELD][:-1]
        )
    for observation in observations(checked, suffix):
        value = observation.pop(FIELD)
        assert value.shape == (*observation["states"].shape[:-1], 20)
        torch.testing.assert_close(value, torch.full_like(value, 0.5))
    assert_equal(payload, checked)
    before = path.read_bytes()
    assert migrate_file(path, "left") == 0
    assert path.read_bytes() == before


def test_conflicting_existing_field_is_rejected():
    obs = {"states": torch.zeros(38), FIELD: torch.ones(20)}
    with pytest.raises(ValueError):
        add_normalized_state(obs, "left")
