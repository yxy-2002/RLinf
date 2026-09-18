# Copyright 2026 The RLinf Authors.
"""Joint preprocessing retains the persisted data contract after name cleanup."""

import numpy as np
import pytest

from rlinf.data.datasets.lamp import dexjoco_lerobot as data


@pytest.fixture
def joint_artifact(tmp_path, monkeypatch):
    actions = np.zeros((4, 22), dtype=np.float32)
    states = np.zeros((4, 23), dtype=np.float32)
    states[:, 3] = 1
    rows = np.arange(4)
    episodes = np.array([0, 0, 1, 1])
    frames = np.array([0, 1, 0, 1])
    joints = np.arange(28, dtype=np.float32).reshape(4, 7)
    diagnostics = {"pos_err": np.zeros(4), "ori_err": np.zeros(4)}
    xml = tmp_path / "robot.xml"
    xml.write_text("test robot model")
    monkeypatch.setattr(data, "_resolve_task_root", lambda *args: tmp_path)
    monkeypatch.setattr(
        data,
        "_read_lowdim_arrays",
        lambda *args: (actions, states, episodes, frames, rows),
    )
    monkeypatch.setattr(
        data, "_solve_arm_joint_ik_sequence", lambda *args: (joints, diagnostics)
    )
    monkeypatch.setattr(data, "_panda_allegro_xml_path", lambda: xml)
    path, _ = data.build_joint_preprocess_artifact("water_plant", tmp_path)
    return (
        path,
        ("water_plant", tmp_path, states, actions, rows, episodes, frames),
        joints,
    )


def test_joint_preprocessing_preserves_existing_disk_contract(joint_artifact):
    path, args, joints = joint_artifact
    assert path.name == "bc_preprocess_v2.npz"
    with np.load(path) as saved:
        assert int(saved["version"]) == 2
        expected_actions = saved["action_quat23"].copy()
    loaded = data.load_joint_preprocess_artifact(*args)
    np.testing.assert_array_equal(loaded.arm_joint_state, joints)
    np.testing.assert_array_equal(loaded.action_quat23, expected_actions)
    provenance = data.joint_preprocess_provenance(args[1], dataset_sha256="a" * 64)
    assert provenance["schema"] == "dexjoco_joint_ik_v2"
    assert provenance["artifact_version"] == 2
    assert provenance["artifact_filename"] == path.name


def test_joint_preprocessing_rejects_wrong_schema(joint_artifact):
    path, args, _ = joint_artifact
    with np.load(path) as saved:
        payload = {key: saved[key].copy() for key in saved.files}
    payload["version"] = np.asarray(1, dtype=np.int32)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="version"):
        data.load_joint_preprocess_artifact(*args)
    with pytest.raises(ValueError, match="version mismatch"):
        data.joint_preprocess_provenance(args[1], dataset_sha256="a" * 64)


def test_joint_preprocessing_missing_cache_does_not_fallback(joint_artifact):
    path, args, _ = joint_artifact
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing Joint preprocessing"):
        data.load_joint_preprocess_artifact(*args)
    assert not path.exists()


@pytest.mark.parametrize("preprocessed", [True, False])
def test_lerobot_adapter_reads_frames_once_and_keeps_image_row_alignment(
    joint_artifact, monkeypatch, preprocessed
):
    import json

    path, args, joints = joint_artifact
    root = args[1]
    keys = (data.SINGLE_ARM_MAIN_IMAGE_KEYS["water_plant"], data.WRIST_IMAGE_KEY)
    (root / "meta" / "info.json").write_text(
        json.dumps({"features": {key: {} for key in keys}})
    )
    monkeypatch.setattr(data, "lerobot_dataset_sha256", lambda root: "a" * 64)
    monkeypatch.setattr(data, "_video_manifest_sha256", lambda *args: "b" * 64)
    calls = []
    _, _, state, action, rows, episodes, frames = args

    def read(*args):
        calls.append("read")
        return action, state, episodes, frames, rows

    monkeypatch.setattr(data, "_read_lowdim_arrays", read)
    source = data.DexjocoLeRobotSource("water_plant", root)
    assert source.metadata.data_sha256 == "a" * 64
    assert calls == []
    if not preprocessed:
        path.unlink()
    loaded = source.load_frames()
    assert source.load_frames() is loaded
    assert calls == ["read"]
    np.testing.assert_array_equal(loaded.arm_state, joints)
    np.testing.assert_array_equal(loaded.hand_state, state[:, 7:23])
    with np.load(path) as stored:
        np.testing.assert_array_equal(loaded.action, stored["action_quat23"])
    selected = []

    def decode(root, key, frame_rows, size, **kwargs):
        selected.append(frame_rows)
        return np.zeros((len(frame_rows), size, size, 3), np.uint8)

    monkeypatch.setattr(data, "decode_video_rows", decode)
    images = source.images_for(np.array([3, 1]), 4, label="test")
    assert set(images) == {"front", "wrist"}
    for selection in selected:
        np.testing.assert_array_equal(selection, rows[[3, 1]])
    assert calls == ["read"]
