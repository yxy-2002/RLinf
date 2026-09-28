# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import yaml
from rlinf_dexhand.pipeline import load_config


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "glove.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "glove": {"type": "psiglove_2", "side": "left", "port": "mock"},
                "retargeting": {
                    "type": "wuji_tier2",
                    "scale_file": "missing-default.yaml",
                },
                "hand": {"type": "wuji1hand", "side": "left"},
            }
        )
    )
    return path


@pytest.mark.parametrize("absolute", [False, True])
def test_override_replaces_default_before_validation(config, absolute):
    scale = config.parent / "operator.yaml"
    scale.write_text("hand: left\nscaling_factor: [1, 1, 1, 1, 1]\n")
    original = config.read_text()
    cfg = load_config(config, scale_file=str(scale) if absolute else scale.name)
    assert cfg["retargeting"]["scale_file"] == str(scale)
    assert config.read_text() == original


def test_none_uses_pipeline_default(config):
    with pytest.raises(FileNotFoundError):
        load_config(config, scale_file=None)
    scale = config.parent / "missing-default.yaml"
    scale.touch()
    assert load_config(config)["retargeting"]["scale_file"] == str(scale)


def test_missing_override_fails_without_default_fallback(config):
    (config.parent / "missing-default.yaml").touch()
    with pytest.raises(FileNotFoundError, match="operator.yaml"):
        load_config(config, scale_file="operator.yaml")


def test_ruiyan_rejects_scale_override(config):
    cfg = yaml.safe_load(config.read_text())
    cfg["glove"]["type"] = "psiglove_1"
    cfg["hand"]["type"] = "ruiyanhand"
    cfg["retargeting"] = {"type": "channel_linear"}
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="WujiTier2"):
        load_config(config, scale_file="operator.yaml")


def test_expert_passes_selected_scale_to_pipeline(config, monkeypatch):
    import rlinf_dexhand.pipeline as pipeline
    from rlinf_dexhand.glove.glove_expert import GloveExpert

    scale = config.parent / "operator.yaml"
    scale.touch()
    factory = Mock()
    monkeypatch.setattr(pipeline, "TeleopPipeline", factory)
    # No reader thread or serial access is needed to check construction.
    monkeypatch.setattr("threading.Thread.start", lambda self: None)
    GloveExpert(pipeline_config=str(config), scale_file=str(scale))
    assert factory.call_args.args[0]["retargeting"]["scale_file"] == str(scale)
