# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check relocated entrypoints without importing robot initialization code."""

import ast
import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = ROOT / "examples/embodiment/franka_ruiyan"


@pytest.mark.parametrize("name", ["collect_reward_data", "collect_demos"])
def test_collector_config_matches_original(name, monkeypatch):
    monkeypatch.setenv("REWARD_PATH", str(ROOT / "examples/reward"))
    tree = ast.parse((ENTRYPOINTS / f"{name}.py").read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    decorator = main.decorator_list[0]
    options = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords}
    moved_path = (ENTRYPOINTS / options["config_path"]).resolve()
    original_path = ROOT / "examples/reward/config"
    assert moved_path == original_path
    results = []
    for directory in (moved_path, original_path):
        with initialize_config_dir(version_base="1.1", config_dir=str(directory)):
            cfg = compose(config_name=options["config_name"])
            results.append(OmegaConf.to_container(cfg, resolve=True))
    assert results[0] == results[1]


def test_framework_protocol_has_no_examples_dependency():
    source = (ROOT / "rlinf/utils/ruiyan_rlpd.py").read_text()
    assert "from examples." not in source
    from rlinf.utils.ruiyan_reward_protocol import RewardClient, SuccessGate

    assert callable(RewardClient)
    assert SuccessGate(0.75, 1).update(0.8)


def test_readme_entrypoints_exist():
    readme = (ENTRYPOINTS / "README.md").read_text()
    paths = re.findall(r"(?:examples|rlinf)/[\w/.-]+\.(?:py|sh|yaml)", readme)
    assert paths
    for path in paths:
        assert (ROOT / path).is_file(), path
    assert "[c]ollect_reward_data" in readme
    assert "[c]ollect_demos" in readme
