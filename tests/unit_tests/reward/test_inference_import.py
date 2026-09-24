"""Control nodes must import inference workers without GPU training dependencies."""

import subprocess
import sys
from pathlib import Path


def test_inference_import_without_training_dependencies():
    script = """
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'peft'} or fullname.startswith('rlinf.hybrid_engines.fsdp'):
            raise ModuleNotFoundError(f'Training dependency imported: {fullname}')

sys.meta_path.insert(0, BlockTrainingImports())
from rlinf.workers.reward import EmbodiedRewardWorker
from rlinf.workers.reward.embodied_reward_worker import EmbodiedRewardWorker as Direct
assert EmbodiedRewardWorker is Direct
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        timeout=60,
    )
