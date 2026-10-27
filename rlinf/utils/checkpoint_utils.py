# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Load exported actor weights for standalone rollout inference."""

from pathlib import Path

import torch


def load_rollout_checkpoint(
    model: torch.nn.Module, checkpoint_path: str | Path
) -> Path:
    """Load full model weights without restoring learner or replay state.

    Args:
        model: Constructed policy with the same architecture as training.
        checkpoint_path: Full state-dict file, or a global_step, actor, or
            model_state_dict directory containing exported full_weights.pt.

    Returns:
        The resolved weights file that was loaded.

    Raises:
        FileNotFoundError: No exported full weights exist at the supplied path.
        RuntimeError: The state dict is incompatible with the constructed policy.
    """
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        candidates = (
            path / "actor" / "model_state_dict" / "full_weights.pt",
            path / "model_state_dict" / "full_weights.pt",
            path / "full_weights.pt",
        )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), path
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"No full model weights found at {checkpoint_path}. Set runner.ckpt_path "
            "to global_step_<N>, its actor directory, or "
            "actor/model_state_dict/full_weights.pt. Sharded learner checkpoints "
            "must be exported to full model weights before evaluation."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return path.resolve()
