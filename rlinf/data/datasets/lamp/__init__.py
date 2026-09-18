# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Format-independent LAMP training data interfaces; dependencies fail on import."""

from rlinf.data.datasets.lamp.action_windows import (
    ActionWindowDataset,
    build_action_windows,
    save_action_artifact,
)
from rlinf.data.datasets.lamp.offline_dataset import (
    LampDataSource,
    LampFrameData,
    LampMMapDataset,
    LampSourceMetadata,
    lamp_steps_per_epoch,
    load_cache_metadata,
    load_cache_statistics,
    prepare_lamp_cache,
)
from rlinf.data.datasets.lamp.residual_replay import (
    LAMP_RESIDUAL_FLAT_ACTION_DIM,
    LAMP_RESIDUAL_REPLAY_CONTRACT,
    LAMP_RESIDUAL_REPLAY_SCHEMA_VERSION,
    validate_lamp_residual_trajectory,
)

__all__ = [
    "LampDataSource",
    "LampSourceMetadata",
    "LampFrameData",
    "LampMMapDataset",
    "LAMP_RESIDUAL_FLAT_ACTION_DIM",
    "LAMP_RESIDUAL_REPLAY_CONTRACT",
    "LAMP_RESIDUAL_REPLAY_SCHEMA_VERSION",
    "lamp_steps_per_epoch",
    "load_cache_metadata",
    "load_cache_statistics",
    "prepare_lamp_cache",
    "validate_lamp_residual_trajectory",
    "ActionWindowDataset",
    "build_action_windows",
    "save_action_artifact",
]
