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

"""Constants shared by the DexJoCo LAMP migration."""

from __future__ import annotations

from pathlib import Path

SINGLE_ARM_TASKS = (
    "water_plant",
    "pick_bucket",
    "hammer_nail",
    "fold_glasses",
    "pinch_tongs",
    "click_mouse",
    "pick_bucket_clean_lerobot",
)

BIMANUAL_TASKS = (
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_microwave_cook_clean_lerobot",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
)


def is_bimanual_task(task: str) -> bool:
    """Return whether a task follows the DexJoCo bimanual naming contract."""

    return str(task).startswith("bimanual")


ENVIRONMENT_TASK_ALIASES = {
    "pick_bucket_clean_lerobot": "pick_bucket",
    "bimanual_microwave_cook_clean_lerobot": "bimanual_microwave_cook",
}


def canonical_environment_task(task: str) -> str:
    """Map dataset variants to the DexJoCo environment configuration they share."""

    return ENVIRONMENT_TASK_ALIASES.get(task, task)


DEFAULT_DATASET_ROOT = Path("datasets/dexjoco_lerobot_datasets")

ARM_ACTION_DIM = 6
ARM_JOINT_DIM = 7
ARM_QUAT_ACTION_DIM = 7
HAND_ACTION_DIM = 16
RECORDED_ROTVEC_ACTION_DIM = ARM_ACTION_DIM + HAND_ACTION_DIM
MODEL_QUAT_ACTION_DIM = ARM_QUAT_ACTION_DIM + HAND_ACTION_DIM
ENV_ACTION_DIM = 3 + 4 + HAND_ACTION_DIM
STATE_QUAT_DIM = ENV_ACTION_DIM

BIMANUAL_STATE_QUAT_DIM = 2 * STATE_QUAT_DIM
BIMANUAL_RECORDED_ROTVEC_ACTION_DIM = 2 * RECORDED_ROTVEC_ACTION_DIM
BIMANUAL_MODEL_QUAT_ACTION_DIM = 2 * MODEL_QUAT_ACTION_DIM
BIMANUAL_ARM_JOINT_DIM = 2 * ARM_JOINT_DIM

RIGHT_STATE_ARM_SLICE = slice(0, 7)
LEFT_STATE_ARM_SLICE = slice(7, 14)
RIGHT_STATE_HAND_SLICE = slice(14, 30)
LEFT_STATE_HAND_SLICE = slice(30, 46)
RIGHT_ACTION_SLICE = slice(0, 22)
LEFT_ACTION_SLICE = slice(22, 44)

SINGLE_ARM_MAIN_IMAGE_KEYS = {
    task: (
        "observation.images.ego_right"
        if task == "click_mouse"
        else "observation.images.front"
    )
    for task in SINGLE_ARM_TASKS
}
WRIST_IMAGE_KEY = "observation.images.wrist"
BIMANUAL_MAIN_IMAGE_KEY = "observation.images.ego"
BIMANUAL_RIGHT_WRIST_IMAGE_KEY = "observation.images.wrist_right"
BIMANUAL_LEFT_WRIST_IMAGE_KEY = "observation.images.wrist_left"

# Compatibility value used by existing single-arm DP artifact architecture.
HISTORY_FRAMES = 8
