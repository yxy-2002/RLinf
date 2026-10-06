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

"""Explicit, lazy construction of LAMP data format adapters."""

from importlib import import_module


def create_lamp_data_source(cfg):
    """Create a source from the legacy name or an explicit module:factory.

    Custom factories receive the data config and return a LampDataSource with
    an explicit robot_spec in metadata. Heavy format dependencies stay local
    to the selected factory.
    """
    if cfg.get("dataset_type") == "dexjoco_lamp":
        from .dexjoco_lerobot import DexjocoLeRobotSource

        return DexjocoLeRobotSource(str(cfg["task_name"]), cfg["dataset_root"])
    target = cfg.get("source_factory")
    if not target or ":" not in target:
        raise ValueError("Custom LAMP data requires source_factory=module:factory")
    module, name = target.split(":", 1)
    source = getattr(import_module(module), name)(cfg)
    if not source.metadata.robot_spec_explicit:
        raise ValueError("Custom LAMP source must declare a robot_spec")
    return source
