"""The :data:`TASKS` registry: training tasks selectable by name from a config."""

from __future__ import annotations

from lifting.registry import Registry
from lifting.training.bev_segmentation_task import BEVSegmentationTask
from lifting.training.occupancy_task import OccupancyTask
from lifting.training.task import Task

#: Pattern: Registry / Factory. ``TASKS.build({"type": "occupancy", "num_classes": 3})``.
TASKS: Registry[Task] = Registry("task")

TASKS.register("bev_segmentation")(BEVSegmentationTask)
TASKS.register("occupancy")(OccupancyTask)
