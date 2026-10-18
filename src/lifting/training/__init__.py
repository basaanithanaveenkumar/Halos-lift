"""Training loop and task strategies."""

from lifting.training.bev_segmentation_task import BEVSegmentationTask
from lifting.training.depth_supervision import DepthSupervisedTask
from lifting.training.distributed_trainer import DistributedTrainer
from lifting.training.occupancy_task import OccupancyTask
from lifting.training.task import Task
from lifting.training.task_registry import TASKS
from lifting.training.trainer import Trainer
from lifting.training.training_history import TrainingHistory

__all__ = [
    "TASKS",
    "BEVSegmentationTask",
    "DepthSupervisedTask",
    "DistributedTrainer",
    "OccupancyTask",
    "Task",
    "Trainer",
    "TrainingHistory",
]
