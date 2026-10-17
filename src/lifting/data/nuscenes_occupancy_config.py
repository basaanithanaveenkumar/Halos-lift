"""Configuration of the nuScenes 3D-occupancy post-training dataset."""

from __future__ import annotations

from dataclasses import dataclass

from lifting.data.stages import TrainingPhase


@dataclass
class NuScenesOccupancyConfig:
    """Configuration for the nuScenes occupancy post-training dataset."""

    data_root: str | None = None
    split: str = "val"
    n_cameras: int = 6
    n_classes: int = 17
    voxel_x: int = 200
    voxel_y: int = 200
    voxel_z: int = 16
    proxy_length: int = 6_019  # ≈ nuScenes val count
    seed: int = 2
    phase: TrainingPhase = TrainingPhase.POST_TRAIN
