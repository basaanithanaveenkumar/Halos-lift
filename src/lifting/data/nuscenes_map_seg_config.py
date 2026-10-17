"""Configuration of the nuScenes BEV map-segmentation post-training dataset."""

from __future__ import annotations

from dataclasses import dataclass

from lifting.data.stages import TrainingPhase


@dataclass
class NuScenesMapSegConfig:
    """Configuration for the nuScenes BEV map segmentation dataset."""

    data_root: str | None = None
    split: str = "val"
    n_cameras: int = 6
    n_classes: int = 6
    bev_h: int = 200
    bev_w: int = 200
    proxy_length: int = 6_019
    seed: int = 3
    phase: TrainingPhase = TrainingPhase.POST_TRAIN
