"""Configuration of the nuScenes BEV mid-training dataset."""

from __future__ import annotations

from dataclasses import dataclass

from lifting.data.stages import TrainingPhase


@dataclass
class NuScenesBEVConfig:
    """Configuration for the nuScenes BEV mid-training dataset."""

    data_root: str | None = None
    """Local nuScenes data root. None → synthetic proxy mode."""

    split: str = "train"
    n_cameras: int = 6
    img_height: int = 450
    img_width: int = 800
    # Proxy fallback
    proxy_length: int = 28_130
    seed: int = 0
    phase: TrainingPhase = TrainingPhase.MID_TRAIN
