"""Configuration of the large synthetic pre-training dataset."""

from __future__ import annotations

from dataclasses import dataclass

from lifting.data.stages import TrainingPhase


@dataclass
class LargeSyntheticSceneConfig:
    """Configuration for the large-scale synthetic pretrain dataset."""

    length: int = 500_000
    """Virtual length — each index maps to a deterministic unique scene."""

    n_cameras: int = 4
    """Number of cameras per scene (randomly rotated rig)."""

    grid_x_size: float = 50.0
    grid_y_size: float = 50.0
    grid_z_bins: int = 4

    img_height: int = 256
    img_width: int = 256
    seed: int = 0

    # Phase label (for dataset-level metadata)
    phase: TrainingPhase = TrainingPhase.PRETRAIN
