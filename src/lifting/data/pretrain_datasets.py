"""Phase 1 — Pre-training datasets for Halos-lift.

Extends the existing ``SyntheticSceneDataset`` with larger-scale, phase-aware
wrappers. No external data required — all scenes are procedurally rendered.

Usage::

    from lifting.data.pretrain_datasets import LargeSyntheticSceneDataset

    ds = LargeSyntheticSceneDataset(length=500_000, n_cameras=6)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4,
                                                   collate_fn=SceneBatch.collate)))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from lifting.data.camera_rig import CameraRig
from lifting.data.label_rasterizer import LabelRasterizer
from lifting.data.object_sampler import ObjectSampler
from lifting.data.ray_cast_renderer import RayCastRenderer
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset
from lifting.geometry import GridSpec

from lifting.data.stages import PRETRAIN_DATASET_SPECS, TrainingPhase


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


class LargeSyntheticSceneDataset(Dataset):
    """Large-scale multi-camera synthetic scene dataset for pre-training lifting models.

    Wraps ``SyntheticSceneDataset`` with a larger virtual length and optional
    random camera rig augmentation (number of cameras varies per scene).

    Each item matches the ``SyntheticSceneDataset`` schema:
      ``images``, ``depth``, ``intrinsics``, ``cam_to_ego``,
      ``bev_labels``, ``occupancy``
    plus:
      ``phase`` : str — always ``"pretrain"``
    """

    # Metadata for the data pipeline
    spec = PRETRAIN_DATASET_SPECS["synthetic-scene-large"]

    def __init__(self, **config_kwargs) -> None:
        self.cfg = LargeSyntheticSceneConfig(**config_kwargs)
        grid = GridSpec(
            x_bounds=(-self.cfg.grid_x_size, self.cfg.grid_x_size),
            y_bounds=(-self.cfg.grid_y_size, self.cfg.grid_y_size),
            z_bounds=(-2.0, 2.0),
            resolution=(200, 200, self.cfg.grid_z_bins),
        )
        rig = CameraRig(num_cameras=self.cfg.n_cameras)
        self._inner = SyntheticSceneDataset(
            length=self.cfg.length,
            grid=grid,
            rig=rig,
            seed=self.cfg.seed,
        )

    def __len__(self) -> int:
        return self.cfg.length

    def __getitem__(self, index: int) -> dict:
        sample = self._inner[index]
        sample["phase"] = self.cfg.phase.value
        return sample


__all__ = [
    "LargeSyntheticSceneConfig",
    "LargeSyntheticSceneDataset",
]
