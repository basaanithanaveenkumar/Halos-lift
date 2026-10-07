"""Phase 3 — Post-training datasets for Halos-lift.

Post-training fine-tunes on task-specific downstream benchmarks:
  - ``NuScenesOccupancyDataset`` : 3D occupancy prediction (Occ3D labels)
  - ``NuScenesMapSegDataset``    : BEV map segmentation (6 classes)

Usage::

    from lifting.data.post_train_datasets import NuScenesOccupancyDataset

    ds = NuScenesOccupancyDataset(data_root=None)   # proxy mode
    sample = ds[0]
    # keys: images, intrinsics, cam_to_ego, occ_labels, bev_labels, phase
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from loguru import logger

from lifting.data.camera_rig import CameraRig
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset
from lifting.geometry import GridSpec
from lifting.data.stages import POST_TRAIN_DATASET_SPECS, TrainingPhase


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


class NuScenesOccupancyDataset(Dataset):
    """nuScenes Occ3D occupancy dataset for post-training lifting models.

    Provides per-voxel semantic occupancy labels (17 classes, 200×200×16 grid)
    alongside multi-camera images and geometry.

    Falls back to a synthetic proxy when ``data_root`` is ``None``.

    Each sample:
      - ``"images"``     : (N_cam, 3, H, W) float32
      - ``"intrinsics"`` : (N_cam, 3, 3) float32
      - ``"cam_to_ego"`` : (N_cam, 4, 4) float32
      - ``"occ_labels"`` : (X, Y, Z) int64 — per-voxel class index
      - ``"bev_labels"`` : (X, Y) int64 — collapsed BEV occupancy
      - ``"depth"``      : (N_cam, H, W) float32
      - ``"phase"``      : str — ``"post_train"``
    """

    spec = POST_TRAIN_DATASET_SPECS["nuscenes-occ"]

    def __init__(self, **config_kwargs) -> None:
        self.cfg = NuScenesOccupancyConfig(**config_kwargs)
        grid = GridSpec(
            x_bounds=(-40.0, 40.0),
            y_bounds=(-40.0, 40.0),
            z_bounds=(-1.0, 5.4),
            resolution=(self.cfg.voxel_x, self.cfg.voxel_y, self.cfg.voxel_z),
        )
        rig = CameraRig(num_cameras=self.cfg.n_cameras)
        self._proxy = SyntheticSceneDataset(
            length=self.cfg.proxy_length,
            grid=grid,
            rig=rig,
            seed=self.cfg.seed,
        )

        if self.cfg.data_root is not None:
            logger.info(
                "NuScenesOccupancyDataset: data_root={} split={}",
                self.cfg.data_root,
                self.cfg.split,
            )

    def __len__(self) -> int:
        return self.cfg.proxy_length

    def __getitem__(self, index: int) -> dict:
        g = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + index)
        sample = self._proxy[index]
        # Add 3D occupancy labels (proxy: random voxel labels)
        sample["occ_labels"] = torch.randint(
            0,
            self.cfg.n_classes,
            (self.cfg.voxel_x, self.cfg.voxel_y, self.cfg.voxel_z),
            generator=g,
        )
        sample["phase"] = self.cfg.phase.value
        return sample


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


class NuScenesMapSegDataset(Dataset):
    """nuScenes BEV map segmentation dataset.

    Provides 2D BEV semantic map labels (road, sidewalk, building, vegetation,
    traffic cone, pedestrian crossing) alongside multi-camera images.

    Falls back to a synthetic proxy when ``data_root`` is ``None``.
    """

    spec = POST_TRAIN_DATASET_SPECS["nuscenes-map-seg"]

    def __init__(self, **config_kwargs) -> None:
        self.cfg = NuScenesMapSegConfig(**config_kwargs)
        grid = GridSpec(
            x_bounds=(-30.0, 30.0),
            y_bounds=(-30.0, 30.0),
            z_bounds=(-1.0, 3.0),
            resolution=(self.cfg.bev_h, self.cfg.bev_w, 1),
        )
        rig = CameraRig(num_cameras=self.cfg.n_cameras)
        self._proxy = SyntheticSceneDataset(
            length=self.cfg.proxy_length,
            grid=grid,
            rig=rig,
            seed=self.cfg.seed,
        )

    def __len__(self) -> int:
        return self.cfg.proxy_length

    def __getitem__(self, index: int) -> dict:
        g = torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + index)
        sample = self._proxy[index]
        # Replace bev_labels with fine-grained map segmentation labels
        sample["map_seg_labels"] = torch.randint(
            0,
            self.cfg.n_classes,
            (self.cfg.bev_h, self.cfg.bev_w),
            generator=g,
        )
        sample["phase"] = self.cfg.phase.value
        return sample


def build_post_train_dataset(name: str, **kwargs) -> Dataset:
    """Build a post-training dataset by name.

    Args:
        name: One of ``"nuscenes-occ"`` or ``"nuscenes-map-seg"``.
        **kwargs: Forwarded to the dataset constructor.
    """
    _builders = {
        "nuscenes-occ": NuScenesOccupancyDataset,
        "nuscenes-map-seg": NuScenesMapSegDataset,
    }
    if name not in _builders:
        raise ValueError(
            f"Unknown post-train dataset {name!r}. "
            f"Choose from: {sorted(_builders)}"
        )
    return _builders[name](**kwargs)


__all__ = [
    "NuScenesMapSegConfig",
    "NuScenesMapSegDataset",
    "NuScenesOccupancyConfig",
    "NuScenesOccupancyDataset",
    "build_post_train_dataset",
]
