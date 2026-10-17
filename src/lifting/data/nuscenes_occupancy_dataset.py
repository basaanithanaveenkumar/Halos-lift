"""nuScenes 3D-occupancy (Occ3D labels) post-training dataset, with a synthetic proxy mode."""

from __future__ import annotations

import torch
from loguru import logger
from torch.utils.data import Dataset

from lifting.data.camera_rig import CameraRig
from lifting.data.nuscenes_occupancy_config import NuScenesOccupancyConfig
from lifting.data.stages import POST_TRAIN_DATASET_SPECS
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset
from lifting.geometry import GridSpec


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
