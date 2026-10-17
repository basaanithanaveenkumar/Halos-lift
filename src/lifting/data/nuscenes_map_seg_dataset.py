"""nuScenes BEV map-segmentation post-training dataset, with a synthetic proxy mode."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from lifting.data.camera_rig import CameraRig
from lifting.data.nuscenes_map_seg_config import NuScenesMapSegConfig
from lifting.data.stages import POST_TRAIN_DATASET_SPECS
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset
from lifting.geometry import GridSpec


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
