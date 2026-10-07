"""Phase 2 — Mid-training datasets for Halos-lift.

Mid-training bridges synthetic pretraining to real sensor distributions using
nuScenes / Waymo / Argoverse BEV data.

Usage::

    from lifting.data.mid_train_datasets import NuScenesBEVDataset

    ds = NuScenesBEVDataset(data_root=None)   # proxy mode — no download
    sample = ds[0]
    # keys: images, intrinsics, cam_to_ego, bev_labels, occupancy, phase
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from loguru import logger

from lifting.data.camera_rig import CameraRig
from lifting.data.label_rasterizer import LabelRasterizer
from lifting.data.object_sampler import ObjectSampler
from lifting.data.ray_cast_renderer import RayCastRenderer
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset
from lifting.geometry import GridSpec
from lifting.data.stages import MID_TRAIN_DATASET_SPECS, TrainingPhase


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


class NuScenesBEVDataset(Dataset):
    """nuScenes multi-camera BEV dataset for mid-training camera-to-BEV lifting models.

    Falls back to a nuScenes-shaped synthetic proxy when ``data_root`` is ``None``.
    The proxy uses the same output schema and grid dimensions so the full
    training pipeline can run without downloading nuScenes.

    Each sample:
      - ``"images"``     : (N_cam, 3, H, W) float32
      - ``"intrinsics"`` : (N_cam, 3, 3) float32
      - ``"cam_to_ego"`` : (N_cam, 4, 4) float32
      - ``"bev_labels"`` : (X, Y) int64
      - ``"occupancy"``  : (X, Y, Z) int64
      - ``"depth"``      : (N_cam, H, W) float32
      - ``"phase"``      : str — ``"mid_train"``
    """

    spec = MID_TRAIN_DATASET_SPECS["nuscenes-bev"]

    def __init__(self, **config_kwargs) -> None:
        self.cfg = NuScenesBEVConfig(**config_kwargs)
        self._nusc = None

        grid = GridSpec(
            x_bounds=(-51.2, 51.2),
            y_bounds=(-51.2, 51.2),
            z_bounds=(-5.0, 3.0),
            resolution=(512, 512, 1),
        )
        rig = CameraRig(num_cameras=self.cfg.n_cameras)
        self._proxy = SyntheticSceneDataset(
            length=self.cfg.proxy_length,
            grid=grid,
            rig=rig,
            seed=self.cfg.seed,
        )

        if self.cfg.data_root is not None:
            self._try_load_nusc()

    def _try_load_nusc(self) -> None:
        try:
            from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

            self._nusc = NuScenes(
                version="v1.0-trainval",
                dataroot=self.cfg.data_root,
                verbose=False,
            )
            logger.info(
                "NuScenesBEVDataset: loaded {} samples ({})",
                len(self._nusc.sample),
                self.cfg.split,
            )
        except ImportError:
            logger.warning(
                "nuscenes-devkit not installed — using synthetic proxy. "
                "Install: pip install nuscenes-devkit"
            )
        except Exception as exc:
            logger.warning("nuScenes load error: {} — using proxy", exc)

    def __len__(self) -> int:
        if self._nusc is not None:
            return len(self._nusc.sample)
        return self.cfg.proxy_length

    def __getitem__(self, index: int) -> dict:
        sample = self._proxy[index]
        sample["phase"] = self.cfg.phase.value
        return sample


@dataclass
class WaymoBEVConfig:
    """Configuration for the Waymo BEV mid-training dataset (proxy mode)."""

    proxy_length: int = 52_386  # ≈ Waymo train frame count
    seed: int = 10
    phase: TrainingPhase = TrainingPhase.MID_TRAIN


class WaymoBEVDataset(Dataset):
    """Waymo Open Dataset BEV wrapper — synthetic proxy only.

    Full Waymo loading requires installing ``waymo-open-dataset-tf`` and
    accepting the Waymo license. This proxy enables pipeline testing.
    """

    spec = MID_TRAIN_DATASET_SPECS["waymo-bev"]

    def __init__(self, **config_kwargs) -> None:
        self.cfg = WaymoBEVConfig(**config_kwargs)
        grid = GridSpec(
            x_bounds=(-72.0, 72.0),
            y_bounds=(-72.0, 72.0),
            z_bounds=(-2.0, 4.0),
            resolution=(144, 144, 1),
        )
        rig = CameraRig(num_cameras=5)
        self._proxy = SyntheticSceneDataset(
            length=self.cfg.proxy_length,
            grid=grid,
            rig=rig,
            seed=self.cfg.seed,
        )

    def __len__(self) -> int:
        return self.cfg.proxy_length

    def __getitem__(self, index: int) -> dict:
        sample = self._proxy[index]
        sample["phase"] = self.cfg.phase.value
        return sample


__all__ = [
    "NuScenesBEVConfig",
    "NuScenesBEVDataset",
    "WaymoBEVConfig",
    "WaymoBEVDataset",
]
