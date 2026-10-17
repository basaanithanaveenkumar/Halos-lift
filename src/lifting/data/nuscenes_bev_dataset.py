"""nuScenes BEV mid-training dataset, with a synthetic proxy mode."""

from __future__ import annotations

from loguru import logger
from torch.utils.data import Dataset

from lifting.data.camera_rig import CameraRig
from lifting.data.nuscenes_bev_config import NuScenesBEVConfig
from lifting.data.stages import MID_TRAIN_DATASET_SPECS
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset
from lifting.geometry import GridSpec


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
