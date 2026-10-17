"""Waymo Open BEV mid-training dataset, with a synthetic proxy mode."""

from __future__ import annotations

from torch.utils.data import Dataset

from lifting.data.camera_rig import CameraRig
from lifting.data.stages import MID_TRAIN_DATASET_SPECS
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset
from lifting.data.waymo_bev_config import WaymoBEVConfig
from lifting.geometry import GridSpec


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
