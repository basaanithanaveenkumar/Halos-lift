"""Large procedurally generated scene dataset for pre-training."""

from __future__ import annotations

from torch.utils.data import Dataset

from lifting.data.camera_rig import CameraRig
from lifting.data.large_synthetic_scene_config import LargeSyntheticSceneConfig
from lifting.data.stages import PRETRAIN_DATASET_SPECS
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset
from lifting.geometry import GridSpec


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
