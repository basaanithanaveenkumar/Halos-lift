"""Phase 3 — Post-training datasets for Halos-lift.

Post-training fine-tunes on task-specific downstream benchmarks:
  - ``NuScenesOccupancyDataset`` : 3D occupancy prediction (Occ3D labels)
  - ``NuScenesMapSegDataset``    : BEV map segmentation (6 classes)

Usage::

    from lifting.data.post_train_datasets import NuScenesOccupancyDataset

    ds = NuScenesOccupancyDataset(data_root=None)   # proxy mode
    sample = ds[0]
    # keys: images, intrinsics, cam_to_ego, occ_labels, bev_labels, phase

Classes moved to their own modules (one public class per file) are re-exported here so
existing ``from lifting.data.post_train_datasets import ...`` statements keep working."""

from __future__ import annotations

from torch.utils.data import Dataset

from lifting.data.nuscenes_map_seg_config import NuScenesMapSegConfig
from lifting.data.nuscenes_map_seg_dataset import NuScenesMapSegDataset
from lifting.data.nuscenes_occupancy_config import NuScenesOccupancyConfig
from lifting.data.nuscenes_occupancy_dataset import NuScenesOccupancyDataset
from lifting.exceptions import InvalidConfigError


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
        raise InvalidConfigError(
            f"Unknown post-train dataset {name!r}. "
            f"Choose from: {sorted(_builders)}"
        )
    return _builders[name](**kwargs)

__all__ = ["NuScenesMapSegConfig", "NuScenesMapSegDataset", "NuScenesOccupancyConfig", "NuScenesOccupancyDataset", "build_post_train_dataset"]
