"""Phase 2 — Mid-training datasets for Halos-lift.

Mid-training bridges synthetic pretraining to real sensor distributions using
nuScenes / Waymo / Argoverse BEV data.

Usage::

    from lifting.data.mid_train_datasets import NuScenesBEVDataset

    ds = NuScenesBEVDataset(data_root=None)   # proxy mode — no download
    sample = ds[0]
    # keys: images, intrinsics, cam_to_ego, bev_labels, occupancy, phase

Classes moved to their own modules (one public class per file) are re-exported here so
existing ``from lifting.data.mid_train_datasets import ...`` statements keep working."""

from __future__ import annotations

from lifting.data.nuscenes_bev_config import NuScenesBEVConfig
from lifting.data.nuscenes_bev_dataset import NuScenesBEVDataset
from lifting.data.waymo_bev_config import WaymoBEVConfig
from lifting.data.waymo_bev_dataset import WaymoBEVDataset

__all__ = ["NuScenesBEVConfig", "NuScenesBEVDataset", "WaymoBEVConfig", "WaymoBEVDataset"]
