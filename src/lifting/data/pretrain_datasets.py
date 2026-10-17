"""Phase 1 — Pre-training datasets for Halos-lift.

Extends the existing ``SyntheticSceneDataset`` with larger-scale, phase-aware
wrappers. No external data required — all scenes are procedurally rendered.

Usage::

    from lifting.data.pretrain_datasets import LargeSyntheticSceneDataset

    ds = LargeSyntheticSceneDataset(length=500_000, n_cameras=6)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4,
                                                   collate_fn=SceneBatch.collate)))

Classes moved to their own modules (one public class per file) are re-exported here so
existing ``from lifting.data.pretrain_datasets import ...`` statements keep working."""

from __future__ import annotations

from lifting.data.large_synthetic_scene_config import LargeSyntheticSceneConfig
from lifting.data.large_synthetic_scene_dataset import LargeSyntheticSceneDataset

__all__ = ["LargeSyntheticSceneConfig", "LargeSyntheticSceneDataset"]
