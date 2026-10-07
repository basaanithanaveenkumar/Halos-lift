"""Training-phase taxonomy for Halos-lift camera-to-BEV lifting models.

Three phases are defined:

  PRETRAIN   — large-scale procedural synthetic multi-camera scenes
  MID_TRAIN  — real-world nuScenes / Waymo / Argoverse BEV data
  POST_TRAIN — task-specific downstream (occupancy, detection, map seg.)

Example::

    from lifting.data.stages import TrainingPhase, PHASE_DATASET_SPECS

    print(PHASE_DATASET_SPECS[TrainingPhase.PRETRAIN])
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class TrainingPhase(StrEnum):
    """High-level training phase for camera-to-BEV lifting models.

      PRETRAIN   — large synthetic scenes, procedural geometry, no downloads
      MID_TRAIN  — real multi-camera driving data (nuScenes / Waymo)
      POST_TRAIN — task-specific fine-tuning (occupancy, 3D det., seg.)
    """

    PRETRAIN = "pretrain"
    MID_TRAIN = "mid_train"
    POST_TRAIN = "post_train"


# ---------------------------------------------------------------------------
# Canonical dataset specs per phase
# ---------------------------------------------------------------------------

PRETRAIN_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "synthetic-scene-large": {
        "name": "synthetic-scene-large",
        "phase": "pretrain",
        "source": "procedural",
        "hf_path": None,
        "description": (
            "Large procedurally-rendered multi-camera 3D scenes. "
            "Random camera rigs, object placements and geometries. "
            "Infinite, seeded — no external downloads required."
        ),
        "n_samples": "infinite (seeded)",
        "n_cameras": "4–8 (random)",
    },
}

MID_TRAIN_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "nuscenes-bev": {
        "name": "nuscenes-bev",
        "phase": "mid_train",
        "hf_path": None,  # local install
        "description": (
            "nuScenes — 1,000 scenes, 6 surround cameras. "
            "BEV occupancy + semantic map labels for lifting validation."
        ),
        "paper_reference": "Caesar et al. (2020) nuScenes",
        "official_url": "https://www.nuscenes.org/nuscenes",
        "n_scenes": 700,
    },
    "waymo-bev": {
        "name": "waymo-bev",
        "phase": "mid_train",
        "hf_path": None,
        "description": (
            "Waymo Open Dataset — 1,950 segments, 5 cameras. "
            "High-quality LiDAR depth for self-supervised lifting."
        ),
        "paper_reference": "Sun et al. (2020) Waymo Open Dataset",
        "official_url": "https://waymo.com/open/",
        "n_scenes": 1_000,
    },
    "argoverse2-bev": {
        "name": "argoverse2-bev",
        "phase": "mid_train",
        "hf_path": None,
        "description": "Argoverse 2 Sensor Dataset — 1,000 segments, 7 ring cameras.",
        "paper_reference": "Wilson et al. (2023) Argoverse 2",
        "n_scenes": 700,
    },
}

POST_TRAIN_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "nuscenes-occ": {
        "name": "nuscenes-occ",
        "phase": "post_train",
        "hf_path": None,
        "description": (
            "nuScenes Occ3D occupancy labels — "
            "200×200×16 voxel grid, 17 semantic classes."
        ),
        "paper_reference": "Tian et al. (2023) Occ3D",
        "n_classes": 17,
    },
    "nuscenes-map-seg": {
        "name": "nuscenes-map-seg",
        "phase": "post_train",
        "hf_path": None,
        "description": (
            "nuScenes BEV map segmentation — "
            "6 static map classes (road, sidewalk, building, vegetation, …)."
        ),
        "paper_reference": "Caesar et al. (2020) nuScenes",
        "n_classes": 6,
    },
}

# Unified lookup by phase
PHASE_DATASET_SPECS: dict[TrainingPhase, dict[str, dict[str, Any]]] = {
    TrainingPhase.PRETRAIN: PRETRAIN_DATASET_SPECS,
    TrainingPhase.MID_TRAIN: MID_TRAIN_DATASET_SPECS,
    TrainingPhase.POST_TRAIN: POST_TRAIN_DATASET_SPECS,
}

PHASE_NOTES: dict[TrainingPhase, str] = {
    TrainingPhase.PRETRAIN: (
        "Procedural synthetic scenes covering diverse camera rigs and 3D layouts. "
        "Trains the lifter to understand arbitrary camera geometry."
    ),
    TrainingPhase.MID_TRAIN: (
        "Real-world multi-camera driving data for domain adaptation. "
        "Aligns synthetic geometry features with real sensor noise and occlusion."
    ),
    TrainingPhase.POST_TRAIN: (
        "Task-specific fine-tuning on a target downstream benchmark. "
        "Maximises metric (mIoU, mAP, NDS) on the deployment dataset."
    ),
}


__all__ = [
    "MID_TRAIN_DATASET_SPECS",
    "PHASE_DATASET_SPECS",
    "PHASE_NOTES",
    "POST_TRAIN_DATASET_SPECS",
    "PRETRAIN_DATASET_SPECS",
    "TrainingPhase",
]
