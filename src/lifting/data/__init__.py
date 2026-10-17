"""Procedural 3D synthetic scenes with exact camera geometry and labels.

Stage-aware datasets:

    from lifting.data.stages import TrainingPhase
    from lifting.data.large_synthetic_scene_dataset import LargeSyntheticSceneDataset
    from lifting.data.nuscenes_bev_dataset import NuScenesBEVDataset
    from lifting.data.nuscenes_occupancy_dataset import NuScenesOccupancyDataset
    from lifting.data.nuscenes_map_seg_dataset import NuScenesMapSegDataset
"""

from lifting.data.camera_rig import CameraRig
from lifting.data.label_rasterizer import LabelRasterizer
from lifting.data.object_class import DEFAULT_CLASSES, PEDESTRIAN, VEHICLE, ObjectClass
from lifting.data.object_sampler import ObjectSampler
from lifting.data.ray_cast_renderer import RayCastRenderer
from lifting.data.render_result import RenderResult
from lifting.data.scene_batch import SceneBatch
from lifting.data.scene_objects import SceneObjects

# Stage-aware datasets (new)
from lifting.data.stages import PHASE_DATASET_SPECS, PHASE_NOTES, TrainingPhase
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset

__all__ = [
    # Existing API — unchanged
    "DEFAULT_CLASSES",
    "PEDESTRIAN",
    "VEHICLE",
    "CameraRig",
    "LabelRasterizer",
    "ObjectClass",
    "ObjectSampler",
    "RayCastRenderer",
    "RenderResult",
    "SceneBatch",
    "SceneObjects",
    "SyntheticSceneDataset",
    # Stage-aware API (new)
    "TrainingPhase",
    "PHASE_DATASET_SPECS",
    "PHASE_NOTES",
]
