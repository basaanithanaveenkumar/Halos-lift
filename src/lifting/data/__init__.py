"""Procedural 3D synthetic scenes with exact camera geometry and labels.

Stage-aware datasets:

    from lifting.data.stages import TrainingPhase
    from lifting.data.pretrain_datasets import LargeSyntheticSceneDataset
    from lifting.data.mid_train_datasets import NuScenesBEVDataset
    from lifting.data.post_train_datasets import NuScenesOccupancyDataset, NuScenesMapSegDataset
"""

from lifting.data.camera_rig import CameraRig
from lifting.data.label_rasterizer import LabelRasterizer
from lifting.data.object_class import DEFAULT_CLASSES, PEDESTRIAN, VEHICLE, ObjectClass
from lifting.data.object_sampler import ObjectSampler
from lifting.data.ray_cast_renderer import RayCastRenderer, RenderResult
from lifting.data.scene_batch import SceneBatch
from lifting.data.scene_objects import SceneObjects
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset

# Stage-aware datasets (new)
from lifting.data.stages import TrainingPhase, PHASE_DATASET_SPECS, PHASE_NOTES

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
