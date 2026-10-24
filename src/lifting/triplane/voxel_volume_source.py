"""What a lifter must offer to lend its camera-sampled voxel volume to another lifter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from torch import Tensor

from lifting.geometry import Cameras


@runtime_checkable
class VoxelVolumeSource(Protocol):
    """A lifter that turns perspective (camera) features into a ``(B, C, X, Y, Z)`` volume.

    ``simple_bev``, ``depth_warp``, ``lift_splat`` and ``tiim`` satisfy it. The volume is the
    expensive, geometric part of lifting - every voxel has already read the cameras - so a
    lifter that only needs *some* planes can reuse it instead of sampling the images again.

    Attributes:
        voxel_channels: ``C`` of the volume.
    """

    voxel_channels: int

    def lift_voxels(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """``(B, voxel_channels, X, Y, Z)`` features sampled from the cameras."""
        ...
