"""Simple-BEV -> TPV: parameter-free bilinear unprojection, then three learned axis collapses."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.bilinear_sampling_lifter import BilinearSamplingLifter
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.plane_compressor import VoxelToTriplane


@TRIPLANE_LIFTERS.register("simple_bev")
class SimpleBEVTriplaneLifter(BaseTriplaneLifter):
    """Every voxel centre is projected into every camera and bilinearly sampled (Simple-BEV);
    the resulting ``(B, C, X, Y, Z)`` volume is collapsed along ``Z``, ``Y`` and ``X``.

    The voxel stage is :meth:`BilinearSamplingLifter.unproject`, reused unchanged.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        level: feature pyramid level to sample.
    """

    def __init__(self, grid: GridSpec, in_channels: int, out_channels: int, level: int = 0) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.core = BilinearSamplingLifter(grid, in_channels, in_channels, level=level)
        # The core's BEV compressor is replaced by the three plane compressors below; dropping
        # it avoids parameters that never receive a gradient.
        self.core.compressor = nn.Identity()
        self.to_planes = VoxelToTriplane(in_channels, grid, out_channels)

    @property
    def level(self) -> int:
        return self.core.level

    def lift_voxels(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """``(B, C, X, Y, Z)`` camera-averaged voxel features (no learning)."""
        return self.core.unproject(features[self.level], cameras)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        return self.to_planes(self.lift_voxels(features, cameras))
