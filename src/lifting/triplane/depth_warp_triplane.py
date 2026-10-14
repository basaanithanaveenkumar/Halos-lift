"""Depth-Warp -> TPV: depth-weighted frustum pulled into voxels, then three axis collapses."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.geometry import Cameras, DepthBins, GridSpec
from lifting.lifters.depth_warp_lifter import DepthWarpLifter
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.plane_compressor import VoxelToTriplane


@TRIPLANE_LIFTERS.register("depth_warp")
class DepthWarpTriplaneLifter(BaseTriplaneLifter):
    """Per-pixel depth distribution x context gives a frustum ``(C, D, H, W)`` per camera;
    every voxel *pulls* it trilinearly at its ``(u, v, depth)`` (Simple-BEV ``liftnet``).
    The voxel volume is then collapsed into the three planes.

    The frustum and warp are :meth:`DepthWarpLifter.frustum_volume` / :meth:`DepthWarpLifter.warp`,
    reused unchanged.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        depth_bins: depth discretisation of the frustum.
        context_channels: lifted feature width (defaults to ``in_channels``).
        temperature: depth softmax temperature.
        level: feature pyramid level to lift.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        depth_bins: DepthBins | None = None,
        context_channels: int | None = None,
        temperature: float = 1.0,
        level: int = 0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        ctx = context_channels or in_channels
        self.core = DepthWarpLifter(
            grid, in_channels, in_channels, depth_bins, ctx, temperature, level
        )
        self.core.compressor = nn.Identity()  # replaced by the plane compressors
        self.to_planes = VoxelToTriplane(ctx, grid, out_channels)

    @property
    def level(self) -> int:
        return self.core.level

    @property
    def depth_bins(self) -> DepthBins:
        return self.core.depth_bins

    def frustum_volume(self, features: Sequence[Tensor]) -> Tensor:
        """``(B, N, C', D, H, W)`` depth-weighted frustums of the chosen level."""
        return self.core.frustum_volume(features[self.level])

    def lift_voxels(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """``(B, C', X, Y, Z)`` voxel features pulled from the frustums."""
        return self.core.warp(self.frustum_volume(features), cameras)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        return self.to_planes(self.lift_voxels(features, cameras))
