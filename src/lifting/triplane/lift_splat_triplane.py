"""Lift-Splat-Shoot -> TPV: frustum point cloud splatted into voxels, then three axis collapses."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.geometry import Cameras, DepthBins, GridSpec
from lifting.lifters.lift_splat_lifter import LiftSplatLifter
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.plane_compressor import VoxelToTriplane


@TRIPLANE_LIFTERS.register("lift_splat")
class LiftSplatTriplaneLifter(BaseTriplaneLifter):
    """Lift: outer product of each pixel's depth distribution and context feature.
    Splat: scatter-sum every frustum point into its voxel (Philion & Fidler, 2020).
    Shoot (here): collapse the voxel volume into the ``xy`` / ``xz`` / ``yz`` planes.

    The depth head and splat are :class:`LiftSplatLifter`'s, reused unchanged.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        depth_bins: depth discretisation of the frustum.
        context_channels: lifted feature width (defaults to ``in_channels``).
        level: feature pyramid level to lift.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        depth_bins: DepthBins | None = None,
        context_channels: int | None = None,
        level: int = 0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.core = LiftSplatLifter(
            grid, in_channels, in_channels, depth_bins, context_channels, level
        )
        self.core.compressor = nn.Identity()  # replaced by the plane compressors
        self.to_planes = VoxelToTriplane(self.core.context_channels, grid, out_channels)

    @property
    def level(self) -> int:
        """Index of the feature-pyramid level the wrapped BEV lifter reads."""
        return self.core.level

    @property
    def depth_bins(self) -> DepthBins:
        """The depth discretisation of the wrapped depth lifter."""
        return self.core.depth_bins

    def depth_and_context(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        """``(B, N, D, H, W)`` depth probabilities and ``(B, N, C', H, W)`` context."""
        feats = features[self.level]
        b, n = feats.shape[:2]
        depth, context = self.core.depth_head(feats.flatten(0, 1))
        return depth.view(b, n, *depth.shape[1:]), context.view(b, n, *context.shape[1:])

    def lift_voxels(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """``(B, C', X, Y, Z)`` sum-pooled frustum features."""
        depth, context = self.depth_and_context(features)
        h, w = context.shape[-2:]
        return self.core.splat(depth, context, cameras.scaled_to(h, w))

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Lift per-camera features into the three TPV planes.

        Args:
            features: feature pyramid from the image encoder.
            cameras: the rig geometry.

        Returns:
            The ``xy`` / ``xz`` / ``yz`` feature planes.
        """
        return self.to_planes(self.lift_voxels(features, cameras))
