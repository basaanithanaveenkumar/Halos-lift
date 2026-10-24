"""TIIM -> TPV: polar rays give range, image rows give elevation; voxels read both."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lifting.geometry import Cameras, DepthBins, GridSpec
from lifting.lifters.polar_ray_lifter import PolarRayLifter
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.ops import masked_camera_mean, sample_multi_camera
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.plane_compressor import VoxelToTriplane


@TRIPLANE_LIFTERS.register("tiim")
class PolarRayTriplaneLifter(BaseTriplaneLifter):
    """Translating-Images-into-Maps extended to three planes.

    TIIM's transformer squeezes every image *column* into a polar ray of ``D`` depth cells,
    ``(C, D, W)`` per camera - the row (elevation) axis is consumed. The BEV version samples
    those rays at one height. For a full 3D volume each voxel ``(x, y, z)`` instead reads:

    1. **range feature**: the polar ray at its ``(column u, depth d)`` - *what is at this
       distance along this column* (TIIM's learned depth reasoning);
    2. **elevation feature**: the image feature at its pixel ``(u, v)`` - the row ``v`` encodes
       the voxel's height, which the polar ray cannot.

    Both are camera-averaged over the cameras that see the voxel and concatenated
    (``2C`` channels), then collapsed into the three planes.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        depth_bins: radial discretisation of the rays.
        num_layers: transformer layers in the column encoder and ray decoder.
        num_heads: attention heads.
        level: feature pyramid level to translate.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        depth_bins: DepthBins | None = None,
        num_layers: int = 1,
        num_heads: int = 4,
        level: int = 0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.core = PolarRayLifter(
            grid, in_channels, in_channels, depth_bins, num_layers, num_heads, level
        )
        self.core.output = nn.Identity()  # BEV output head replaced by the plane compressors
        #: Channels of the voxel volume :meth:`lift_voxels` returns (``VoxelVolumeSource``).
        self.voxel_channels = 2 * in_channels
        self.to_planes = VoxelToTriplane(self.voxel_channels, grid, out_channels)

    @property
    def level(self) -> int:
        """Index of the feature-pyramid level the wrapped BEV lifter reads."""
        return self.core.level

    @property
    def depth_bins(self) -> DepthBins:
        """The depth discretisation of the wrapped depth lifter."""
        return self.core.depth_bins

    def image_to_polar(self, features: Sequence[Tensor]) -> Tensor:
        """``(B, N, C, D, W)`` polar rays of the chosen level (TIIM, unchanged)."""
        return self.core.image_to_polar(features[self.level])

    def polar_to_voxels(self, polar: Tensor, image: Tensor, cameras: Cameras) -> Tensor:
        """Resample polar rays and image features at every voxel centre.

        Args:
            polar: ``(B, N, C, D, W)`` polar rays.
            image: ``(B, N, C, H, W)`` image features of the same level.
            cameras: calibration (input-image resolution).

        Returns:
            ``(B, 2C, X, Y, Z)``: channels ``[:C]`` range features, ``[C:]`` elevation features.
        """
        b, n, c, d, w = polar.shape
        nx, ny, nz = self.grid.resolution
        points = self.grid.voxel_centers(polar.device).view(1, -1, 3).expand(b, -1, -1)
        uv, depth, valid = cameras.project(points)

        # (1) range: grid_sample over (depth rows, column cols), same convention as TIIM's BEV.
        u_norm = 2.0 * uv[..., 0] / cameras.image_size[1] - 1.0
        d_norm = self.depth_bins.to_normalized(depth)
        polar_valid = (depth > 0) & (u_norm.abs() < 1.0) & self.depth_bins.contains(depth)
        grid = torch.stack([u_norm, d_norm], -1).reshape(b * n, 1, -1, 2)
        ranged = F.grid_sample(
            polar.reshape(b * n, c, d, w),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(b, n, c, -1)
        range_vox = masked_camera_mean(ranged, polar_valid)

        # (2) elevation: plain bilinear read of the pixel the voxel projects to.
        sampled = sample_multi_camera(image, cameras.normalize_uv(uv))
        elev_vox = masked_camera_mean(sampled, valid)

        return torch.cat([range_vox, elev_vox], 1).view(b, 2 * c, nx, ny, nz)

    def lift_voxels(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """Lift features to a voxel volume through polar rays (the wrapped lifter's geometry)."""
        image = features[self.level]
        return self.polar_to_voxels(self.image_to_polar(features), image, cameras)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Lift per-camera features into the three TPV planes.

        Args:
            features: feature pyramid from the image encoder.
            cameras: the rig geometry.

        Returns:
            The ``xy`` / ``xz`` / ``yz`` feature planes.
        """
        return self.to_planes(self.lift_voxels(features, cameras))
