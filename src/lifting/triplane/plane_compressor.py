"""Extract all three TPV planes from a dense voxel volume ``(B, C, X, Y, Z)``."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.geometry import GridSpec
from lifting.layers import HeightCompressor
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes


class VoxelToTriplane(nn.Module):
    """Collapse one axis at a time with a learned compressor (inverse of ``to_voxels``).

    * ``xy``: fold ``Z`` into channels and convolve over ``(X, Y)`` - the usual BEV compressor.
    * ``xz``: fold ``Y`` into channels and convolve over ``(X, Z)``.
    * ``yz``: fold ``X`` into channels and convolve over ``(Y, Z)``.

    Folding (instead of mean / max pooling) keeps *where along the collapsed axis* a feature
    was, so e.g. the ``xz`` plane can still tell a car on the left from one on the right.
    Used by every lifter that already builds a full voxel volume (Simple-BEV, Depth-Warp,
    Lift-Splat, TIIM): their planes are therefore geometrically grounded on all three axes.

    Args:
        in_channels: voxel feature channels.
        grid: provides the ``(X, Y, Z)`` resolution.
        out_channels: channels of every plane.
    """

    def __init__(self, in_channels: int, grid: GridSpec, out_channels: int) -> None:
        super().__init__()
        nx, ny, nz = grid.resolution
        self.compress_xy = HeightCompressor(in_channels, nz, out_channels)
        self.compress_xz = HeightCompressor(in_channels, ny, out_channels)
        self.compress_yz = HeightCompressor(in_channels, nx, out_channels)

    def forward(self, voxels: Tensor) -> TPVPlanes:
        """``(B, C, X, Y, Z)`` -> :class:`TPVPlanes`."""
        # HeightCompressor always collapses the *last* axis, so move the axis to drop there.
        xy = self.compress_xy(voxels)  # collapse Z
        xz = self.compress_xz(voxels.permute(0, 1, 2, 4, 3))  # (B, C, X, Z, Y) -> collapse Y
        yz = self.compress_yz(voxels.permute(0, 1, 3, 4, 2))  # (B, C, Y, Z, X) -> collapse X
        return TPVPlanes(xy=xy, xz=xz, yz=yz)
