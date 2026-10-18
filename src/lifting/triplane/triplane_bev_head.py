"""Fuse three TPV planes into a BEV map."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.layers import HeightCompressor
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes


class TriplaneBEVHead(nn.Module):
    """``TPVPlanes -> (B, C, X, Y, Z) voxels -> (B, out_channels, X, Y)`` BEV map.

    The voxel feature is the TPVFormer sum ``f(x, y, z) = xy(x, y) + xz(x, z) + yz(y, z)``;
    the height axis is then folded into channels and compressed (Simple-BEV style), exactly
    what :class:`TPVFormerLifter.forward` does - so all triplane lifters report BEV the same way.

    Args:
        channels: plane channels.
        height: number of height cells ``Z``.
        out_channels: BEV channels.
    """

    def __init__(self, channels: int, height: int, out_channels: int) -> None:
        super().__init__()
        self.compressor = HeightCompressor(channels, height, out_channels)

    def forward(self, planes: TPVPlanes) -> Tensor:
        """Collapse the planes to a BEV map ``(B, C, X, Y)``."""
        return self.compressor(planes.to_voxels())
