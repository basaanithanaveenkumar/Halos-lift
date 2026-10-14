"""Per-plane channel projection for TPV planes."""

from __future__ import annotations

from torch import nn

from lifting.lifters.tpvformer.tpv_planes import TPVPlanes


class PlaneProjection(nn.Module):
    """Independent ``1x1`` convolution on each of the ``xy`` / ``xz`` / ``yz`` planes.

    Query-based lifters (BEVFormer, TPVFormer, Alpamayo) produce planes with the attention
    width ``in_channels``; this maps them to the lifter's ``out_channels``. The planes keep
    separate weights because they see the scene from different directions.

    Args:
        in_channels: input plane channels.
        out_channels: output plane channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.ModuleList(nn.Conv2d(in_channels, out_channels, 1) for _ in range(3))

    def forward(self, planes: TPVPlanes) -> TPVPlanes:
        xy, xz, yz = (
            p(x) for p, x in zip(self.proj, (planes.xy, planes.xz, planes.yz), strict=True)
        )
        return TPVPlanes(xy, xz, yz)
