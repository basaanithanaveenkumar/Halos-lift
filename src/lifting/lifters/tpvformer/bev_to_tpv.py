"""Convert a BEV feature map into three TPV planes."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from lifting.layers.conv_norm_act import ConvNormAct
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes


class BEVToTPV(nn.Module):
    """``(B, C_in, X, Y)`` BEV map -> :class:`TPVPlanes`, so any BEV lifter can feed a TPV decoder.

    * ``xy`` comes from the BEV map directly (two convs, then a resize to ``(X, Y)``).
    * ``xz`` average-pools the BEV map along ``Y`` and a 1x1 conv expands each column to ``Z``.
    * ``yz`` average-pools the BEV map along ``X`` and does the same.

    The height axis cannot be recovered from a BEV map, so ``xz`` / ``yz`` are *learned*
    expansions of the pooled columns - the inverse direction of :class:`HeightCompressor`.

    Args:
        in_channels: channels of the BEV map.
        out_channels: channels of every plane.
        resolution: target ``(X, Y, Z)`` plane resolution (usually ``grid.resolution``).
    """

    def __init__(
        self, in_channels: int, out_channels: int, resolution: tuple[int, int, int]
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.resolution = tuple(resolution)
        nz = self.resolution[2]
        self.proj_xy = nn.Sequential(
            ConvNormAct(in_channels, out_channels), ConvNormAct(out_channels, out_channels)
        )
        self.proj_xz = self._column_expander(in_channels, out_channels, nz)
        self.proj_yz = self._column_expander(in_channels, out_channels, nz)

    @staticmethod
    def _column_expander(in_channels: int, out_channels: int, nz: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels * nz, 1, bias=False),
            nn.BatchNorm1d(out_channels * nz),
            nn.ReLU(inplace=True),
        )

    def _expand(self, column: Tensor, proj: nn.Module, length: int) -> Tensor:
        """``(B, C_in, L0)`` -> ``(B, C_out, L, Z)``."""
        column = F.interpolate(column, size=length, mode="linear", align_corners=False)
        b = column.shape[0]
        out = proj(column).reshape(b, self.out_channels, self.resolution[2], length)
        return out.transpose(2, 3)

    def forward(self, bev: Tensor) -> TPVPlanes:
        nx, ny, _ = self.resolution
        xy = F.interpolate(self.proj_xy(bev), size=(nx, ny), mode="bilinear", align_corners=False)
        xz = self._expand(bev.mean(dim=3), self.proj_xz, nx)
        yz = self._expand(bev.mean(dim=2), self.proj_yz, ny)
        return TPVPlanes(xy, xz, yz)
