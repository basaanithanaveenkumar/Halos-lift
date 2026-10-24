"""Exchange information between the BEV (``xy``) and side (``xz``) planes along their shared X."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lifting.exceptions import InvalidConfigError


class BEVSideMixer(nn.Module):
    """One layer of BEV <-> side cross-attention plus a per-plane convolutional FFN.

    The two planes share the forward axis ``x``. For every ``x`` row, the ``Y`` BEV cells
    attend to the ``Z`` side cells of the same row and vice versa, so a BEV cell learns the
    height profile of its row and a side cell learns the lateral layout of its row. Both
    directions read the *input* planes (a symmetric update). A depthwise 3x3 convolution in
    each plane's FFN then spreads information across rows.

    This is the two-plane counterpart of TPVFormer's cross-view hybrid attention, restricted to
    the one axis the planes have in common - cheap, because attention never spans more than
    ``max(Y, Z)`` tokens.

    Args:
        channels: plane channels.
        resolution: grid ``(X, Y, Z)``.
        num_heads: attention heads (must divide ``channels``).
        dropout: attention dropout.
    """

    def __init__(
        self,
        channels: int,
        resolution: tuple[int, int, int],
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise InvalidConfigError(f"num_heads={num_heads} must divide channels={channels}")
        _, ny, nz = resolution
        self.pos_y = nn.Parameter(0.02 * torch.randn(ny, channels))
        self.pos_z = nn.Parameter(0.02 * torch.randn(nz, channels))
        self.norm_bev = nn.LayerNorm(channels)
        self.norm_side = nn.LayerNorm(channels)
        self.bev_reads_side = nn.MultiheadAttention(channels, num_heads, dropout, batch_first=True)
        self.side_reads_bev = nn.MultiheadAttention(channels, num_heads, dropout, batch_first=True)
        self.ffn_bev = self._ffn(channels)
        self.ffn_side = self._ffn(channels)

    @staticmethod
    def _ffn(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, 2 * channels, 1),
            nn.GELU(),
            nn.Conv2d(2 * channels, channels, 1),
        )

    @staticmethod
    def _rows(plane: Tensor) -> Tensor:
        """``(B, C, X, L)`` -> ``(B * X, L, C)``: one token sequence per ``x`` row."""
        b, c, x, length = plane.shape
        return plane.permute(0, 2, 3, 1).reshape(b * x, length, c)

    @staticmethod
    def _plane(rows: Tensor, b: int, x: int) -> Tensor:
        """Inverse of :meth:`_rows`."""
        _, length, c = rows.shape
        return rows.view(b, x, length, c).permute(0, 3, 1, 2)

    def forward(self, bev: Tensor, side: Tensor) -> tuple[Tensor, Tensor]:
        """Mix ``bev`` ``(B, C, X, Y)`` and ``side`` ``(B, C, X, Z)``; same shapes out."""
        b, _, x, _ = bev.shape
        bev_rows, side_rows = self._rows(bev), self._rows(side)
        nb, ns = self.norm_bev(bev_rows), self.norm_side(side_rows)
        qb, qs = nb + self.pos_y[: nb.shape[1]], ns + self.pos_z[: ns.shape[1]]
        bev_rows = bev_rows + self.bev_reads_side(qb, qs, ns, need_weights=False)[0]
        side_rows = side_rows + self.side_reads_bev(qs, qb, nb, need_weights=False)[0]
        bev, side = self._plane(bev_rows, b, x), self._plane(side_rows, b, x)
        return bev + self.ffn_bev(bev), side + self.ffn_side(side)
