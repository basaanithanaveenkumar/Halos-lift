"""Collapse one axis of a voxel volume with learned, content-dependent attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lifting.exceptions import InvalidConfigError


class AxisAttentionPool(nn.Module):
    """``(B, C, P, Q, L) -> (B, out_channels, P, Q)`` by attending over the last axis ``L``.

    Every ``(p, q)`` column of ``L`` cells is reduced to one feature: each head scores the
    cells (from their content plus a learned position embedding along ``L``), softmaxes the
    scores over ``L`` and takes the weighted sum of the cells' values. For the BEV plane ``L``
    is height - each pillar decides which heights matter; for the side plane ``L`` is the
    lateral axis - each ``(x, z)`` line decides which lateral positions matter.

    Compared with folding the axis into channels (:class:`~lifting.layers.HeightCompressor`),
    attention pooling has ``O(C^2)`` parameters independent of ``L``, keeps the reduction
    interpretable (:attr:`last_weights`) and adapts to the scene; the position embedding keeps
    *where* along ``L`` a feature came from.

    Args:
        channels: volume channels ``C``.
        length: size of the pooled axis ``L``.
        out_channels: output plane channels.
        num_heads: attention heads (must divide ``channels``).
    """

    def __init__(self, channels: int, length: int, out_channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads:
            raise InvalidConfigError(f"num_heads={num_heads} must divide channels={channels}")
        self.num_heads = num_heads
        self.position = nn.Parameter(0.02 * torch.randn(channels, length))
        self.score = nn.Conv3d(channels, num_heads, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.output = nn.Conv2d(channels, out_channels, 1)
        #: ``(B, heads, P, Q, L)`` attention of the last forward (detached), for inspection.
        self.last_weights: Tensor | None = None

    def forward(self, volume: Tensor) -> Tensor:
        """Pool the last axis of ``volume`` ``(B, C, P, Q, L)``."""
        b, c, p, q, length = volume.shape
        x = volume + self.position[None, :, None, None, :length]
        weights = self.score(x).softmax(dim=-1)  # (B, h, P, Q, L)
        values = self.value(x).view(b, self.num_heads, c // self.num_heads, p, q, length)
        pooled = (values * weights[:, :, None]).sum(-1).reshape(b, c, p, q)
        self.last_weights = weights.detach()
        return self.output(pooled)
