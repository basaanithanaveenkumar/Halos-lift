"""Auxiliary per-plane head for plane-level supervision."""

from __future__ import annotations

from torch import Tensor, nn


def _num_groups(channels: int, max_groups: int = 4) -> int:
    """Largest divisor of ``channels`` that is at most ``max_groups``."""
    return next(g for g in range(min(max_groups, channels), 0, -1) if channels % g == 0)


class PlaneAuxHead(nn.Module):
    """Small conv head that classifies every cell of one TPV plane.

    Supervising each plane directly (see ``OccupancyTask(aux_weight=...)``) gives the plane
    features a gradient that does not have to pass through the aggregator.

    Args:
        channels: plane channels.
        num_classes: output classes per plane cell.
    """

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 1)
        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, plane: Tensor) -> Tensor:
        """``(B, C, H, W)`` -> ``(B, K, H, W)``."""
        return self.head(plane)
