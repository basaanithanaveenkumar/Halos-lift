"""One axis-aligned box of the toy scene used by the triplane animations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyObject:
    """Axis-aligned box standing on the ground.

    Attributes:
        name: label shown in the animation.
        cls: class id (``1`` vehicle, ``2`` pedestrian, ``3`` pole).
        center: ``(x, y)`` of the footprint centre in metres.
        size: ``(dx, dy, dz)`` in metres.
    """

    name: str
    cls: int
    center: tuple[float, float]
    size: tuple[float, float, float]

    @property
    def mins(self) -> np.ndarray:
        """Lower box corner."""
        return np.array([self.center[0] - self.size[0] / 2, self.center[1] - self.size[1] / 2, 0.0])

    @property
    def maxs(self) -> np.ndarray:
        """Upper box corner."""
        return np.array(
            [self.center[0] + self.size[0] / 2, self.center[1] + self.size[1] / 2, self.size[2]]
        )

    def corners(self) -> list[tuple[float, float, float]]:
        """The eight box corners."""
        lo, hi = self.mins, self.maxs
        return [
            (float(x), float(y), float(z))
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]

    def contains(self, points: np.ndarray) -> np.ndarray:
        """``(..., 3)`` -> boolean mask of points inside the box."""
        return np.all((points >= self.mins) & (points <= self.maxs), axis=-1)
