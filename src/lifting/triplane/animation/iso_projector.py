"""Isometric projection of the toy 3D scene onto the frame."""

from __future__ import annotations

import math
from collections.abc import Sequence

Point3 = Sequence[float]


class IsoProjector:
    """Maps ego-frame ``(x, y, z)`` to frame pixels.

    The viewer looks from the ``+x, +y`` corner, above the scene: ``+x`` runs to the lower right,
    ``+y`` to the lower left and ``+z`` straight up. The walls ``x = x_min`` and ``y = y_min``
    are therefore the visible back walls (where the ``yz`` and ``xz`` planes are drawn) and the
    floor ``z = 0`` holds the ``xy`` plane.

    Args:
        scale: pixels per metre.
        origin: frame pixel of the ego origin ``(0, 0, 0)``.
    """

    COS30 = math.cos(math.radians(30))
    SIN30 = 0.5

    def __init__(self, scale: float, origin: tuple[float, float]) -> None:
        self.scale = scale
        self.origin = origin

    def __call__(self, p: Point3) -> tuple[float, float]:
        x, y, z = p
        s = self.scale
        return (
            self.origin[0] + (x - y) * s * self.COS30,
            self.origin[1] + (x + y) * s * self.SIN30 - z * s,
        )

    def many(self, points: Sequence[Point3]) -> list[tuple[float, float]]:
        """Project several 3D points at once."""
        return [self(p) for p in points]

    @staticmethod
    def depth_key(p: Point3) -> float:
        """Painter's-algorithm key: larger is closer to the viewer (draw later)."""
        return p[0] + p[1] + 0.01 * p[2]
