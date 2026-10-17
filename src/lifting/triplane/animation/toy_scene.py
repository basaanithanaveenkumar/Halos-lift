"""A tiny metric world (grid + three boxes) with exact occupancy and plane labels.

Classes moved to their own modules (one public class per file) are re-exported here so
existing ``from lifting.triplane.animation.toy_scene import ...`` statements keep working."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lifting.triplane.animation.toy_object import ToyObject


@dataclass
class ToyScene:
    """Region ``[-4, 4] x [-4, 4] x [0, 2]`` m cut into ``8 x 8 x 4`` cells, plus three objects.

    Mirrors :class:`lifting.geometry.GridSpec` conventions (cell centres, ``(X, Y, Z)`` order) so
    the animated quantities match what the real lifters compute, only at toy size.
    """

    x_bounds: tuple[float, float] = (-4.0, 4.0)
    y_bounds: tuple[float, float] = (-4.0, 4.0)
    z_bounds: tuple[float, float] = (0.0, 2.0)
    resolution: tuple[int, int, int] = (8, 8, 4)
    objects: list[ToyObject] = field(
        default_factory=lambda: [
            ToyObject("vehicle", 1, (2.4, 1.3), (1.9, 1.0, 0.9)),
            ToyObject("pedestrian", 2, (-1.4, -2.6), (0.5, 0.5, 1.6)),
            ToyObject("pole", 3, (-2.5, 2.5), (0.35, 0.35, 2.0)),
        ]
    )

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return (self.x_bounds, self.y_bounds, self.z_bounds)

    @property
    def cell(self) -> tuple[float, float, float]:
        return tuple((b[1] - b[0]) / n for b, n in zip(self.bounds, self.resolution, strict=True))  # type: ignore[return-value]

    def edges(self, axis: int) -> np.ndarray:
        lo, hi = self.bounds[axis]
        return np.linspace(lo, hi, self.resolution[axis] + 1)

    def centers(self, axis: int) -> np.ndarray:
        e = self.edges(axis)
        return 0.5 * (e[1:] + e[:-1])

    def voxel_centers(self) -> np.ndarray:
        """``(X, Y, Z, 3)`` metric centres."""
        return np.stack(np.meshgrid(*(self.centers(a) for a in range(3)), indexing="ij"), -1)

    def voxel_center(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        return float(self.centers(0)[i]), float(self.centers(1)[j]), float(self.centers(2)[k])

    def occupancy(self) -> np.ndarray:
        """``(X, Y, Z)`` class ids (0 = empty), sampled at a 3x3x3 sub-grid per voxel."""
        nx, ny, nz = self.resolution
        sub = (np.arange(3) + 0.5) / 3
        occ = np.zeros((nx, ny, nz), dtype=np.int64)
        ex, ey, ez = (self.edges(a) for a in range(3))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pts = np.stack(
                        np.meshgrid(
                            ex[i] + sub * (ex[i + 1] - ex[i]),
                            ey[j] + sub * (ey[j + 1] - ey[j]),
                            ez[k] + sub * (ez[k + 1] - ez[k]),
                            indexing="ij",
                        ),
                        -1,
                    )
                    for obj in self.objects:
                        if obj.contains(pts).any():
                            occ[i, j, k] = max(occ[i, j, k], obj.cls)
        return occ

    def plane_labels(self, occupancy: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
        """``xy (X, Y)``, ``xz (X, Z)``, ``yz (Y, Z)``: any occupied voxel marks its column
        (the same collapse ``OccupancyTask.plane_loss`` supervises)."""
        occ = self.occupancy() if occupancy is None else occupancy
        return occ.max(2), occ.max(1), occ.max(0)

    def plane_cell(self, plane: int, a: int, b: int) -> list[tuple[float, float, float]]:
        """Four 3D corners of cell ``(a, b)`` of ``plane`` drawn on the floor / back walls.

        ``xy`` lies on ``z = 0``, ``xz`` on the wall ``y = y_min``, ``yz`` on ``x = x_min``.
        """
        ex, ey, ez = (self.edges(i) for i in range(3))
        if plane == 0:
            x0, x1, y0, y1 = ex[a], ex[a + 1], ey[b], ey[b + 1]
            return [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
        if plane == 1:
            x0, x1, z0, z1 = ex[a], ex[a + 1], ez[b], ez[b + 1]
            y = self.y_bounds[0]
            return [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        y0, y1, z0, z1 = ey[a], ey[a + 1], ez[b], ez[b + 1]
        x = self.x_bounds[0]
        return [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]

    def inside(self, p) -> bool:
        """Whether metric point ``p`` lies inside the grid (points outside are discarded)."""
        return all(lo <= c < hi for c, (lo, hi) in zip(p, self.bounds, strict=True))

    def project_to_plane(
        self, p: tuple[float, float, float], plane: int
    ) -> tuple[float, float, float]:
        """Orthogonal "shadow" of ``p`` on the drawn plane."""
        x, y, z = p
        if plane == 0:
            return x, y, 0.0
        if plane == 1:
            return x, self.y_bounds[0], z
        return self.x_bounds[0], y, z

__all__ = ["ToyObject"]
