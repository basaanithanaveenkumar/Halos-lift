"""Draws the toy 3D scene (grid, planes, objects, cameras, rays, voxels) in isometric view."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from lifting.triplane.animation.canvas import Canvas
from lifting.triplane.animation.iso_projector import IsoProjector
from lifting.triplane.animation.theme import RGB, Theme
from lifting.triplane.animation.toy_rig import ToyRig
from lifting.triplane.animation.toy_scene import ToyScene

Point3 = Sequence[float]
PLANE_NAMES = ("xy  (BEV)", "xz  (side)", "yz  (front)")


class ScenePainter:
    """All 3D drawing of one frame. Every method is pure drawing; nothing is cached here.

    Args:
        canvas: target frame.
        projector: 3D -> pixel mapping.
        scene: toy world.
        rig: toy cameras.
    """

    def __init__(
        self, canvas: Canvas, projector: IsoProjector, scene: ToyScene, rig: ToyRig
    ) -> None:
        self.canvas = canvas
        self.proj = projector
        self.scene = scene
        self.rig = rig
        self.theme: Theme = canvas.theme

    # ------------------------------------------------------------------ static scene
    def floor_grid(self, alpha: float = 0.55) -> None:
        """Draw the ground-plane grid."""
        col = self.theme.rgba(self.theme.grid, alpha)
        (x0, x1), (y0, y1) = self.scene.x_bounds, self.scene.y_bounds
        for x in self.scene.edges(0):
            self.canvas.line(self.proj.many([(x, y0, 0), (x, y1, 0)]), col)
        for y in self.scene.edges(1):
            self.canvas.line(self.proj.many([(x0, y, 0), (x1, y, 0)]), col)

    def back_walls(self, alpha: float = 0.35) -> None:
        """Faint cell grid on the two back walls where the ``xz`` / ``yz`` planes live."""
        col = self.theme.rgba(self.theme.grid, alpha)
        (x0, x1), (y0, y1), (z0, z1) = self.scene.bounds
        for x in self.scene.edges(0):
            self.canvas.line(self.proj.many([(x, y0, z0), (x, y0, z1)]), col)
        for y in self.scene.edges(1):
            self.canvas.line(self.proj.many([(x0, y, z0), (x0, y, z1)]), col)
        for z in self.scene.edges(2):
            self.canvas.line(self.proj.many([(x0, y0, z), (x1, y0, z)]), col)
            self.canvas.line(self.proj.many([(x0, y0, z), (x0, y1, z)]), col)

    def bounding_box(self, alpha: float = 0.8) -> None:
        """Draw the edges of the scene volume."""
        col = self.theme.rgba(self.theme.box, alpha)
        (x0, x1), (y0, y1), (z0, z1) = self.scene.bounds
        top = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1)]
        self.canvas.line(self.proj.many(top), col)
        for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            self.canvas.line(self.proj.many([(x, y, z0), (x, y, z1)]), col)
        bottom = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z0)]
        self.canvas.line(self.proj.many(bottom), col)

    def plane_outline(
        self, plane: int, alpha: float = 1.0, label: bool = True, width: int = 2
    ) -> None:
        """Draw the outline of one TPV plane inside the volume."""
        corners = self._plane_corners(plane)
        color = self.theme.planes[plane]
        self.canvas.polygon(
            self.proj.many(corners),
            self.theme.rgba(color, 0.07 * alpha),
            self.theme.rgba(color, alpha),
            width,
        )
        if label:
            anchor = self.proj(self._plane_label_point(plane))
            self.canvas.text(
                anchor, PLANE_NAMES[plane], 12, self.theme.rgba(color, alpha), anchor="mm"
            )

    def _plane_corners(self, plane: int) -> list[tuple[float, float, float]]:
        (x0, x1), (y0, y1), (z0, z1) = self.scene.bounds
        if plane == 0:
            return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
        if plane == 1:
            return [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
        return [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)]

    def _plane_label_point(self, plane: int) -> tuple[float, float, float]:
        (x0, x1), (y0, y1), (z0, z1) = self.scene.bounds
        if plane == 0:
            return (x1 + 0.9, y1 + 0.9, 0.0)
        if plane == 1:
            return (0.5 * (x0 + x1), y0, z1 + 0.45)
        return (x0, 0.5 * (y0 + y1), z1 + 0.45)

    def plane_cells(
        self,
        plane: int,
        labels: np.ndarray,
        alpha: float = 1.0,
        empty_alpha: float = 0.10,
        order: float = 1.0,
    ) -> None:
        """Fill the cells of ``plane``: occupied cells in their class colour, empty cells tinted.

        ``order`` in ``[0, 1]`` reveals cells progressively in raster order (for "filling" animations).
        """
        h, w = labels.shape
        limit = int(round(order * h * w))
        for n, (a, b) in enumerate(np.ndindex(h, w)):
            if n >= limit:
                break
            cls = int(labels[a, b])
            if cls > 0:
                fill = self.theme.rgba(self.theme.classes[cls], 0.85 * alpha)
            else:
                fill = self.theme.rgba(self.theme.planes[plane], empty_alpha * alpha)
            self.canvas.polygon(
                self.proj.many(self.scene.plane_cell(plane, a, b)),
                fill,
                self.theme.rgba(self.theme.planes[plane], 0.35 * alpha),
            )

    def plane_queries(self, plane: int, alpha: float = 1.0, radius: float = 2.2) -> None:
        """A dot at every cell centre of ``plane``: the learned queries of attention lifters."""
        color = self.theme.rgba(self.theme.planes[plane], alpha)
        nx, ny, nz = self.scene.resolution
        shape = [(nx, ny), (nx, nz), (ny, nz)][plane]
        for a, b in np.ndindex(*shape):
            cell = np.array(self.scene.plane_cell(plane, a, b)).mean(0)
            self.canvas.circle(self.proj(cell), radius, color)

    # ------------------------------------------------------------------ solids
    def cuboid(
        self, lo: Point3, hi: Point3, color: RGB, alpha: float = 1.0, outline: bool = True
    ) -> None:
        """Box with its three visible faces (top, ``+x``, ``+y``) shaded differently."""
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        faces = [
            ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], 0.78),  # +x face
            ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], 0.6),  # +y face
            ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], 1.0),  # top
        ]
        edge = self.theme.rgba(self.theme.shade(color, 1.35), 0.9 * alpha) if outline else None
        for pts, shade in faces:
            self.canvas.polygon(
                self.proj.many(pts), self.theme.rgba(self.theme.shade(color, shade), alpha), edge
            )

    def objects(self, alpha: float = 1.0) -> None:
        """Draw the scene objects as boxes."""
        for obj in sorted(self.scene.objects, key=lambda o: self.proj.depth_key((*o.center, 0.0))):
            self.cuboid(obj.mins, obj.maxs, self.theme.classes[obj.cls], alpha)

    def voxels(
        self, cells: Iterable[tuple[int, int, int, RGB, float]], shrink: float = 0.82
    ) -> None:
        """Draw ``(i, j, k, colour, alpha)`` voxels back to front."""
        cx, cy, cz = self.scene.cell
        items = []
        for i, j, k, color, alpha in cells:
            c = self.scene.voxel_center(i, j, k)
            items.append((self.proj.depth_key(c), c, color, alpha))
        for _, c, color, alpha in sorted(items, key=lambda it: it[0]):
            half = np.array([cx, cy, cz]) * shrink / 2
            self.cuboid(np.array(c) - half, np.array(c) + half, color, alpha)

    # ------------------------------------------------------------------ cameras and rays
    def cameras(self, highlight: Sequence[int] = (), alpha: float = 1.0) -> None:
        """Draw the camera rig; cameras in ``highlight`` are emphasised."""
        for idx, cam in enumerate(self.rig.cameras):
            color = self.theme.ray if idx in highlight else self.theme.camera
            o = np.asarray(cam.position)
            tip = [
                o + 0.45 * (cam.forward + sx * 0.6 * cam.right + sz * 0.35 * cam.up)
                for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
            pts = self.proj.many(tip)
            self.canvas.polygon(
                pts, self.theme.rgba(color, 0.25 * alpha), self.theme.rgba(color, alpha)
            )
            for p in pts:
                self.canvas.line([self.proj(o), p], self.theme.rgba(color, alpha))
        self.canvas.circle(self.proj((0, 0, 0.9)), 3, self.theme.rgba(self.theme.camera, alpha))

    def fov_fan(self, cam: int, depth: float = 4.5, alpha: float = 0.12) -> None:
        """Ground footprint of a camera's horizontal field of view."""
        c = self.rig.cameras[cam]
        o = np.array([c.position[0], c.position[1], 0.0])
        a = o + depth * (c.forward - self.rig.tan_h * c.right)
        b = o + depth * (c.forward + self.rig.tan_h * c.right)
        self.canvas.polygon(self.proj.many([o, a, b]), self.theme.rgba(self.theme.ray, alpha))

    def ray(
        self,
        a: Point3,
        b: Point3,
        t: float = 1.0,
        color: RGB | None = None,
        alpha: float = 0.9,
        width: int = 1,
    ) -> None:
        """Draw a camera ray with optional marker."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        end = a + (b - a) * max(0.0, min(1.0, t))
        self.canvas.line(
            [self.proj(a), self.proj(end)], self.theme.rgba(color or self.theme.ray, alpha), width
        )

    def dashed(self, a: Point3, b: Point3, color: RGB, alpha: float = 0.8) -> None:
        """Draw a dashed segment between two 3D points."""
        self.canvas.dashed(self.proj(a), self.proj(b), self.theme.rgba(color, alpha))

    def dot(self, p: Point3, color: RGB, radius: float = 3.0, alpha: float = 1.0) -> None:
        """Draw a dot at a 3D point."""
        self.canvas.circle(self.proj(p), radius, self.theme.rgba(color, alpha))

    def label(self, p: Point3, text: str, color: RGB | None = None, size: int = 12) -> None:
        """Draw a text label at a 3D point."""
        self.canvas.text(self.proj(p), text, size, color or self.theme.text, anchor="mm")
