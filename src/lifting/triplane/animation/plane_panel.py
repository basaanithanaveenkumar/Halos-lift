"""Flat view of the three output planes, shown below the camera images."""

from __future__ import annotations

import numpy as np

from lifting.triplane.animation.canvas import Canvas
from lifting.triplane.animation.theme import Theme
from lifting.triplane.animation.toy_scene import ToyScene


class PlanePanel:
    """``xy`` (BEV, ``x`` up / ``y`` left like a map), ``xz`` and ``yz`` (``z`` up) as small grids.

    Args:
        scene: provides plane resolutions.
        theme: colours.
        origin: top-left pixel of the panel.
    """

    XY = 104
    STRIP = (150, 50)

    def __init__(self, scene: ToyScene, theme: Theme, origin: tuple[int, int]) -> None:
        self.scene = scene
        self.theme = theme
        self.origin = origin

    def _boxes(self) -> list[tuple[float, float, float, float]]:
        x, y = self.origin
        sw, sh = self.STRIP
        return [
            (x, y + 18, x + self.XY, y + 18 + self.XY),
            (x + self.XY + 18, y + 18, x + self.XY + 18 + sw, y + 18 + sh),
            (x + self.XY + 18, y + 18 + sh + 22, x + self.XY + 18 + sw, y + 18 + 2 * sh + 22),
        ]

    def _shape(self, plane: int) -> tuple[int, int]:
        nx, ny, nz = self.scene.resolution
        return [(nx, ny), (nx, nz), (ny, nz)][plane]

    def cell_box(self, plane: int, a: int, b: int) -> tuple[float, float, float, float]:
        """Pixel box of plane cell ``(a, b)``."""
        x0, y0, x1, y1 = self._boxes()[plane]
        h, w = self._shape(plane)
        if plane == 0:  # rows = x (forward up), cols = y (left to the left)
            col, row, ncol, nrow = w - 1 - b, h - 1 - a, w, h
        else:  # columns = first axis, rows = z (up)
            col, row, ncol, nrow = a, w - 1 - b, h, w
        cw, ch = (x1 - x0) / ncol, (y1 - y0) / nrow
        return x0 + col * cw, y0 + row * ch, x0 + (col + 1) * cw, y0 + (row + 1) * ch

    def draw(self, canvas: Canvas, title: str = "output triplane") -> None:
        """Draw the three planes with their axes and a title."""
        canvas.text((self.origin[0], self.origin[1] - 4), title, 13, self.theme.muted)
        names = ("xy = BEV  (x↑ y←)", "xz  (x→ z↑)", "yz  (y→ z↑)")
        for plane, box in enumerate(self._boxes()):
            color = self.theme.planes[plane]
            canvas.rect(box, self.theme.rgba(color, 0.06), self.theme.rgba(color, 0.9))
            canvas.text((box[0], box[3] + 3), names[plane], 11, color)

    def fill(
        self, canvas: Canvas, plane: int, labels: np.ndarray, alpha: float = 1.0, order: float = 1.0
    ) -> None:
        """Colour cells by class; ``order`` reveals them progressively in raster order."""
        h, w = labels.shape
        limit = int(round(order * h * w))
        for n, (a, b) in enumerate(np.ndindex(h, w)):
            if n >= limit:
                break
            cls = int(labels[a, b])
            color = self.theme.classes[cls] if cls > 0 else self.theme.planes[plane]
            canvas.rect(
                self.cell_box(plane, a, b), self.theme.rgba(color, (0.9 if cls else 0.13) * alpha)
            )

    def grid_lines(self, canvas: Canvas, plane: int, alpha: float = 0.35) -> None:
        """Draw the cell grid of one plane."""
        h, w = self._shape(plane)
        color = self.theme.rgba(self.theme.planes[plane], alpha)
        for a, b in np.ndindex(h, w):
            canvas.rect(self.cell_box(plane, a, b), None, color)

    def queries(self, canvas: Canvas, plane: int, alpha: float = 1.0) -> None:
        """Draw the query cell centres of one plane."""
        h, w = self._shape(plane)
        for a, b in np.ndindex(h, w):
            x0, y0, x1, y1 = self.cell_box(plane, a, b)
            canvas.circle(
                ((x0 + x1) / 2, (y0 + y1) / 2),
                1.6,
                self.theme.rgba(self.theme.planes[plane], alpha),
            )

    def drop(self, canvas: Canvas, plane: int, label: str = "not computed") -> None:
        """Grey out a plane the technique does not produce and cross it through."""
        x0, y0, x1, y1 = self._boxes()[plane]
        canvas.rect((x0, y0, x1, y1), self.theme.rgba(self.theme.background, 0.85))
        dim = self.theme.rgba(self.theme.muted, 0.7)
        canvas.line([(x0, y0), (x1, y1)], dim, 1)
        canvas.line([(x0, y1), (x1, y0)], dim, 1)
        canvas.text(((x0 + x1) / 2, (y0 + y1) / 2), label, 11, self.theme.muted, anchor="mm")

    def center(self, plane: int) -> tuple[float, float]:
        """Pixel centre of a plane's box."""
        x0, y0, x1, y1 = self._boxes()[plane]
        return (x0 + x1) / 2, (y0 + y1) / 2

    def highlight(self, canvas: Canvas, plane: int, a: int, b: int) -> None:
        """Highlight cell ``(a, b)`` of one plane."""
        canvas.rect(self.cell_box(plane, a, b), None, self.theme.ray, 2)
