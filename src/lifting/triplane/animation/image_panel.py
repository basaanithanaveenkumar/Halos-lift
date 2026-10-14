"""The four camera images (rendered from the toy scene) shown on the right of every frame."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lifting.triplane.animation.canvas import Canvas
from lifting.triplane.animation.theme import RGB, Theme
from lifting.triplane.animation.toy_rig import ToyRig


class CameraImages:
    """Camera thumbnails (from the ray-cast class images) and their placement on the frame.

    Args:
        labels: per-camera ``(H, W)`` class-id images (:attr:`SceneFacts.images`).
        rig: camera names.
        theme: colours.
        origin: top-left pixel of the 2x2 thumbnail block.
        thumb: ``(width, height)`` of one thumbnail (nearest-neighbour upsampled).
    """

    GAP = 8
    LABEL = 15

    def __init__(
        self,
        labels: list[np.ndarray],
        rig: ToyRig,
        theme: Theme,
        origin: tuple[int, int],
        thumb: tuple[int, int] = (150, 66),
    ) -> None:
        self.rig = rig
        self.theme = theme
        self.origin = origin
        self.thumb = thumb
        self.labels = labels
        self.thumbnails = [self._colorize(lbl) for lbl in self.labels]

    def _colorize(self, labels: np.ndarray) -> Image.Image:
        lut = {-1: self.theme.sky, 0: self.theme.ground, **self.theme.classes}
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for cls, color in lut.items():
            rgb[labels == cls] = color
        return Image.fromarray(rgb).resize(self.thumb, Image.NEAREST)

    def box(self, cam: int) -> tuple[int, int, int, int]:
        """Pixel box ``(x0, y0, x1, y1)`` of camera ``cam``'s thumbnail."""
        col, row = cam % 2, cam // 2
        x = self.origin[0] + col * (self.thumb[0] + self.GAP)
        y = self.origin[1] + self.LABEL + row * (self.thumb[1] + self.GAP + self.LABEL)
        return x, y, x + self.thumb[0], y + self.thumb[1]

    def to_screen(self, cam: int, u: float, v: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box(cam)
        return x0 + u * (x1 - x0), y0 + v * (y1 - y0)

    def draw(self, canvas: Canvas, highlight: tuple[int, ...] = ()) -> None:
        canvas.text((self.origin[0], self.origin[1] - 16), "camera images", 13, self.theme.muted)
        for cam, img in enumerate(self.thumbnails):
            x0, y0, x1, y1 = self.box(cam)
            canvas.paste(img, (x0, y0))
            edge = self.theme.ray if cam in highlight else self.theme.panel_edge
            canvas.rect((x0 - 1, y0 - 1, x1, y1), None, edge, 2 if cam in highlight else 1)
            canvas.text(
                (x0, y0 - 14), f"cam {cam}: {self.rig.cameras[cam].name}", 11, self.theme.muted
            )

    def highlight(self, canvas: Canvas, cam: int) -> None:
        x0, y0, x1, y1 = self.box(cam)
        canvas.rect((x0 - 2, y0 - 2, x1 + 1, y1 + 1), None, self.theme.ray, 2)

    def dot(
        self,
        canvas: Canvas,
        cam: int,
        u: float,
        v: float,
        color: RGB,
        radius: float = 3.0,
        alpha: float = 1.0,
    ) -> None:
        canvas.circle(
            self.to_screen(cam, u, v),
            radius,
            self.theme.rgba(color, alpha),
            self.theme.rgba((0, 0, 0), alpha),
        )

    def column(self, canvas: Canvas, cam: int, u: float, color: RGB, alpha: float = 0.5) -> None:
        x0, y0, x1, y1 = self.box(cam)
        x = x0 + u * (x1 - x0)
        canvas.rect(
            (x - 3, y0, x + 3, y1),
            self.theme.rgba(color, alpha * 0.4),
            self.theme.rgba(color, alpha),
        )

    def feature_grid(
        self, canvas: Canvas, cam: int, rows: int, cols: int, alpha: float = 0.6
    ) -> None:
        """Overlay the coarse feature-map cells an encoder would produce."""
        x0, y0, x1, y1 = self.box(cam)
        col = self.theme.rgba(self.theme.learned, alpha)
        for r in range(rows + 1):
            y = y0 + r * (y1 - y0) / rows
            canvas.line([(x0, y), (x1, y)], col)
        for c in range(cols + 1):
            x = x0 + c * (x1 - x0) / cols
            canvas.line([(x, y0), (x, y1)], col)
