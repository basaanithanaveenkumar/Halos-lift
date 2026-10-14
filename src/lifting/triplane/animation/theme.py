"""Colours shared by every frame of the lifting animations."""

from __future__ import annotations

from dataclasses import dataclass, field

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class Theme:
    """Dark palette; plane colours are fixed so ``xy`` / ``xz`` / ``yz`` read the same in every GIF.

    Class ids follow the toy scene: ``1`` vehicle, ``2`` pedestrian, ``3`` pole.
    """

    background: RGB = (16, 18, 24)
    panel: RGB = (27, 30, 39)
    panel_edge: RGB = (58, 63, 78)
    text: RGB = (236, 237, 242)
    muted: RGB = (150, 156, 172)
    grid: RGB = (58, 63, 78)
    box: RGB = (120, 128, 150)
    camera: RGB = (245, 245, 245)
    ray: RGB = (255, 210, 60)
    learned: RGB = (190, 125, 255)
    geometry: RGB = (90, 165, 255)
    sky: RGB = (54, 70, 98)
    ground: RGB = (44, 50, 46)
    plane_xy: RGB = (40, 200, 170)
    plane_xz: RGB = (255, 150, 60)
    plane_yz: RGB = (185, 125, 255)
    classes: dict[int, RGB] = field(
        default_factory=lambda: {1: (238, 74, 62), 2: (64, 146, 255), 3: (96, 205, 96)}
    )

    @property
    def planes(self) -> tuple[RGB, RGB, RGB]:
        return self.plane_xy, self.plane_xz, self.plane_yz

    @staticmethod
    def rgba(color: RGB, alpha: float) -> RGBA:
        """``color`` with opacity ``alpha`` in ``[0, 1]``."""
        return (*color, int(round(255 * max(0.0, min(1.0, alpha)))))

    @staticmethod
    def mix(a: RGB, b: RGB, t: float) -> RGB:
        """Linear blend ``(1 - t) * a + t * b``."""
        return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]

    @staticmethod
    def shade(color: RGB, factor: float) -> RGB:
        """Darken (``factor < 1``) or lighten (``> 1``) a colour."""
        return tuple(max(0, min(255, int(c * factor))) for c in color)  # type: ignore[return-value]
