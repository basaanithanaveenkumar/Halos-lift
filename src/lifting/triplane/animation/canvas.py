"""Thin, typed wrapper around a Pillow image with the 2D primitives the animations need."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from lifting.triplane.animation.theme import Theme

Point = tuple[float, float]
Color = tuple[int, ...]


#: Glyphs the built-in font lacks, replaced by ASCII so captions never show tofu boxes.
_ASCII = str.maketrans({"→": "->", "←": "<-", "↑": "^", "↓": "v", "⊗": "x", "Σ": "sum", "×": "x"})


def ascii_safe(text: str) -> str:
    """Replace characters the bundled bitmap fonts cannot draw with ASCII look-alikes."""
    return text.translate(_ASCII)


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has a single bitmap font
        return ImageFont.load_default()


class Canvas:
    """One animation frame. All drawing is alpha-blended (``ImageDraw`` in ``RGBA`` mode).

    Args:
        size: ``(width, height)`` in pixels.
        theme: colours.
    """

    def __init__(self, size: tuple[int, int], theme: Theme) -> None:
        self.theme = theme
        self.image = Image.new("RGB", size, theme.background)
        self.draw = ImageDraw.Draw(self.image, "RGBA")

    @property
    def size(self) -> tuple[int, int]:
        """Canvas ``(width, height)`` in pixels."""
        return self.image.size

    # ------------------------------------------------------------------ text
    def text(
        self, xy: Point, text: str, size: int = 13, color: Color | None = None, anchor: str = "la"
    ) -> None:
        """Draw one line of text."""
        self.draw.text(
            xy, ascii_safe(text), fill=color or self.theme.text, font=_font(size), anchor=anchor
        )

    def multiline(
        self, xy: Point, text: str, size: int = 13, color: Color | None = None, spacing: int = 4
    ) -> None:
        """Draw word-wrapped text and return the height it used."""
        self.draw.multiline_text(
            xy, ascii_safe(text), fill=color or self.theme.text, font=_font(size), spacing=spacing
        )

    def text_width(self, text: str, size: int = 13) -> float:
        """Rendered width of ``text`` in pixels at ``size``."""
        return self.draw.textlength(ascii_safe(text), font=_font(size))

    # ------------------------------------------------------------------ shapes
    def line(self, points: Sequence[Point], color: Color, width: int = 1) -> None:
        """Draw a polyline through ``points``."""
        self.draw.line([tuple(p) for p in points], fill=color, width=width)

    def dashed(self, p0: Point, p1: Point, color: Color, width: int = 1, dash: float = 5) -> None:
        """Draw a dashed segment from ``p0`` to ``p1``."""
        length = math.dist(p0, p1)
        if length < 1e-6:
            return
        steps = max(1, int(length // dash))
        for s in range(0, steps, 2):
            a, b = s / steps, min(1.0, (s + 1) / steps)
            self.line(
                [
                    (p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a),
                    (p0[0] + (p1[0] - p0[0]) * b, p0[1] + (p1[1] - p0[1]) * b),
                ],
                color,
                width,
            )

    def polygon(
        self,
        points: Sequence[Point],
        fill: Color | None,
        outline: Color | None = None,
        width: int = 1,
    ) -> None:
        """Draw a filled and/or outlined polygon."""
        self.draw.polygon([tuple(p) for p in points], fill=fill, outline=outline, width=width)

    def circle(self, c: Point, r: float, fill: Color | None, outline: Color | None = None) -> None:
        """Draw a circle of radius ``r`` centred at ``c``."""
        self.draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill, outline=outline)

    def rect(
        self,
        box: Sequence[float],
        fill: Color | None,
        outline: Color | None = None,
        width: int = 1,
        radius: int = 0,
    ) -> None:
        """Draw a rectangle, optionally filled, outlined and rounded."""
        if radius:
            self.draw.rounded_rectangle(list(box), radius, fill=fill, outline=outline, width=width)
        else:
            self.draw.rectangle(list(box), fill=fill, outline=outline, width=width)

    def arrow(self, p0: Point, p1: Point, color: Color, width: int = 2, head: float = 8) -> None:
        """Draw an arrow from ``p0`` to ``p1`` with a triangular head."""
        self.line([p0, p1], color, width)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        left = (p1[0] - head * math.cos(ang - 0.45), p1[1] - head * math.sin(ang - 0.45))
        right = (p1[0] - head * math.cos(ang + 0.45), p1[1] - head * math.sin(ang + 0.45))
        self.polygon([p1, left, right], fill=color)

    def paste(self, image: Image.Image, xy: tuple[int, int]) -> None:
        """Paste ``image`` with its top-left corner at ``xy``."""
        self.image.paste(image, xy)

    def wrap(self, text: str, max_width: float, size: int = 13) -> str:
        """Greedy word wrap of ``text`` (paragraphs separated by ``\\n``) to ``max_width`` pixels."""
        lines = []
        for paragraph in text.split("\n"):
            current = ""
            for word in paragraph.split():
                trial = f"{current} {word}".strip()
                if current and self.text_width(trial, size) > max_width:
                    lines.append(current)
                    current = word
                else:
                    current = trial
            lines.append(current)
        return "\n".join(lines)
