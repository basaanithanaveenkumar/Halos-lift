"""One captioned step of a lifting animation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime cycle: FrameContext imports nothing from here
    from lifting.triplane.animation.frame_context import FrameContext

DrawFn = Callable[["FrameContext", float], None]


@dataclass
class Step:
    """A step is animated over ``frames`` frames with progress ``t`` going ``0 -> 1``.

    Attributes:
        title: short name shown as "Step k/n - title".
        caption: one or two sentences explaining what happens (word-wrapped).
        under: drawn after the floor / walls but *before* the objects (plane fills).
        over: drawn after objects and cameras (rays, voxels, panel overlays).
        frames: number of animated frames.
        hold_ms: how long the final frame of the step stays on screen.
        objects_alpha: opacity of the scene objects during this step.
    """

    title: str
    caption: str
    under: DrawFn | None = None
    over: DrawFn | None = None
    frames: int = 9
    hold_ms: int = 1700
    objects_alpha: float = 1.0
