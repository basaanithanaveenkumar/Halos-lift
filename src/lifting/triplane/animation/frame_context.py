"""Everything a step's draw functions can use while drawing one frame."""

from __future__ import annotations

from dataclasses import dataclass

from lifting.triplane.animation.canvas import Canvas
from lifting.triplane.animation.image_panel import CameraImages
from lifting.triplane.animation.plane_panel import PlanePanel
from lifting.triplane.animation.scene_facts import SceneFacts
from lifting.triplane.animation.scene_painter import ScenePainter
from lifting.triplane.animation.theme import Theme


@dataclass
class FrameContext:
    """Per-frame drawing handles plus the storyboard's shared, pre-computed facts."""

    canvas: Canvas
    painter: ScenePainter
    images: CameraImages
    planes: PlanePanel
    facts: SceneFacts
    theme: Theme
