"""Base class of every lifting animation: frame layout, timing, GIF export, shared steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from lifting.registry import Registry
from lifting.triplane.animation.canvas import Canvas
from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.image_panel import CameraImages
from lifting.triplane.animation.iso_projector import IsoProjector
from lifting.triplane.animation.plane_panel import PlanePanel
from lifting.triplane.animation.scene_facts import SceneFacts
from lifting.triplane.animation.scene_painter import ScenePainter
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.theme import Theme
from lifting.triplane.animation.toy_rig import ToyRig
from lifting.triplane.animation.toy_scene import ToyScene


class Storyboard(ABC):
    """A sequence of :class:`Step` rendered into an animated GIF.

    Frame layout (``880 x 500``)::

        +-------------------------------------------+------------------+
        | title / reference                         |  camera images   |
        |   isometric 3D scene (grid, planes,       |  (2 x 2)         |
        |   objects, cameras, step overlays)        +------------------+
        |                                           |  output planes   |
        | [Step k/n - title ...... caption]  o o o  |  xy | xz | yz    |
        +-------------------------------------------+------------------+

    Subclasses set :attr:`name`, :attr:`title`, :attr:`reference` and implement :meth:`steps`,
    usually composing the shared steps below with their own technique-specific ones.

    Args:
        theme: colours (default dark theme).
        frame_ms: duration of every animated frame.
    """

    name: ClassVar[str]
    title: ClassVar[str]
    reference: ClassVar[str] = ""

    SIZE = (880, 500)
    CAPTION_BOX = (14, 398, 540, 488)

    def __init__(self, theme: Theme | None = None, frame_ms: int = 85) -> None:
        self.theme = theme or Theme()
        self.frame_ms = frame_ms
        self.scene = ToyScene()
        self.rig = ToyRig()
        self.facts = SceneFacts(self.scene, self.rig)
        self.projector = IsoProjector(26.0, (272.0, 250.0))
        self.camera_images = CameraImages(self.facts.images, self.rig, self.theme, (560, 74))
        self.plane_panel = PlanePanel(self.scene, self.theme, (560, 270))

    @abstractmethod
    def steps(self) -> list[Step]:
        """The ordered steps of this technique."""

    # ------------------------------------------------------------------ rendering
    def render(self) -> tuple[list[Image.Image], list[int]]:
        """All frames and their durations in milliseconds."""
        steps = self.steps()
        frames, durations = [], []
        for idx, step in enumerate(steps):
            n = max(step.frames, 1)
            for f in range(n):
                t = f / (n - 1) if n > 1 else 1.0
                frames.append(self.render_frame(steps, idx, t))
                durations.append(step.hold_ms if f == n - 1 else self.frame_ms)
        return frames, durations

    def render_frame(self, steps: list[Step], idx: int, t: float) -> Image.Image:
        """Render frame ``t`` (in ``[0, 1]``) of step ``idx``."""
        step = steps[idx]
        canvas = Canvas(self.SIZE, self.theme)
        painter = ScenePainter(canvas, self.projector, self.scene, self.rig)
        ctx = FrameContext(
            canvas, painter, self.camera_images, self.plane_panel, self.facts, self.theme
        )

        self._header(canvas)
        painter.floor_grid()
        painter.back_walls()
        for plane in range(3):
            painter.plane_outline(plane, alpha=0.75)
        if step.under is not None:
            step.under(ctx, t)
        painter.bounding_box()
        painter.objects(step.objects_alpha)
        painter.cameras()
        self.camera_images.draw(canvas)
        self.plane_panel.draw(canvas)
        if step.over is not None:
            step.over(ctx, t)
        self._caption(canvas, steps, idx)
        return canvas.image

    def save(self, path: str | Path) -> Path:
        """Write the animated GIF (palette-quantised per frame, looping forever)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames, durations = self.render()
        paletted = [
            f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            for f in frames
        ]
        paletted[0].save(
            path,
            save_all=True,
            append_images=paletted[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=1,
        )
        return path

    def _header(self, canvas: Canvas) -> None:
        canvas.text((18, 12), self.title, 20, self.theme.text)
        if self.reference:
            canvas.text((18, 38), self.reference, 12, self.theme.muted)

    def _caption(self, canvas: Canvas, steps: list[Step], idx: int) -> None:
        x0, y0, x1, y1 = self.CAPTION_BOX
        canvas.rect(
            (x0, y0, x1, y1),
            self.theme.rgba(self.theme.panel, 0.94),
            self.theme.panel_edge,
            radius=8,
        )
        step = steps[idx]
        canvas.text(
            (x0 + 12, y0 + 9), f"Step {idx + 1}/{len(steps)}  ·  {step.title}", 15, self.theme.ray
        )
        body = canvas.wrap(step.caption, x1 - x0 - 24, 12)
        canvas.multiline((x0 + 12, y0 + 32), body, 12, self.theme.text, spacing=3)
        for n in range(len(steps)):
            cx = x1 - 14 - (len(steps) - 1 - n) * 13
            fill = (
                self.theme.ray
                if n == idx
                else (self.theme.muted if n < idx else self.theme.panel_edge)
            )
            canvas.circle((cx, y0 + 16), 4, fill)

    # ------------------------------------------------------------------ shared steps
    def step_inputs(self, extra: str = "") -> Step:
        """Cameras sweep their fields of view; the camera images light up one after another."""

        def under(ctx: FrameContext, t: float) -> None:
            for cam in range(len(self.rig)):
                ctx.painter.fov_fan(cam, alpha=0.06 + 0.08 * t)

        def over(ctx: FrameContext, t: float) -> None:
            cam = min(int(t * len(self.rig)), len(self.rig) - 1)
            ctx.painter.cameras(highlight=(cam,))
            ctx.images.highlight(ctx.canvas, cam)

        caption = (
            "Four calibrated cameras (intrinsics K, extrinsics cam→ego) see the scene. "
            "Target: a metric 3D grid X×Y×Z, stored compactly as three planes - "
            "xy (the BEV), xz (side) and yz (front)."
        )
        return Step("Inputs", f"{caption} {extra}".strip(), under, over, frames=8)

    def step_voxel_grid(self, caption: str) -> Step:
        """Voxel centres appear layer by layer."""

        def over(ctx: FrameContext, t: float) -> None:
            nx, ny, nz = self.scene.resolution
            shown = int(round(t * nx * ny * nz))
            for n, idx in enumerate(np.ndindex(nx, ny, nz)):
                if n >= shown:
                    break
                ctx.painter.dot(self.scene.voxel_center(*idx), self.theme.geometry, 1.8, 0.8)

        return Step("Voxel centres", caption, None, over, frames=8, objects_alpha=0.45)

    def step_collapse(self, voxels: np.ndarray, title: str, caption: str) -> Step:
        """Voxel features fold onto the three planes (learned compression of Z, Y and X)."""
        labels = self.facts.plane_labels
        occupied = [idx for idx in np.ndindex(*voxels.shape) if voxels[idx] > 0]

        def under(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_cells(plane, labels[plane], alpha=t)

        def over(ctx: FrameContext, t: float) -> None:
            ctx.painter.voxels(
                (i, j, k, self.theme.classes[int(voxels[i, j, k])], 0.8 * (1 - 0.75 * t))
                for i, j, k in occupied
            )
            for idx in occupied[::2]:
                c = self.scene.voxel_center(*idx)
                for plane in range(3):
                    ctx.painter.ray(
                        c, self.scene.project_to_plane(c, plane), t, self.theme.planes[plane], 0.55
                    )
            for plane in range(3):
                ctx.planes.fill(ctx.canvas, plane, labels[plane], order=t)

        return Step(title, caption, under, over, frames=10, objects_alpha=0.25)

    def step_output(
        self, caption: str | None = None, labels: tuple[np.ndarray, ...] | None = None
    ) -> Step:
        """Final triplane; one 3D point decoded as the sum of its three plane features.

        ``labels`` overrides the (default: exact) plane contents, e.g. for the BEV adapter whose
        side planes are only a learned guess.
        """
        labels = labels if labels is not None else self.facts.plane_labels
        obj = self.scene.objects[0]
        p = (obj.center[0], obj.center[1], obj.size[2] * 0.6)

        def under(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_cells(plane, labels[plane])

        def over(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.planes.fill(ctx.canvas, plane, labels[plane])
                shadow = self.scene.project_to_plane(p, plane)
                ctx.painter.dashed(p, shadow, self.theme.planes[plane], min(1.0, 2 * t))
                ctx.painter.dot(shadow, self.theme.planes[plane], 4)
            ctx.painter.dot(p, self.theme.ray, 5)
            ctx.painter.label((p[0] + 0.2, p[1] + 1.6, p[2] + 1.3), "query point P", self.theme.ray)
            ctx.canvas.rect(
                (22, 66, 252, 112),
                self.theme.rgba(self.theme.panel, 0.9),
                self.theme.panel_edge,
                radius=6,
            )
            ctx.canvas.text((32, 72), "f(P) = xy(x,y) + xz(x,z) + yz(y,z)", 13, self.theme.text)
            ctx.canvas.text((32, 92), "BEV = HeightCompressor(planes)", 12, self.theme.plane_xy)

        text = caption or (
            "Output: three planes (X·Y + X·Z + Y·Z cells instead of X·Y·Z voxels). Any 3D "
            "point is decoded by summing its three plane features (TPVAggregator); the "
            "BEV map for BEV heads is read off the same planes."
        )
        return Step("Triplane output", text, under, over, frames=10, hold_ms=2600)


#: Registry of animations, keyed like :data:`lifting.triplane.TRIPLANE_LIFTERS`.
STORYBOARDS: Registry[Storyboard] = Registry("storyboard")
