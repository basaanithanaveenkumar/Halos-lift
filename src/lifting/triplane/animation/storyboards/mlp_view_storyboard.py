"""Geometry-free MLP view transform -> triplane (negative control), step by step."""

from __future__ import annotations

import math

from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS, Storyboard


@STORYBOARDS.register("mlp_view")
class MLPViewStoryboard(Storyboard):
    """Fixed per-camera fully connected maps from image cells to plane cells; no calibration."""

    title = "MLP view (negative control)  →  triplane"
    reference = "VPN-style view parsing  ·  lifting.triplane.MLPViewTriplaneLifter"
    ROWS, COLS = 3, 6

    def steps(self) -> list[Step]:
        return [
            self.step_inputs(),
            self._step_cells(),
            self._step_mlps(),
            self._step_sum(),
            self._step_rotate(),
        ]

    def _step_cells(self) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for cam in range(len(self.rig)):
                ctx.images.feature_grid(ctx.canvas, cam, self.ROWS, self.COLS, alpha=t)

        return Step(
            "Flatten coarse feature cells",
            "Take the coarsest feature map (h x w cells) of every camera and flatten it to h·w numbers "
            "per channel. Positions in the image are just indices - no rays, no depth.",
            None,
            over,
            frames=7,
        )

    def _step_mlps(self) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for cam in range(len(self.rig)):
                ctx.images.feature_grid(ctx.canvas, cam, self.ROWS, self.COLS, alpha=0.6)
                x0, y0, x1, y1 = ctx.images.box(cam)
                for r in range(self.ROWS):
                    for c in range(self.COLS):
                        src = (
                            x0 + (c + 0.5) * (x1 - x0) / self.COLS,
                            y0 + (r + 0.5) * (y1 - y0) / self.ROWS,
                        )
                        for plane in range(3):
                            a = (r * 5 + c * 3 + cam * 7) % self.scene.resolution[0]
                            b = (r * 3 + c + cam) % (
                                self.scene.resolution[1] if plane == 0 else self.scene.resolution[2]
                            )
                            bx0, by0, bx1, by1 = ctx.planes.cell_box(plane, a, b)
                            dst = ((bx0 + bx1) / 2, (by0 + by1) / 2)
                            end = (src[0] + (dst[0] - src[0]) * t, src[1] + (dst[1] - src[1]) * t)
                            ctx.canvas.line(
                                [src, end], self.theme.rgba(self.theme.planes[plane], 0.12)
                            )

        return Step(
            "Per-camera, per-plane MLP",
            "One Linear(h·w -> H_p·W_p) for every (camera, plane): each image cell is wired to every "
            "plane cell with a learned weight. The mapping is memorised, not derived from calibration.",
            None,
            over,
            frames=9,
        )

    def _step_sum(self) -> Step:
        labels = self.facts.plane_labels

        def under(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_cells(plane, labels[plane], alpha=t)

        def over(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.planes.fill(ctx.canvas, plane, labels[plane], order=t)

        return Step(
            "Sum over cameras -> planes",
            "Plane = Σ_cameras MLP(features), then a conv head per plane. With a FIXED rig the MLPs "
            "can memorise the layout and the planes look right...",
            under,
            over,
            frames=9,
        )

    def _step_rotate(self) -> Step:
        labels = self.facts.plane_labels

        def under(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_cells(plane, labels[plane])

        def over(ctx: FrameContext, t: float) -> None:
            ang = math.radians(90 * t)
            for cam in self.rig.cameras:
                fx, fy = (
                    math.cos(math.radians(cam.yaw) + ang),
                    math.sin(math.radians(cam.yaw) + ang),
                )
                ctx.painter.ray((0, 0, 0.9), (2.2 * fx, 2.2 * fy, 0.9), 1.0, self.theme.ray, 1.0, 3)
            for plane in range(3):
                ctx.planes.fill(ctx.canvas, plane, labels[plane])
            ctx.canvas.rect(
                (22, 66, 300, 112),
                self.theme.rgba(self.theme.panel, 0.92),
                self.theme.classes[1],
                radius=6,
            )
            ctx.canvas.text((32, 72), f"rig rotated by {90 * t:.0f} deg", 13, self.theme.ray)
            ctx.canvas.text((32, 92), "planes unchanged -> now WRONG", 13, self.theme.classes[1])

        return Step(
            "Rotate the rig: it cannot follow",
            "Rotate the extrinsics by 90°: every geometric lifter's planes rotate with them, the MLP "
            "planes do not (uses_geometry = False). That is why it is the negative control.",
            under,
            over,
            frames=10,
            hold_ms=2600,
        )
