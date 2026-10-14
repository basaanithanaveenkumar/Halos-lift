"""Shared steps of the query-based lifters (BEVFormer, TPVFormer, Alpamayo)."""

from __future__ import annotations

import math

import numpy as np

from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import Storyboard


class QueryStoryboard(Storyboard):
    """Learned plane queries -> anchors along plane normals -> projection -> deformable attention.

    One highlighted query per plane, chosen on an object so its anchors hit something:
    ``xy`` on the vehicle (pillar along z), ``xz`` on the vehicle (line along y) and ``yz`` on the
    pedestrian (line along x).
    """

    #: plane -> (a, b) query cell
    QUERIES = {0: (6, 5), 1: (6, 1), 2: (1, 2)}
    ANCHORS = (4, 8, 8)

    def anchors(self, plane: int) -> list[tuple[float, float, float]]:
        """Metric anchors of the highlighted query, spread along the plane normal."""
        a, b = self.QUERIES[plane]
        sc = self.scene
        n = self.ANCHORS[plane]
        if plane == 0:
            x, y = sc.centers(0)[a], sc.centers(1)[b]
            return [(x, y, z) for z in np.linspace(0.25, 1.75, n)]
        if plane == 1:
            x, z = sc.centers(0)[a], sc.centers(2)[b]
            return [(x, y, z) for y in np.linspace(-3.5, 3.5, n)]
        y, z = sc.centers(1)[a], sc.centers(2)[b]
        return [(x, y, z) for x in np.linspace(-3.5, 3.5, n)]

    def query_center(self, plane: int) -> tuple[float, float, float]:
        a, b = self.QUERIES[plane]
        return tuple(np.array(self.scene.plane_cell(plane, a, b)).mean(0))  # type: ignore[return-value]

    def step_queries(self, caption: str) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_queries(plane, alpha=min(1.0, 1.6 * t))
                ctx.planes.queries(ctx.canvas, plane, alpha=min(1.0, 1.6 * t))

        return Step("Learned plane queries", caption, None, over, frames=7, objects_alpha=0.6)

    def step_anchors(self) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_queries(plane, alpha=0.35)
                color = self.theme.planes[plane]
                pts = self.anchors(plane)
                q = self.query_center(plane)
                ctx.painter.dot(q, color, 5)
                shown = pts[: max(1, int(round(t * len(pts))))]
                ctx.painter.ray(q, shown[-1], 1.0, color, 0.7, 2)
                for p in shown:
                    ctx.painter.dot(p, color, 3.5)
                a, b = self.QUERIES[plane]
                ctx.planes.highlight(ctx.canvas, plane, a, b)

        return Step(
            "3D anchors along each plane normal",
            "Each query stands for a whole line of space: xy cells own vertical pillars (4 anchors "
            "along z), xz cells lines along y, yz cells lines along x (8 anchors) - TPVReferencePoints.",
            None,
            over,
            frames=9,
            objects_alpha=0.5,
        )

    def step_project_anchors(self) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                color = self.theme.planes[plane]
                for p in self.anchors(plane):
                    ctx.painter.dot(p, color, 3)
                    for cam, u, v, _ in self.rig.project(p):
                        ctx.painter.ray(p, self.rig.cameras[cam].position, t, color, 0.35)
                        if t > 0.6:
                            ctx.images.dot(ctx.canvas, cam, u, v, color, 3)

        return Step(
            "Project anchors -> reference points",
            "cameras.project(anchors) gives each query a few pixels in each camera that can see it "
            "(the rest are masked). Pure geometry: this is where calibration enters.",
            None,
            over,
            frames=10,
            objects_alpha=0.5,
        )

    def step_deformable(self, title: str, caption: str, self_attention: bool = True) -> Step:
        refs = [
            (plane, cam, u, v)
            for plane in range(3)
            for p in self.anchors(plane)[::2]
            for cam, u, v, _ in self.rig.project(p)
        ]

        def over(ctx: FrameContext, t: float) -> None:
            spin = 2 * math.pi * t
            for plane, cam, u, v in refs:
                color = self.theme.planes[plane]
                c = ctx.images.to_screen(cam, u, v)
                for k in range(4):
                    ang = spin + k * math.pi / 2 + plane
                    s = (c[0] + 9 * math.cos(ang), c[1] + 6 * math.sin(ang))
                    ctx.canvas.line([c, s], self.theme.rgba(color, 0.7))
                    ctx.canvas.circle(s, 1.8, self.theme.rgba(color, 0.95))
                ctx.canvas.circle(c, 2.5, self.theme.rgba(color, 1.0))
            if self_attention:
                for plane in range(3):
                    q = self.query_center(plane)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = self.QUERIES[plane]
                        h, w = (
                            [self.scene.resolution[i] for i in (0, 1)],
                            [self.scene.resolution[i] for i in (0, 2)],
                            [self.scene.resolution[i] for i in (1, 2)],
                        )[plane]
                        na, nb = min(max(a + dx, 0), h - 1), min(max(b + dy, 0), w - 1)
                        nbr = tuple(np.array(self.scene.plane_cell(plane, na, nb)).mean(0))
                        ctx.painter.ray(q, nbr, t, self.theme.planes[plane], 0.9, 2)
            for plane in range(3):
                ctx.painter.dot(self.query_center(plane), self.theme.planes[plane], 5)

        return Step(title, caption, None, over, frames=12, objects_alpha=0.5)

    def step_refine(self, caption: str) -> Step:
        labels = self.facts.plane_labels

        def under(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_cells(plane, labels[plane], alpha=t)

        def over(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_queries(plane, alpha=1 - t)
                ctx.planes.fill(ctx.canvas, plane, labels[plane], alpha=t)
            layer = 1 + min(int(t * 2), 1)
            ctx.canvas.text((24, 70), f"encoder layer {layer} / 2", 14, self.theme.learned)

        return Step("Stack layers -> planes", caption, under, over, frames=10, objects_alpha=0.6)
