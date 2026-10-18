"""Shared steps of the depth-based lifters (Depth-Warp, Lift-Splat)."""

from __future__ import annotations

import math

import numpy as np

from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import Storyboard

#: depth-bin centres (metres) drawn along each pixel ray
DEPTH_BINS = np.arange(0.75, 6.0, 0.5)


class DepthStoryboard(Storyboard):
    """Adds the per-pixel categorical depth step and frustum-point helpers.

    The depth distribution drawn is the one a well-trained ``DepthHead`` predicts: a peak at
    the true surface depth (found by ray casting), softmax-shaped with width ``SIGMA``.
    """

    SIGMA = 0.45

    def pixel_samples(self, stride: int = 5) -> list[tuple[int, float, float, int, float]]:
        """``(cam, u, v, cls, depth)`` object pixels of every camera (subsampled)."""
        out = []
        for cam in range(len(self.rig)):
            out += [(cam, *px) for px in self.facts.object_pixels(cam, stride)]
        return out

    def depth_probs(self, hit_depth: float) -> np.ndarray:
        """Depth distribution a depth head would predict for a surface at ``hit_depth``."""
        logits = -0.5 * ((DEPTH_BINS - hit_depth) / self.SIGMA) ** 2
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def frustum_points(self, cam: int, u: float, v: float, hit_depth: float):
        """``(point, probability)`` along the ray of pixel ``(u, v)``."""
        for d, p in zip(DEPTH_BINS, self.depth_probs(hit_depth), strict=True):
            yield self.rig.point_at(cam, u, v, float(d)), float(p)

    def draw_depth_ray(
        self,
        ctx: FrameContext,
        cam: int,
        u: float,
        v: float,
        cls: int,
        hit_depth: float,
        t: float = 1.0,
    ) -> None:
        """Ray of one pixel with a dot per depth bin, radius / opacity ~ probability."""
        far = self.rig.point_at(cam, u, v, float(DEPTH_BINS[-1]))
        ctx.painter.ray(self.rig.cameras[cam].position, far, t, self.theme.muted, 0.5)
        color = self.theme.classes[cls]
        for n, (pt, p) in enumerate(self.frustum_points(cam, u, v, hit_depth)):
            if n / len(DEPTH_BINS) > t:
                break
            if not self.scene.inside(pt):  # outside the grid: discarded by the lifter too
                continue
            ctx.painter.dot(pt, color, 1.2 + 6 * math.sqrt(p), 0.25 + 0.75 * min(1.0, 3 * p))

    def step_depth_distribution(self) -> Step:
        """Step showing the per-pixel depth distribution along each probed ray."""
        probes = [s for s in self.pixel_samples(stride=9)][:8]

        def over(ctx: FrameContext, t: float) -> None:
            for cam, u, v, cls, depth in probes:
                self.draw_depth_ray(ctx, cam, u, v, cls, depth, t)
                ctx.images.dot(ctx.canvas, cam, u, v, self.theme.ray, 3)
            self._depth_histogram(ctx, probes[0][4], t)

        return Step(
            "Per-pixel depth distribution",
            "DepthHead predicts, for every feature pixel, a softmax over D depth bins plus a context "
            "vector. One pixel = one ray; the distribution says where along the ray the surface is.",
            None,
            over,
            frames=10,
            objects_alpha=0.35,
        )

    def _depth_histogram(self, ctx: FrameContext, hit_depth: float, t: float) -> None:
        """Small bar chart of one pixel's depth softmax, drawn in the scene's top-left corner."""
        x0, y0, w, h = 24, 70, 190, 60
        ctx.canvas.rect(
            (x0 - 8, y0 - 6, x0 + w + 8, y0 + h + 22),
            self.theme.rgba(self.theme.panel, 0.9),
            self.theme.panel_edge,
            radius=6,
        )
        probs = self.depth_probs(hit_depth)
        bw = w / len(probs)
        for n, p in enumerate(probs):
            bh = h * p / probs.max() * min(1.0, 1.5 * t)
            ctx.canvas.rect(
                (x0 + n * bw + 1, y0 + h - bh, x0 + (n + 1) * bw - 1, y0 + h),
                self.theme.rgba(self.theme.learned, 0.9),
            )
        ctx.canvas.text((x0, y0 + h + 4), "p(depth | pixel)   near -> far", 11, self.theme.muted)
