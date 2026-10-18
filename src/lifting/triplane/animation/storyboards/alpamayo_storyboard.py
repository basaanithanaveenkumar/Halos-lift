"""Alpamayo (Mip-NeRF 360 triplane) -> tokens, step by step."""

from __future__ import annotations

import numpy as np

from lifting.lifters.alpamayo.mipnerf360 import MipNeRF360Parametrizer
from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.query_storyboard import QueryStoryboard
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS


@STORYBOARDS.register("alpamayo")
class AlpamayoStoryboard(QueryStoryboard):
    """TPVFormer layers on non-uniform (dense near, coarse far) planes, then patch tokens."""

    title = "Alpamayo triplane  (Mip-NeRF 360)"
    reference = "Ivanovic et al. arXiv:2506.12251  ·  lifting.triplane.AlpamayoTriplaneAdapter"

    def steps(self) -> list[Step]:
        """Build the ordered animation steps explaining this lifter."""
        return [
            self.step_inputs(),
            self._step_layout(),
            self.step_queries(
                "Same learned plane queries as TPVFormer - but their 3D positions follow the "
                "non-uniform layout, so anchors are dense around the ego vehicle."
            ),
            self.step_project_anchors(),
            self.step_deformable(
                "Hybrid + image cross-attention",
                "TPVFormerLayer, unchanged: cross-view hybrid attention between planes, then per-plane "
                "deformable image cross-attention around the projected reference points.",
            ),
            self.step_refine("num_layers TPVFormer layers produce the three planes."),
            self._step_patchify(),
            self.step_output(),
        ]

    def _centers(self, axis: int) -> np.ndarray:
        lo, hi = self.scene.bounds[axis]
        return (
            MipNeRF360Parametrizer(
                self.scene.resolution[axis], 0.75, hi - lo, center_m=(lo + hi) / 2
            )
            .cell_centers_m()
            .numpy()
        )

    def _step_layout(self) -> Step:
        xs, ys = self._centers(0), self._centers(1)

        def over(ctx: FrameContext, t: float) -> None:
            for x in xs:
                ctx.painter.ray((x, ys[0], 0.02), (x, ys[-1], 0.02), t, self.theme.plane_xy, 0.9, 1)
            for y in ys:
                ctx.painter.ray((xs[0], y, 0.02), (xs[-1], y, 0.02), t, self.theme.plane_xy, 0.9, 1)
            for x in xs:
                for y in ys:
                    ctx.painter.dot((x, y, 0.02), self.theme.plane_xy, 2.4, alpha=t)

        return Step(
            "Non-uniform cell layout",
            "MipNeRF360Parametrizer: cells uniform in contracted space - narrow, evenly spaced cells "
            "in the inner zone, widening smoothly over the far tails.",
            None,
            over,
            frames=9,
            objects_alpha=0.5,
        )

    def _step_patchify(self) -> Step:
        labels = self.facts.plane_labels
        p = 2
        sizes = [(lbl.shape[0] // p) * (lbl.shape[1] // p) for lbl in labels]

        def under(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_cells(plane, labels[plane])

        def over(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.planes.fill(ctx.canvas, plane, labels[plane])
            x, y = 24, 70
            ctx.canvas.rect(
                (x - 8, y - 6, x + 250, y + 52),
                self.theme.rgba(self.theme.panel, 0.9),
                self.theme.panel_edge,
                radius=6,
            )
            total = sum(sizes)
            shown = int(round(t * total))
            n = 0
            for plane, count in enumerate(sizes):
                for _ in range(count):
                    if n >= shown:
                        break
                    cx, cy = x + (n % 24) * 10, y + (n // 24) * 9
                    ctx.canvas.rect(
                        (cx, cy, cx + 8, cy + 7), self.theme.rgba(self.theme.planes[plane], 0.9)
                    )
                    n += 1
            ctx.canvas.text(
                (x, y + 36),
                f"{total} plane tokens  (patch {p}x{p}) -> VLA policy",
                11,
                self.theme.text,
            )

        return Step(
            "Patchify planes -> tokens",
            "patchify_planes cuts each plane into p x p patches: a fixed-length token sequence "
            "(X·Y + X·Z + Y·Z)/p² that feeds the Alpamayo VLA policy - far fewer than per-camera patches.",
            under,
            over,
            frames=10,
        )
