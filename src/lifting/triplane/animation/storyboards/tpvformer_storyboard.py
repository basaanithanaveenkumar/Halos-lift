"""TPVFormer (native triplane), step by step."""

from __future__ import annotations

import numpy as np

from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.query_storyboard import QueryStoryboard
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS


@STORYBOARDS.register("tpvformer")
class TPVFormerStoryboard(QueryStoryboard):
    """Plane queries + per-plane image cross-attention + cross-view hybrid attention."""

    title = "TPVFormer  (native triplane)"
    reference = "Huang et al., TPVFormer (CVPR 2023)  ·  lifting.triplane.TPVFormerTriplaneLifter"

    def steps(self) -> list[Step]:
        return [
            self.step_inputs(),
            self.step_queries(
                "One token sequence of X·Y + X·Z + Y·Z learned queries (the three planes concatenated)."
            ),
            self.step_anchors(),
            self.step_project_anchors(),
            self._step_hybrid(),
            self.step_deformable(
                "Per-plane image cross-attention",
                "Each plane has its own SpatialCrossAttention (own offsets and anchor count): deformable "
                "sampling around every projected reference point, averaged over the cameras that see it.",
                self_attention=False,
            ),
            self.step_refine(
                "TPVFormerLayer = hybrid attention -> image cross-attention -> FFN, repeated num_layers "
                "times; tokens are split back into the three planes."
            ),
            self.step_output(),
        ]

    def _step_hybrid(self) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for plane in range(3):
                ctx.painter.plane_queries(plane, alpha=0.3)
            # the xy query's 3D location (missing z = centre) seen on all three planes
            a, b = self.QUERIES[0]
            x, y = self.scene.centers(0)[a], self.scene.centers(1)[b]
            p = (x, y, float(np.mean(self.scene.z_bounds)))
            q = self.query_center(0)
            ctx.painter.dot(q, self.theme.plane_xy, 6)
            for plane in (1, 2):
                target = self.scene.project_to_plane(p, plane)
                ctx.painter.ray(q, target, t, self.theme.planes[plane], 0.95, 3)
                ctx.painter.dot(target, self.theme.planes[plane], 5, alpha=t)
            ctx.painter.label(
                (x + 1.2, y + 1.2, 0.0), "same 3D spot on every plane", self.theme.ray
            )

        return Step(
            "Cross-view hybrid attention",
            "Planes talk: the three planes are treated as three levels of one pyramid, and every query "
            "samples deformable points at the SAME 3D location on xy, xz and yz (missing axis = centre).",
            None,
            over,
            frames=10,
            objects_alpha=0.5,
        )
