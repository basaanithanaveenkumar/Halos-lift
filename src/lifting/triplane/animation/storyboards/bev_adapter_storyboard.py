"""Any BEV lifter -> triplane via BEVToTPV, step by step."""

from __future__ import annotations

import numpy as np

from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS, Storyboard


@STORYBOARDS.register("bev_adapter")
class BEVAdapterStoryboard(Storyboard):
    """BEV map from any lifter; side planes are learned expansions of pooled BEV columns."""

    title = "Any BEV lifter  →  triplane  (BEVToTPV)"
    reference = "lifting.triplane.BEVLifterTriplaneAdapter(base=<any LIFTERS name>)"

    def guessed_planes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """What a BEV map can support: xy exact; xz / yz = pooled column broadcast over z."""
        xy = self.facts.plane_labels[0]
        nz = self.scene.resolution[2]
        xz = np.repeat(xy.max(1)[:, None], nz, 1)
        yz = np.repeat(xy.max(0)[:, None], nz, 1)
        return xy, xz, yz

    def steps(self) -> list[Step]:
        """Build the ordered animation steps explaining this lifter."""
        xy, xz, yz = self.guessed_planes()
        return [
            self.step_inputs(),
            self._step_bev(xy),
            self._step_expand(
                xy,
                1,
                xz,
                "Pool along y -> expand to z (xz)",
                "bev.mean(dim=Y) gives one feature per x; a 1x1 Conv1d expands each to Z "
                "height cells - a LEARNED guess, height is not in a BEV map.",
            ),
            self._step_expand(
                xy,
                2,
                yz,
                "Pool along x -> expand to z (yz)",
                "bev.mean(dim=X) gives one feature per y, expanded to Z the same way "
                "(separate weights).",
            ),
            self.step_output(
                "Output: xy is the exact BEV of the wrapped lifter; xz / yz carry a learned height "
                "prior. Works for every LIFTERS entry, including your own plug-ins - but prefer the "
                "dedicated triplane lifters, which recover height from the images.",
                labels=(xy, xz, yz),
            ),
        ]

    def _step_bev(self, xy: np.ndarray) -> Step:
        def under(ctx: FrameContext, t: float) -> None:
            ctx.painter.plane_cells(0, xy, alpha=t, order=t)

        def over(ctx: FrameContext, t: float) -> None:
            ctx.planes.fill(ctx.canvas, 0, xy, order=t)

        return Step(
            "Any BEV lifter -> BEV map",
            "LIFTERS.build(base, ...) runs unchanged (simple_bev, lift_splat, bevformer, tiim, your "
            "own...) and returns (B, C, X, Y). BEVToTPV refines it with two convs into the xy plane.",
            under,
            over,
            frames=9,
        )

    def _step_expand(
        self, xy: np.ndarray, plane: int, target: np.ndarray, title: str, caption: str
    ) -> Step:
        occupied = [idx for idx in np.ndindex(*xy.shape) if xy[idx] > 0]

        def under(ctx: FrameContext, t: float) -> None:
            ctx.painter.plane_cells(0, xy)
            if plane == 2:
                ctx.painter.plane_cells(1, self.guessed_planes()[1])
            ctx.painter.plane_cells(plane, target, alpha=t)

        def over(ctx: FrameContext, t: float) -> None:
            for i, j in occupied:
                c = (self.scene.centers(0)[i], self.scene.centers(1)[j], 0.0)
                ctx.painter.ray(
                    c, self.scene.project_to_plane(c, plane), t, self.theme.planes[plane], 0.8, 2
                )
            ctx.planes.fill(ctx.canvas, 0, xy)
            if plane == 2:
                ctx.planes.fill(ctx.canvas, 1, self.guessed_planes()[1])
            ctx.planes.fill(ctx.canvas, plane, target, order=t)

        return Step(title, caption, under, over, frames=9, objects_alpha=0.5)
