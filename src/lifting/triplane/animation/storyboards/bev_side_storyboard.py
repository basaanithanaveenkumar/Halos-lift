"""BEV + side (no front plane) from one reused perspective volume, step by step."""

from __future__ import annotations

from dataclasses import replace

from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS
from lifting.triplane.animation.storyboards.simple_bev_storyboard import SimpleBEVStoryboard


@STORYBOARDS.register("bev_side")
class BEVSideStoryboard(SimpleBEVStoryboard):
    """Sample the cameras once (Simple-BEV volume), pool it twice, mix BEV and side along x.

    Inherits the projection and sampling steps of :class:`SimpleBEVStoryboard` because the
    default ``volume`` of :class:`~lifting.triplane.BEVSideTriplaneLifter` is ``simple_bev``.
    """

    title = "BEV + side  (no front view)"
    reference = (
        "two-plane TPV, reused perspective volume  ·  lifting.triplane.BEVSideTriplaneLifter"
    )
    planes = (0, 1)

    def steps(self) -> list[Step]:
        """Build the ordered animation steps explaining this lifter."""
        return [
            self.step_inputs(
                "This variant keeps only xy (BEV) and xz (side): the yz front plane is never built."
            ),
            self.step_voxel_grid(
                "Every voxel centre (x, y, z) is a fixed 3D point. The perspective features will be "
                "read for these points exactly once."
            ),
            self._step_project(),
            replace(
                self._step_sample(),
                title="Sample the perspective features ONCE",
                caption=(
                    "volume = lift_voxels(features, cameras): one bilinear read per voxel and camera, "
                    "averaged over cameras. This (B, C, X, Y, Z) volume is the only image access - "
                    "both planes below reuse it."
                ),
            ),
            self.step_collapse(
                self.facts.smear,
                "Reuse the volume twice",
                "pool_height attends over Z -> xy (BEV); pool_lateral attends over Y -> xz (side). "
                "Same volume, no second camera pass - unlike BEVFormer / TPVFormer, which run one "
                "image cross-attention per plane.",
            ),
            self._step_mix(),
            self.step_output(
                "Output: two planes (X·Y + X·Z cells). A 3D point is decoded as xy(x, y) + xz(x, z): "
                "the side plane carries height, the BEV plane the lateral layout, so every voxel is "
                "still addressed on all three axes - with no Y·Z front plane to lift or store."
            ),
        ]

    def _step_mix(self) -> Step:
        labels = self.facts.plane_labels
        nx = self.scene.resolution[0]

        def under(ctx: FrameContext, t: float) -> None:
            for plane in self.planes:
                ctx.painter.plane_cells(plane, labels[plane])

        def over(ctx: FrameContext, t: float) -> None:
            for plane in self.planes:
                ctx.planes.fill(ctx.canvas, plane, labels[plane])
            row = min(int(t * nx), nx - 1)
            ny, nz = labels[0].shape[1], labels[1].shape[1]
            for b in range(ny):
                ctx.planes.highlight(ctx.canvas, 0, row, b)
            for b in range(nz):
                ctx.planes.highlight(ctx.canvas, 1, row, b)
            bev = ctx.planes.cell_box(0, row, 0)
            side = ctx.planes.cell_box(1, row, nz - 1)
            a = (bev[2] + 2, (bev[1] + bev[3]) / 2)
            c = ((side[0] + side[2]) / 2, side[1] - 2)
            ctx.canvas.arrow(a, c, self.theme.ray, 2)
            ctx.canvas.arrow(c, a, self.theme.ray, 2)
            (x0, x1), (y0, y1), (z0, z1) = self.scene.bounds
            xs = x0 + (row + 0.5) * (x1 - x0) / nx
            ctx.painter.dashed((xs, y0, z0), (xs, y1, z0), self.theme.ray, 0.9)
            ctx.painter.dashed((xs, y0, z0), (xs, y0, z1), self.theme.ray, 0.9)
            ctx.painter.label((xs, y1 + 0.4, z0), f"row x = {row}", self.theme.ray, 11)

        return Step(
            "BEV <-> side mixing along x",
            "The planes share the forward axis x. For every x row, the Y BEV cells attend to the Z "
            "side cells of that row and vice versa (BEVSideMixer): BEV learns the row's height "
            "profile, side learns its lateral layout.",
            under,
            over,
            frames=12,
            objects_alpha=0.35,
        )
