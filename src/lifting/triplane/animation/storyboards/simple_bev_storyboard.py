"""Simple-BEV -> triplane, step by step."""

from __future__ import annotations

import numpy as np

from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS, Storyboard


@STORYBOARDS.register("simple_bev")
class SimpleBEVStoryboard(Storyboard):
    """Project voxel centres, read pixels, average cameras, collapse three axes."""

    title = "Simple-BEV  →  triplane"
    reference = "Harley et al., Simple-BEV (ICRA 2023)  ·  lifting.triplane.SimpleBEVTriplaneLifter"

    #: a few voxels whose projection rays are drawn (vehicle side, pedestrian, empty ground)
    PROBES = ((6, 5, 1), (2, 1, 2), (5, 2, 0), (1, 6, 3))

    def steps(self) -> list[Step]:
        return [
            self.step_inputs(),
            self.step_voxel_grid(
                "Every voxel centre (x, y, z) of the grid is a fixed 3D point - no learning, no depth "
                "prediction. Simple-BEV pulls a feature for each of them."
            ),
            self._step_project(),
            self._step_sample(),
            self.step_collapse(
                self.facts.smear,
                "Collapse Z, Y and X",
                "VoxelToTriplane folds one axis into channels at a time (Z→xy, Y→xz, X→yz) and "
                "convolves: the learned compressors resolve the smear because each object is seen "
                "from several directions.",
            ),
            self.step_output(),
        ]

    def _step_project(self) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for n, idx in enumerate(self.PROBES):
                local = min(1.0, max(0.0, t * len(self.PROBES) - n))
                if local <= 0:
                    continue
                c = self.scene.voxel_center(*idx)
                ctx.painter.dot(c, self.theme.geometry, 4)
                for cam, u, v, _ in self.facts.voxel_views(idx):
                    ctx.painter.ray(
                        c, self.rig.cameras[cam].position, local, self.theme.ray, 0.9, 2
                    )
                    if local >= 1.0:
                        ctx.images.dot(ctx.canvas, cam, u, v, self.theme.ray, 4)
                        ctx.images.highlight(ctx.canvas, cam)

        return Step(
            "Project into every camera",
            "cameras.project(voxel_centres): ego → camera frame → pinhole → pixel (u, v). Each voxel "
            "lands on one pixel per camera that sees it; the rest are masked as invalid.",
            None,
            over,
            frames=12,
            objects_alpha=0.5,
        )

    def _step_sample(self) -> Step:
        smear = self.facts.smear
        cells = [idx for idx in np.ndindex(*smear.shape) if smear[idx] > 0]

        def over(ctx: FrameContext, t: float) -> None:
            shown = cells[: int(round(t * len(cells)))]
            ctx.painter.voxels(
                (i, j, k, self.theme.classes[int(smear[i, j, k])], 0.75) for i, j, k in shown
            )

        return Step(
            "Bilinear sample + camera average",
            "sample_multi_camera reads the feature under each pixel; masked_camera_mean averages the "
            "cameras. With no depth, every voxel on a ray gets the same feature - the colour smears "
            "along the rays behind each object.",
            None,
            over,
            frames=10,
            objects_alpha=0.3,
        )
