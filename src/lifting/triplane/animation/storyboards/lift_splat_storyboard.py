"""Lift-Splat-Shoot -> triplane, step by step."""

from __future__ import annotations

import numpy as np

from lifting.triplane.animation.depth_storyboard import DepthStoryboard
from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS


@STORYBOARDS.register("lift_splat")
class LiftSplatStoryboard(DepthStoryboard):
    """Lift every pixel to a weighted point cloud, splat (scatter-sum) it into voxels."""

    title = "Lift-Splat-Shoot  →  triplane"
    reference = "Philion & Fidler (ECCV 2020)  ·  lifting.triplane.LiftSplatTriplaneLifter"

    def steps(self) -> list[Step]:
        """Build the ordered animation steps explaining this lifter."""
        return [
            self.step_inputs(),
            self.step_depth_distribution(),
            self._step_lift(),
            self._step_splat(),
            self.step_collapse(
                self.facts.depth_consistent(),
                "Shoot: collapse Z, Y and X",
                "Instead of one BEV compressor, VoxelToTriplane folds each axis into channels in turn, "
                "producing xy, xz and yz planes from the same splatted volume.",
            ),
            self.step_output(),
        ]

    def _cloud(self) -> list[tuple[tuple[float, float, float], float, int]]:
        cloud = []
        for cam, u, v, cls, depth in self.pixel_samples(stride=6):
            for pt, p in self.frustum_points(cam, u, v, depth):
                if p > 0.02 and self.scene.inside(pt):
                    cloud.append((pt, p, cls))
        return cloud

    def _step_lift(self) -> Step:
        cloud = self._cloud()

        def over(ctx: FrameContext, t: float) -> None:
            for pt, p, cls in cloud[: int(round(t * len(cloud)))]:
                ctx.painter.dot(
                    pt, self.theme.classes[cls], 1.2 + 4 * np.sqrt(p), 0.3 + 0.7 * min(1.0, 3 * p)
                )

        return Step(
            "Lift: pixel -> weighted 3D points",
            "frustum_points unprojects every (pixel, depth-bin) pair; its feature is depth_prob x "
            "context. The result is a point cloud of D·H·W points per camera.",
            None,
            over,
            frames=10,
            objects_alpha=0.25,
        )

    def _step_splat(self) -> Step:
        cloud = self._cloud()
        targets = []
        for pt, _, _ in cloud:
            idx = tuple(
                int(
                    np.clip(
                        np.floor((pt[a] - self.scene.bounds[a][0]) / self.scene.cell[a]),
                        0,
                        self.scene.resolution[a] - 1,
                    )
                )
                for a in range(3)
            )
            inside = all(
                self.scene.bounds[a][0] <= pt[a] < self.scene.bounds[a][1] for a in range(3)
            )
            targets.append(idx if inside else None)
        voxels = self.facts.depth_consistent()
        cells = [idx for idx in np.ndindex(*voxels.shape) if voxels[idx] > 0]

        def over(ctx: FrameContext, t: float) -> None:
            for (pt, p, cls), idx in zip(cloud, targets, strict=True):
                if idx is None:
                    continue
                c = np.array(self.scene.voxel_center(*idx))
                pos = np.array(pt) + (c - np.array(pt)) * t
                ctx.painter.dot(
                    pos, self.theme.classes[cls], 1.5 + 3 * np.sqrt(p), 0.8 * (1 - 0.6 * t)
                )
            ctx.painter.voxels(
                (i, j, k, self.theme.classes[int(voxels[i, j, k])], 0.8 * t) for i, j, k in cells
            )

        return Step(
            "Splat: scatter-sum into voxels",
            "grid.flat_voxel_index + voxel_pooling(reduce='sum') drop every point into the voxel that "
            "contains it (a push). Points outside the grid are discarded.",
            None,
            over,
            frames=12,
            objects_alpha=0.25,
        )
