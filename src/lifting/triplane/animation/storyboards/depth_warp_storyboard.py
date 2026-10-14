"""Depth-Warp -> triplane, step by step."""

from __future__ import annotations

import numpy as np

from lifting.triplane.animation.depth_storyboard import DepthStoryboard
from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS


@STORYBOARDS.register("depth_warp")
class DepthWarpStoryboard(DepthStoryboard):
    """Depth x context frustum per camera; every voxel *pulls* from it at (u, v, depth)."""

    title = "Depth-Warp  →  triplane"
    reference = "Simple-BEV liftnet (backward warping)  ·  lifting.triplane.DepthWarpTriplaneLifter"
    PROBES = ((6, 5, 1), (5, 5, 1), (2, 1, 2), (4, 6, 1))

    def steps(self) -> list[Step]:
        voxels = self.facts.depth_consistent()
        return [
            self.step_inputs(),
            self.step_depth_distribution(),
            self._step_frustum(),
            self._step_pull(voxels),
            self.step_collapse(
                voxels,
                "Collapse Z, Y and X",
                "The pulled voxel volume (C', X, Y, Z) is folded along each axis by VoxelToTriplane. "
                "Because depth already localised the features, the planes are sharp from the start.",
            ),
            self.step_output(),
        ]

    def _step_frustum(self) -> Step:
        samples = self.pixel_samples(stride=6)

        def over(ctx: FrameContext, t: float) -> None:
            for cam, u, v, cls, depth in samples[: max(1, int(round(t * len(samples))))]:
                self.draw_depth_ray(ctx, cam, u, v, cls, depth)

        return Step(
            "Frustum volume = depth ⊗ context",
            "Outer product of depth probabilities (D) and context (C') gives every camera a frustum "
            "volume (C', D, H, W): features sit at the predicted depth along each pixel ray.",
            None,
            over,
            frames=10,
            objects_alpha=0.3,
        )

    def _step_pull(self, voxels: np.ndarray) -> Step:
        cells = [idx for idx in np.ndindex(*voxels.shape) if voxels[idx] > 0]

        def over(ctx: FrameContext, t: float) -> None:
            for n, idx in enumerate(self.PROBES):
                local = min(1.0, max(0.0, 2 * t - 0.3 * n))  # probes start staggered
                c = self.scene.voxel_center(*idx)
                ctx.painter.dot(c, self.theme.geometry, 4)
                for cam, u, v, _ in self.facts.voxel_views(idx):
                    ctx.painter.ray(
                        c, self.rig.cameras[cam].position, local, self.theme.ray, 0.8, 2
                    )
                    ctx.images.dot(ctx.canvas, cam, u, v, self.theme.ray, 3, alpha=local)
            shown = cells[: int(round(t * len(cells)))]
            ctx.painter.voxels(
                (i, j, k, self.theme.classes[int(voxels[i, j, k])], 0.8) for i, j, k in shown
            )

        return Step(
            "Pull: trilinear sample at (u, v, depth)",
            "Each voxel is projected into every camera and the frustum is sampled trilinearly at its "
            "(u, v, depth) - a gather, so no holes. Only voxels near a predicted surface receive features.",
            None,
            over,
            frames=12,
            objects_alpha=0.25,
        )
