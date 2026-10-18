"""TIIM (polar rays) -> triplane, step by step."""

from __future__ import annotations

import numpy as np

from lifting.triplane.animation.frame_context import FrameContext
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS, Storyboard


@STORYBOARDS.register("tiim")
class TIIMStoryboard(Storyboard):
    """Image column -> polar ray (range), plus image row (elevation) per voxel."""

    title = "TIIM polar rays  →  triplane"
    reference = "Saha et al., Translating Images into Maps (ICRA 2022)  ·  lifting.triplane.PolarRayTriplaneLifter"
    CAMERA = 0
    COLUMNS = np.linspace(0.08, 0.92, 12)
    RANGE = np.arange(0.75, 6.0, 0.5)

    def steps(self) -> list[Step]:
        """Build the ordered animation steps explaining this lifter."""
        voxels = self.facts.depth_consistent()
        return [
            self.step_inputs(),
            self._step_columns(),
            self._step_polar(),
            self._step_voxels(voxels),
            self.step_collapse(
                voxels,
                "Collapse Z, Y and X",
                "The (2C, X, Y, Z) volume - range half + elevation half - is folded along each axis "
                "into the xy, xz and yz planes.",
            ),
            self.step_output(),
        ]

    def _column_hits(self, u: float) -> list[tuple[float, int]]:
        """(depth, class) of every object surface seen anywhere in image column ``u``."""
        hits = {}
        for v in np.linspace(0.3, 0.95, 14):
            cls, d = self.rig.cast(self.scene, self.CAMERA, float(u), float(v))
            if cls > 0:
                hits[round(d * 2) / 2] = cls
        return sorted(hits.items())

    def _step_columns(self) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for n, u in enumerate(self.COLUMNS):
                if n / len(self.COLUMNS) <= t:
                    ctx.images.column(ctx.canvas, self.CAMERA, float(u), self.theme.learned, 0.7)
            ctx.images.highlight(ctx.canvas, self.CAMERA)
            ctx.painter.cameras(highlight=(self.CAMERA,))

        return Step(
            "Column transformer",
            "Every image column (H pixels) is a sequence: a transformer encoder self-attends along it, "
            "with a row embedding so it knows which pixel is high and which is low.",
            None,
            over,
            frames=9,
            objects_alpha=0.6,
        )

    def _step_polar(self) -> Step:
        def over(ctx: FrameContext, t: float) -> None:
            for n, u in enumerate(self.COLUMNS):
                if n / len(self.COLUMNS) > t:
                    break
                hits = self._column_hits(float(u))
                for d in self.RANGE:
                    p = self.rig.point_at(self.CAMERA, float(u), 0.5, float(d))
                    if not self.scene.inside((p[0], p[1], 0.05)):
                        continue
                    hit = [cls for hd, cls in hits if abs(hd - d) < 0.3]
                    color = self.theme.classes[hit[0]] if hit else self.theme.learned
                    ctx.painter.dot(
                        (p[0], p[1], 0.05), color, 3.5 if hit else 1.6, 1.0 if hit else 0.5
                    )
                ctx.images.column(ctx.canvas, self.CAMERA, float(u), self.theme.learned, 0.5)
            ctx.painter.fov_fan(self.CAMERA, depth=3.4, alpha=0.08)

        return Step(
            "Depth queries -> polar ray",
            "D learned depth queries cross-attend to the encoded column: the column becomes a polar "
            "ray of D range cells, (C, D, W) per camera. The row axis is consumed here.",
            None,
            over,
            frames=12,
            objects_alpha=0.45,
        )

    def _step_voxels(self, voxels: np.ndarray) -> Step:
        probes = ((6, 5, 1), (6, 4, 0), (5, 5, 1))
        cells = [idx for idx in np.ndindex(*voxels.shape) if voxels[idx] > 0]

        def over(ctx: FrameContext, t: float) -> None:
            for idx in probes:
                c = self.scene.voxel_center(*idx)
                ctx.painter.dot(c, self.theme.geometry, 4)
                for cam, u, v, d in self.facts.voxel_views(idx):
                    ground = self.rig.point_at(cam, u, 0.5, d)
                    ctx.painter.ray(c, (ground[0], ground[1], 0.05), t, self.theme.learned, 0.9, 2)
                    ctx.painter.ray(c, self.rig.cameras[cam].position, t, self.theme.ray, 0.8, 1)
                    ctx.images.dot(ctx.canvas, cam, u, v, self.theme.ray, 3, alpha=t)
            shown = cells[: int(round(t * len(cells)))]
            ctx.painter.voxels(
                (i, j, k, self.theme.classes[int(voxels[i, j, k])], 0.8) for i, j, k in shown
            )

        return Step(
            "Voxel reads range + elevation",
            "Each voxel samples the polar ray at its (column u, depth d) [purple: what is at this range] "
            "and the image at its pixel (u, v) [yellow: the row v encodes its height]. Both are "
            "camera-averaged and concatenated.",
            None,
            over,
            frames=11,
            objects_alpha=0.25,
        )
