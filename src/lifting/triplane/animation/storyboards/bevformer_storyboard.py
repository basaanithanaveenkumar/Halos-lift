"""BEVFormer -> triplane (one query encoder per plane), step by step."""

from __future__ import annotations

from lifting.triplane.animation.query_storyboard import QueryStoryboard
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS


@STORYBOARDS.register("bevformer")
class BEVFormerStoryboard(QueryStoryboard):
    """BEVFormer's spatial encoder run independently on the xy, xz and yz planes."""

    title = "BEVFormer  →  triplane"
    reference = "Li et al., BEVFormer (ECCV 2022)  ·  lifting.triplane.BEVFormerTriplaneLifter"

    def steps(self) -> list[Step]:
        """Build the ordered animation steps explaining this lifter."""
        return [
            self.step_inputs(),
            self.step_queries(
                "Three PlaneQueryEncoders, one per plane: a learned query vector for every cell of xy "
                "(X·Y), xz (X·Z) and yz (Y·Z), plus a learned positional embedding."
            ),
            self.step_anchors(),
            self.step_project_anchors(),
            self.step_deformable(
                "Deformable attention (in-plane + image)",
                "Self-attention: each query samples a few learned offsets on its OWN plane. Spatial "
                "cross-attention: around every reference pixel it samples learned offsets and averages "
                "the cameras that see it. The planes never exchange information.",
            ),
            self.step_refine(
                "BEVFormerLayer x num_layers per plane; a 1x1 PlaneProjection maps the queries to "
                "out_channels. xy alone is exactly BEVFormerLifter."
            ),
            self.step_output(),
        ]
