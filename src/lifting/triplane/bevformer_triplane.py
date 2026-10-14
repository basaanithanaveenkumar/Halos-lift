"""BEVFormer -> TPV: one BEVFormer query encoder per plane, anchors along each plane normal."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.pyramid import flatten_pyramid
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.lifters.tpvformer.tpv_reference_points import TPVReferencePoints
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.plane_projection import PlaneProjection
from lifting.triplane.plane_query_encoder import PlaneQueryEncoder


@TRIPLANE_LIFTERS.register("bevformer")
class BEVFormerTriplaneLifter(BaseTriplaneLifter):
    """BEVFormer's spatial encoder run on all three planes.

    * ``xy`` queries own pillars along ``z`` - exactly :class:`BEVFormerLifter`;
    * ``xz`` queries own lines along ``y``; ``yz`` queries own lines along ``x``.

    Each plane has its own learned queries, in-plane deformable self-attention and spatial
    cross-attention into the cameras (:class:`PlaneQueryEncoder`). Anchors come from
    :class:`TPVReferencePoints`, so the geometry is identical to TPVFormer's - the only
    difference is that the planes never talk to each other (no cross-view hybrid attention),
    which makes this the "independent planes" ablation of TPVFormer.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        num_layers: encoder depth per plane.
        num_heads: attention heads.
        num_levels: pyramid levels read by the cross-attention (1 = ``bevformernet``).
        num_anchors: anchors per query for the ``xy``, ``xz`` and ``yz`` planes.
        points_per_anchor: deformable sampling points per anchor.
        self_attn_points: sampling points of the in-plane self-attention.
        dropout: dropout probability.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        num_levels: int = 1,
        num_anchors: tuple[int, int, int] = (4, 8, 8),
        points_per_anchor: int = 2,
        self_attn_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.num_levels = num_levels
        self.references = TPVReferencePoints(grid, tuple(num_anchors))
        self.level_embed = nn.Parameter(torch.zeros(num_levels, in_channels))
        self.encoders = nn.ModuleList(
            PlaneQueryEncoder(
                shape,
                in_channels,
                num_layers,
                num_heads,
                num_levels,
                anchors,
                points_per_anchor,
                self_attn_points,
                dropout,
            )
            for shape, anchors in zip(self.references.plane_shapes, num_anchors, strict=True)
        )
        self.projection = PlaneProjection(in_channels, out_channels)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        values, shapes = flatten_pyramid(features[: self.num_levels], self.level_embed)
        camera_reference = self.references.camera_reference_points(cameras)
        planes = [
            encoder(values, shapes, ref, visible)
            for encoder, (ref, visible) in zip(self.encoders, camera_reference, strict=True)
        ]
        return self.projection(TPVPlanes(*planes))
