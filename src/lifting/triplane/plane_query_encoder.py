"""A BEVFormer-style query encoder for one arbitrary TPV plane."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lifting.lifters.bevformer.bevformer_layer import BEVFormerLayer


class PlaneQueryEncoder(nn.Module):
    """Learned queries on one ``(H, W)`` plane refined by a stack of :class:`BEVFormerLayer`.

    :class:`BEVFormerLifter` is this module for the ``xy`` plane with pillar anchors. Nothing in
    a BEVFormer layer is specific to the ground plane: the self-attention only needs the plane
    shape and cell centres, and the image cross-attention only needs projected anchors. Running
    it on the ``xz`` / ``yz`` planes with anchors along ``y`` / ``x`` gives a per-plane
    BEVFormer (no cross-plane attention - that is what TPVFormer adds on top).

    Args:
        plane_shape: ``(H, W)`` = rows x columns of the plane.
        embed_dims: query width (= image feature channels).
        num_layers: encoder depth.
        num_heads: attention heads.
        num_levels: image pyramid levels read by the cross-attention.
        num_anchors: 3D anchors per query along the plane normal.
        points_per_anchor: deformable sampling points per anchor.
        self_attn_points: sampling points of the in-plane deformable self-attention.
        dropout: dropout probability.
    """

    def __init__(
        self,
        plane_shape: tuple[int, int],
        embed_dims: int,
        num_layers: int = 2,
        num_heads: int = 4,
        num_levels: int = 1,
        num_anchors: int = 4,
        points_per_anchor: int = 2,
        self_attn_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.plane_shape = tuple(plane_shape)
        cells = self.plane_shape[0] * self.plane_shape[1]
        # Same small-normal init as BEVFormerLifter / TPVFormerLifter queries.
        self.queries = nn.Parameter(0.1 * torch.randn(cells, embed_dims))
        self.query_pos = nn.Parameter(0.1 * torch.randn(cells, embed_dims))
        self.layers = nn.ModuleList(
            BEVFormerLayer(
                embed_dims,
                num_heads,
                num_levels,
                num_anchors * points_per_anchor,
                num_anchors,
                self_attn_points,
                dropout,
            )
            for _ in range(num_layers)
        )

    def reference_points(self, device: torch.device | str | None = None) -> Tensor:
        """``(H*W, 2)`` cell centres as normalised ``(col, row)`` in ``[0, 1]``."""
        h, w = self.plane_shape
        rows = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
        cols = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
        r, c = torch.meshgrid(rows, cols, indexing="ij")
        return torch.stack([c, r], -1).view(-1, 2)

    def forward(
        self,
        image_values: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
        reference_points_cam: Tensor,
        visible: Tensor,
    ) -> Tensor:
        """
        Args:
            image_values: ``(B, N, S, C)`` flattened multi-camera feature pyramid.
            spatial_shapes: ``L`` tuples ``(H_l, W_l)``.
            reference_points_cam: ``(B, N, H*W, A, 2)`` projected anchors in ``[0, 1]``.
            visible: ``(B, N, H*W, A)`` anchor visibility.

        Returns:
            ``(B, C, H, W)`` plane features.
        """
        b = image_values.shape[0]
        ref = self.reference_points(image_values.device)[None].expand(b, -1, -1)
        query = self.queries[None].expand(b, -1, -1)
        pos = self.query_pos[None].expand(b, -1, -1)
        for layer in self.layers:
            query = layer(
                query,
                pos,
                ref,
                self.plane_shape,
                image_values,
                spatial_shapes,
                reference_points_cam,
                visible,
            )
        h, w = self.plane_shape
        return query.transpose(1, 2).reshape(b, -1, h, w)
