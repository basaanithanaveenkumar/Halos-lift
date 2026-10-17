"""Alpamayo triplane lifter (Ivanovic et al., arXiv:2506.12251; Alpamayo-R1, arXiv:2511.00088).

Three orthogonal feature planes (Pxy, Pxz, Pyz) are built via deformable cross-attention
into multi-camera image features — identical architecture to TPVFormer — but the reference
points follow the Mip-NeRF 360 non-linear scene parametrization so the grid is denser near
the ego vehicle and coarser at long range.

The class also exposes :meth:`patchify_planes` to produce the fixed-length token sequence
used by Alpamayo's VLA policy head (72% fewer tokens than per-camera patch tokens).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.layers import HeightCompressor
from lifting.lifters.alpamayo.triplane_ref_points import AlpamayoReferencePoints
from lifting.lifters.base import LIFTERS, BaseLifter
from lifting.lifters.pyramid import flatten_pyramid
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.lifters.tpvformer.tpvformer_layer import TPVFormerLayer


@LIFTERS.register("alpamayo")
class AlpamayoTriplaneLifter(BaseLifter):
    """Lifts images to three orthogonal planes using Mip-NeRF 360 non-linear scene coords.

    Architecturally equivalent to :class:`TPVFormerLifter`, but the reference points are
    computed by :class:`AlpamayoReferencePoints` (Mip-NeRF 360 non-linear parametrization)
    instead of the uniform grid used by TPVFormer.  This is the perception module described
    in Ivanovic et al. 2025 (arXiv:2506.12251) and used by Alpamayo-R1 (arXiv:2511.00088).

    * :meth:`lift_planes` → :class:`TPVPlanes` (three feature planes).
    * :meth:`patchify_planes` → ``(B, N_tokens, C * p * p)`` flat VLA token sequence.
    * :meth:`forward` → ``(B, out_channels, X, Y)`` BEV map for benchmarking.

    Args:
        grid: metric region and resolution.
        in_channels: image feature channels.
        out_channels: BEV output channels.
        num_layers: number of transformer encoder layers.
        num_heads: attention heads per layer.
        num_levels: feature pyramid levels consumed.
        num_anchors: per-plane anchors along the normal axis (xy, xz, yz).
        points_per_anchor: deformable sampling points per anchor.
        hybrid_points: cross-view hybrid attention sample count.
        inner_frac: fraction of cells in the Mip-NeRF 360 inner high-res zone.
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
        hybrid_points: int = 4,
        inner_frac: float = 0.75,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.num_levels = num_levels
        self.inner_frac = inner_frac
        self.references = AlpamayoReferencePoints(grid, tuple(num_anchors), inner_frac)
        tokens = sum(h * w for h, w in self.references.plane_shapes)
        self.queries = nn.Parameter(0.1 * torch.randn(tokens, in_channels))
        self.query_pos = nn.Parameter(0.1 * torch.randn(tokens, in_channels))
        self.level_embed = nn.Parameter(torch.zeros(num_levels, in_channels))
        self.layers = nn.ModuleList(
            TPVFormerLayer(
                in_channels,
                num_heads,
                num_levels,
                tuple(num_anchors),
                points_per_anchor,
                hybrid_points,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.compressor = HeightCompressor(in_channels, grid.resolution[2], out_channels)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Lift image features to :class:`TPVPlanes` using Mip-NeRF 360 reference points."""
        values, shapes = flatten_pyramid(features[: self.num_levels], self.level_embed)
        b = values.shape[0]
        camera_reference = self.references.camera_reference_points(cameras)
        hybrid_reference = self.references.hybrid_reference_points(values.device)[None].expand(
            b, -1, -1, -1
        )
        query = self.queries[None].expand(b, -1, -1)
        pos = self.query_pos[None].expand(b, -1, -1)
        for layer in self.layers:
            query = layer(
                query,
                pos,
                self.references.plane_shapes,
                hybrid_reference,
                values,
                shapes,
                camera_reference,
            )
        return TPVPlanes.from_tokens(query, self.grid.resolution)

    def patchify_planes(self, planes: TPVPlanes, patch_size: int = 4) -> Tensor:
        """Fold the three triplane feature maps into a flat VLA token sequence.

        Each plane ``(B, C, H, W)`` is divided into non-overlapping ``patch_size × patch_size``
        patches, then flattened and projected to ``C * p * p`` tokens.  All three planes are
        concatenated along the token dimension.

        This matches Alpamayo-R1's tokenization step: for ``Sx=Sy=96``, ``Sz=48``,
        ``patch_size=8`` the total is ``(96/8)² + (96/8)×(48/8) + (96/8)×(48/8) = 144 + 72 + 72
        = 288`` tokens — 72% fewer than patch-based per-camera tokenization.

        Args:
            planes: output of :meth:`lift_planes`.
            patch_size: spatial patch side length (must divide all plane dims).

        Returns:
            ``(B, N_tokens, C * patch_size²)`` token tensor.
        """
        p = patch_size
        parts = []
        for plane in (planes.xy, planes.xz, planes.yz):
            b, c, h, w = plane.shape
            # Reshape into patches: (B, C, H/p, p, W/p, p) → (B, N, C*p*p)
            ph, pw = h // p, w // p
            x = plane.reshape(b, c, ph, p, pw, p)
            x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, ph * pw, c * p * p)
            parts.append(x)
        return torch.cat(parts, dim=1)

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """Return ``(B, out_channels, X, Y)`` BEV map (for benchmarking)."""
        return self.compressor(self.lift_planes(features, cameras).to_voxels())
