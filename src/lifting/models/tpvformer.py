"""Full TPVFormer for 3D semantic occupancy / LiDAR segmentation."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.encoders import ImageEncoder
from lifting.geometry import Cameras
from lifting.lifters.tpvformer import TPVAggregator, TPVFormerLifter


class TPVFormer(nn.Module):
    """Image encoder -> TPV encoder -> TPV aggregator.

    Args:
        encoder, lifter, aggregator: the three stages.
        aux_heads: optional ``(xy, xz, yz)`` :class:`PlaneAuxHead` modules; when given the
            output gains ``aux_logits`` for per-plane supervision.
    """

    def __init__(
        self,
        encoder: ImageEncoder,
        lifter: TPVFormerLifter,
        aggregator: TPVAggregator,
        aux_heads: Sequence[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lifter = lifter
        self.aggregator = aggregator
        if aux_heads is not None and len(aux_heads) != 3:
            raise ValueError(f"aux_heads needs one head per plane (3), got {len(aux_heads)}")
        self.aux_heads = nn.ModuleList(aux_heads) if aux_heads is not None else None

    def forward(self, images: Tensor, cameras: Cameras, points: Tensor | None = None) -> dict:
        """
        Args:
            images: ``(B, N, 3, H, W)`` in ``[0, 1]``.
            cameras: calibration.
            points: optional ``(B, P, 3)`` metric points to classify.

        Returns:
            ``planes`` (:class:`TPVPlanes`), ``voxel_logits`` ``(B, K, X, Y, Z)`` and
            optionally ``point_logits`` ``(B, K, P)``, and with ``aux_heads`` the per-plane
            ``aux_logits`` ``((B, K, X, Y), (B, K, X, Z), (B, K, Y, Z))``.
        """
        b, n = images.shape[:2]
        feats = [f.view(b, n, *f.shape[1:]) for f in self.encoder(images.flatten(0, 1))]
        planes = self.lifter.lift_planes(feats, cameras)
        out = {"planes": planes, **self.aggregator(planes, points)}
        if self.aux_heads is not None:
            out["aux_logits"] = tuple(
                head(plane)
                for head, plane in zip(self.aux_heads, (planes.xy, planes.xz, planes.yz), strict=True)
            )
        return out
