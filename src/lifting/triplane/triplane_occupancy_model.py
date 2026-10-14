"""Encoder -> any triplane lifter -> TPV aggregator (3D semantic occupancy)."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from lifting.encoders import ImageEncoder
from lifting.lifters.tpvformer.tpv_aggregator import TPVAggregator
from lifting.models.tpvformer import TPVFormer
from lifting.triplane.base import BaseTriplaneLifter


class TriplaneOccupancyModel(TPVFormer):
    """:class:`TPVFormer` generalised to every :class:`BaseTriplaneLifter`.

    ``TPVFormer.forward`` only calls ``lifter.lift_planes`` - the contract every triplane lifter
    satisfies - so the full forward pass (``planes``, ``voxel_logits``, ``point_logits`` and
    optional per-plane ``aux_logits``) is inherited unchanged and trains with
    :class:`lifting.training.OccupancyTask`.

    Args:
        encoder: image encoder.
        lifter: any triplane lifter.
        aggregator: plane -> voxel / point classifier.
        aux_heads: optional ``(xy, xz, yz)`` :class:`PlaneAuxHead` modules.
    """

    def __init__(
        self,
        encoder: ImageEncoder,
        lifter: BaseTriplaneLifter,
        aggregator: TPVAggregator,
        aux_heads: Sequence[nn.Module] | None = None,
    ) -> None:
        if not isinstance(lifter, BaseTriplaneLifter):
            raise TypeError(f"lifter must be a BaseTriplaneLifter, got {type(lifter).__name__}")
        super().__init__(encoder, lifter, aggregator, aux_heads)
        # Occupancy only consumes the planes; the lifter's BEV head would never get a gradient.
        self.lifter.bev_head.requires_grad_(False)
