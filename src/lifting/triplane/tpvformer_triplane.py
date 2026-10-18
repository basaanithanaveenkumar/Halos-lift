"""TPVFormer in the triplane interface (it is natively a triplane lifter)."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.lifters.tpvformer.tpvformer_lifter import TPVFormerLifter
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.plane_projection import PlaneProjection


@TRIPLANE_LIFTERS.register("tpvformer")
class TPVFormerTriplaneLifter(BaseTriplaneLifter):
    """Wraps :class:`TPVFormerLifter` (planes from learned queries, cross-view hybrid attention
    between planes and per-plane image cross-attention) and projects its planes to
    ``out_channels`` so all triplane lifters share one contract.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        **kwargs: forwarded to :class:`TPVFormerLifter` (``num_layers``, ``num_heads``,
            ``num_levels``, ``num_anchors``, ``points_per_anchor``, ``hybrid_points``, ``dropout``).
    """

    def __init__(self, grid: GridSpec, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.core = TPVFormerLifter(grid, in_channels, in_channels, **kwargs)
        self.core.compressor = nn.Identity()  # BEV comes from BaseTriplaneLifter.bev_head
        self.projection = PlaneProjection(in_channels, out_channels)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Lift per-camera features into the three TPV planes.

        Args:
            features: feature pyramid from the image encoder.
            cameras: the rig geometry.

        Returns:
            The ``xy`` / ``xz`` / ``yz`` feature planes.
        """
        return self.projection(self.core.lift_planes(features, cameras))
