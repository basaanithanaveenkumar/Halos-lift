"""The single interface every triplane (TPV) lifter implements, and its registry."""

from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.base import BaseLifter
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.registry import Registry
from lifting.triplane.triplane_bev_head import TriplaneBEVHead


class BaseTriplaneLifter(BaseLifter):
    """Turns multi-camera image features into three orthogonal feature planes.

    A triplane lifter is a :class:`BaseLifter` (Liskov): :meth:`forward` still returns a
    ``(B, out_channels, X, Y)`` BEV map, so it drops into ``BEVSegmentationModel`` and the
    verifier unchanged. On top of that it exposes :meth:`lift_planes`, which returns
    :class:`TPVPlanes`::

        xy: (B, out_channels, X, Y)   top view   (this *is* the BEV plane)
        xz: (B, out_channels, X, Z)   side view
        yz: (B, out_channels, Y, Z)   front view

    so it can also feed :class:`TPVAggregator` for 3D occupancy and point queries.

    Design decisions:

    * The BEV output is derived from the planes (``planes -> voxels -> height compression``)
      rather than from the original 2D lifter, so the BEV path and the TPV path share every
      parameter and the BEV is guaranteed to be consistent with the planes.
    * Subclasses only implement :meth:`lift_planes` (template method); the BEV head lives here.

    Args:
        grid: metric region and resolution of the planes.
        in_channels: channels of every image feature level.
        out_channels: channels of every plane and of the BEV map.
    """

    def __init__(self, grid: GridSpec, in_channels: int, out_channels: int) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.bev_head = TriplaneBEVHead(out_channels, grid.resolution[2], out_channels)

    @abstractmethod
    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Lift ``features`` (``(B, N, C, H_l, W_l)`` per level, fine -> coarse) to planes."""

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """``(B, out_channels, X, Y)`` BEV map computed from the three planes."""
        return self.bev_head(self.lift_planes(features, cameras))

    @classmethod
    def resolve_kwargs(cls, kwargs: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
        """Fill constructor arguments the caller did not give from build ``context``.

        ``context`` holds values derived from the encoder / rig (``feature_size``,
        ``num_cameras``, ``depth_bins``). Only keys that appear in the constructor signature
        are injected; explicit ``kwargs`` always win. Wrappers override this to forward the
        context to the lifter they wrap.
        """
        params = inspect.signature(cls).parameters
        out = dict(kwargs)
        for key, value in context.items():
            if key in params and key not in out:
                out[key] = value
        return out


#: Registry of triplane lifters. Names mirror :data:`lifting.lifters.LIFTERS` so every BEV
#: method has a TPV counterpart: ``TRIPLANE_LIFTERS.build("lift_splat", grid=..., ...)``.
#: It is deliberately a *separate* registry so the existing ``LIFTERS`` set is unchanged.
TRIPLANE_LIFTERS: Registry[BaseTriplaneLifter] = Registry("triplane_lifter")
