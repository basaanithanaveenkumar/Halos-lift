"""Turn *any* registered BEV lifter (including user plug-ins) into a triplane lifter."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.base import LIFTERS
from lifting.lifters.tpvformer.bev_to_tpv import BEVToTPV
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter


@TRIPLANE_LIFTERS.register("bev_adapter")
class BEVLifterTriplaneAdapter(BaseTriplaneLifter):
    """``LIFTERS[base]`` -> BEV map -> :class:`BEVToTPV` -> planes.

    The generic fallback: works for every entry of ``LIFTERS``, also lifters registered by
    users (e.g. ``examples/custom_lifter.py``). ``xy`` is the BEV map itself; ``xz`` / ``yz``
    are *learned* expansions of the BEV map pooled along ``y`` / ``x``, because height is no
    longer present in a BEV map. Prefer the dedicated triplane lifters when they exist - they
    recover height from the images.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        base: name of the wrapped lifter in :data:`lifting.lifters.LIFTERS`.
        base_kwargs: extra constructor arguments for the wrapped lifter.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        base: str = "simple_bev",
        base_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.base = base
        self.core = LIFTERS.build(
            base,
            grid=grid,
            in_channels=in_channels,
            out_channels=in_channels,
            **dict(base_kwargs or {}),
        )
        # Geometry dependence is inherited from the wrapped lifter (instance attribute).
        self.uses_geometry = self.core.uses_geometry
        self.to_planes = BEVToTPV(in_channels, out_channels, grid.resolution)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Lift per-camera features into the three TPV planes.

        Args:
            features: feature pyramid from the image encoder.
            cameras: the rig geometry.

        Returns:
            The ``xy`` / ``xz`` / ``yz`` feature planes.
        """
        return self.to_planes(self.core(features, cameras))

    @classmethod
    def resolve_kwargs(cls, kwargs: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
        """Forward the build context to the *wrapped* lifter's constructor."""
        out = dict(kwargs)
        base_cls = LIFTERS.get(out.get("base", "simple_bev"))
        params = inspect.signature(base_cls).parameters
        base_kwargs = dict(out.get("base_kwargs") or {})
        for key, value in context.items():
            if key in params and key not in base_kwargs:
                base_kwargs[key] = value
        out["base_kwargs"] = base_kwargs
        return out
