"""Alpamayo (Mip-NeRF 360 triplane) in the triplane interface."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.alpamayo.alpamayo_lifter import AlpamayoTriplaneLifter
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.plane_projection import PlaneProjection

#: In-plane axes (rows, cols) of the xy, xz and yz planes.
_PLANE_AXES = ((0, 1), (0, 2), (1, 2))


@TRIPLANE_LIFTERS.register("alpamayo")
class AlpamayoTriplaneAdapter(BaseTriplaneLifter):
    """Wraps :class:`AlpamayoTriplaneLifter` (TPVFormer layers with Mip-NeRF 360 non-uniform
    reference points: dense near the ego, coarse far away) behind the triplane contract.

    Alpamayo's planes are **non-uniform**: cell ``i`` of an axis sits at
    ``MipNeRF360Parametrizer.cell_centers_m()[i]``, not at the grid's uniform cell centre. Every
    consumer of the triplane contract (``TriplaneBEVHead``, ``TPVAggregator`` voxel and point
    decoding, per-plane auxiliary losses, occupancy labels) indexes planes on the grid's uniform
    cells, so:

    * :meth:`lift_native_planes` returns the planes in Alpamayo's own non-uniform layout - use
      these for :meth:`patchify_planes` (the VLA token sequence, as in Alpamayo-R1);
    * :meth:`lift_planes` resamples them onto the uniform grid (bilinear, at the exact
      continuous index of every uniform cell centre), so cell ``(i, j)`` means the same metric
      location as for every other triplane lifter.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        **kwargs: forwarded to :class:`AlpamayoTriplaneLifter` (``inner_frac`` etc.).
    """

    def __init__(self, grid: GridSpec, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.core = AlpamayoTriplaneLifter(grid, in_channels, in_channels, **kwargs)
        self.core.compressor = nn.Identity()  # BEV comes from BaseTriplaneLifter.bev_head
        self.projection = PlaneProjection(in_channels, out_channels)
        # Continuous native index of every uniform cell centre, per axis (fixed geometry).
        for axis in range(3):
            parametrizer = self.core.references.parametrizer(axis)
            self.register_buffer(
                f"_native_index_{axis}",
                parametrizer.metric_to_index(grid.axis_centers(axis)),
                persistent=False,
            )

    def lift_native_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Planes in Alpamayo's non-uniform (Mip-NeRF 360) cell layout."""
        return self.projection(self.core.lift_planes(features, cameras))

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Planes resampled onto the grid's uniform cells (the shared triplane contract)."""
        return self.to_uniform(self.lift_native_planes(features, cameras))

    def to_uniform(self, planes: TPVPlanes) -> TPVPlanes:
        """Resample native non-uniform planes at the grid's uniform cell centres."""
        resampled = [
            self._resample(plane, *axes)
            for plane, axes in zip((planes.xy, planes.xz, planes.yz), _PLANE_AXES, strict=True)
        ]
        return TPVPlanes(*resampled)

    def patchify_planes(self, planes: TPVPlanes, patch_size: int = 4) -> Tensor:
        """``(B, N_tokens, C * p * p)`` tokens - see :meth:`AlpamayoTriplaneLifter.patchify_planes`.

        Pass :meth:`lift_native_planes` output to tokenise Alpamayo's own layout.
        """
        return self.core.patchify_planes(planes, patch_size)

    def _resample(self, plane: Tensor, row_axis: int, col_axis: int) -> Tensor:
        rows = self._normalized_index(row_axis)
        cols = self._normalized_index(col_axis)
        r, c = torch.meshgrid(rows, cols, indexing="ij")
        grid = torch.stack([c, r], -1)[None].expand(plane.shape[0], -1, -1, -1)
        # align_corners=True: -1 / +1 are the centres of the first / last native cell.
        return F.grid_sample(
            plane, grid.to(plane.dtype), mode="bilinear", padding_mode="border", align_corners=True
        )

    def _normalized_index(self, axis: int) -> Tensor:
        index: Tensor = getattr(self, f"_native_index_{axis}")
        n = self.grid.resolution[axis]
        return torch.zeros_like(index) if n == 1 else 2.0 * index / (n - 1) - 1.0
