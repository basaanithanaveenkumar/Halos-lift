"""Mip-NeRF 360 non-linear scene parametrization (Barron et al., CVPR 2022).

Used by the Alpamayo triplane tokenizer (Ivanovic et al., arXiv:2506.12251) to place
denser grid cells near the ego vehicle and sparser cells at long range.

Mip-NeRF 360 (eq. 10) contracts unbounded space into a ball of radius 2::

    contract(x) = x                         if |x| <= 1
                  (2 - 1 / |x|) * x / |x|    otherwise

A grid that is **uniform in contracted space** is therefore linear (and dense) inside the unit
ball and grows like ``1 / (2 - s)`` outside it. This module applies the contraction per axis,
scaled so the unit ball becomes an *inner zone* of radius ``inner_radius_m`` and the outermost
cell edge lands exactly on the axis bounds.
"""

from __future__ import annotations

import torch
from torch import Tensor


class MipNeRF360Parametrizer:
    """Non-linear mapping from integer cell index to metric coordinate on one axis.

    Cells are spaced uniformly in Mip-NeRF 360 contracted space ``s``:

    * **Inner zone** (``|s| <= 1``): ``n_inner`` cells, linear in metres, covering
      ``[-inner_radius_m, +inner_radius_m]`` around ``center_m``.
    * **Outer tails** (``1 < |s| <= s_max``): ``n_outer / 2`` cells per side whose width grows
      with distance, mapped through the inverse contraction ``x = 1 / (2 - |s|)``.

    Because every cell has the same width ``ds = 2 / n_inner`` in ``s``, the metric width is
    continuous at the zone boundary and increases monotonically outwards; the inner cells are
    always narrower than a uniform grid's ``total_extent_m / total``. ``inner_radius_m`` is
    derived so the outermost edge equals ``total_extent_m / 2``::

        s_max          = 1 + (n_outer / 2) * ds
        inner_radius_m = (total_extent_m / 2) * (2 - s_max)

    The contraction only fits inside ``|s| < 2``, so the tails need fewer cells than the inner
    zone (``n_outer < n_inner``, hence ``inner_frac`` > 0.5).

    ``n_inner`` starts at ``floor(total * inner_frac)``; it gains one cell if ``n_outer`` would be
    odd (both tails stay symmetric) and two more while rounding leaves ``n_outer >= n_inner``.
    Tiny axes such as 4 height cells therefore become a plain uniform grid.

    Args:
        total: total cells along this axis.
        inner_frac: fraction of cells in the high-resolution inner zone, in ``(0.5, 1]``.
        total_extent_m: metric span of this axis (``hi - lo``) in metres.
        center_m: metric centre of the axis (``(lo + hi) / 2``); cells are placed around it.
    """

    def __init__(
        self,
        total: int,
        inner_frac: float = 0.75,
        total_extent_m: float = 32.0,
        center_m: float = 0.0,
    ) -> None:
        if total < 1:
            raise ValueError(f"total must be >= 1, got {total}")
        if not 0.5 < inner_frac <= 1.0:
            raise ValueError(
                f"inner_frac must be in (0.5, 1] (the contracted tails need fewer cells than the "
                f"inner zone), got {inner_frac}"
            )
        if total_extent_m <= 0:
            raise ValueError(f"total_extent_m must be > 0, got {total_extent_m}")
        self.total = total
        self.inner_frac = inner_frac
        self.total_extent_m = total_extent_m
        self.center_m = center_m

        n_inner = max(1, int(total * inner_frac))
        if (total - n_inner) % 2:  # odd outer count -> give the spare cell to the inner zone
            n_inner += 1
        while total - n_inner >= n_inner:  # rounding at small totals: keep s_max < 2
            n_inner += 2
        self.n_inner = n_inner
        self.n_outer = total - n_inner

    # ------------------------------------------------------------------ derived geometry
    @property
    def ds(self) -> float:
        """Cell width in contracted space (identical for every cell)."""
        return 2.0 / self.n_inner

    @property
    def s_max(self) -> float:
        """Contracted coordinate of the outermost cell edge (``1 <= s_max < 2``)."""
        return 1.0 + (self.n_outer // 2) * self.ds

    @property
    def inner_radius_m(self) -> float:
        """Metric half-width of the inner (linear, high-resolution) zone."""
        return (self.total_extent_m / 2.0) * (2.0 - self.s_max)

    # ------------------------------------------------------------------ mapping
    def contracted_to_metric(self, s: Tensor) -> Tensor:
        """Inverse Mip-NeRF 360 contraction, scaled to metres and shifted to ``center_m``."""
        mag = s.abs()
        unit = torch.where(mag <= 1.0, s, torch.sign(s) / (2.0 - mag))
        return self.center_m + self.inner_radius_m * unit

    def metric_to_index(self, x: Tensor) -> Tensor:
        """Continuous cell index of metric coordinates (inverse of :meth:`cell_centers_m`).

        Cell ``i``'s centre maps to exactly ``i``; values in between interpolate in contracted
        space, so resampling a non-uniform plane at these indices is exact inside the linear
        zone and smooth in the tails.
        """
        unit = (x - self.center_m) / self.inner_radius_m
        mag = unit.abs()
        s = torch.where(mag <= 1.0, unit, torch.sign(unit) * (2.0 - 1.0 / mag.clamp_min(1.0)))
        return (s + self.s_max) / self.ds - 0.5

    def cell_edges_m(self, device: torch.device | str | None = None) -> Tensor:
        """``(total + 1,)`` metric cell boundaries, from ``center - extent/2`` to ``center + extent/2``."""
        s = torch.linspace(-self.s_max, self.s_max, self.total + 1, dtype=torch.float64)
        return self.contracted_to_metric(s).to(device=device, dtype=torch.float32)

    def cell_centers_m(self, device: torch.device | str | None = None) -> Tensor:
        """``(total,)`` metric cell centres, strictly increasing.

        Centres are taken at the middle of each cell *in contracted space*, the natural sample
        point of a grid that is uniform in ``s``.
        """
        # Computed in float64 on the CPU (MPS has no float64), then moved to ``device``.
        s = -self.s_max + (torch.arange(self.total, dtype=torch.float64) + 0.5) * self.ds
        return self.contracted_to_metric(s).to(device=device, dtype=torch.float32)
