"""Mip-NeRF 360 non-linear scene parametrization (Barron et al., CVPR 2022).

Used by the Alpamayo triplane tokenizer (Ivanovic et al., arXiv:2506.12251) to place
denser grid cells near the ego vehicle and sparser cells at long range, without needing
an explicit "maximum range" cutoff.
"""

from __future__ import annotations

import torch
from torch import Tensor


class MipNeRF360Parametrizer:
    """Non-linear mapping from integer grid index to ego-relative metric coordinate.

    Splits each axis into two regions:
    - **Inner**: ``floor(total * inner_frac)`` cells, evenly spaced in
      ``[-half_extent, +half_extent]`` — high resolution near the ego.
    - **Outer**: remaining cells, coarser, covering the tails on each side.

    Following the Mip-NeRF 360 contract, the outer region compresses unbounded
    metric space into a finite number of cells, so the model still has some
    representation of very distant content.

    Args:
        total: total cells along this axis.
        inner_frac: fraction of cells in the high-res inner region (default 0.75).
        total_extent_m: total metric span of this axis in metres.
    """

    def __init__(
        self,
        total: int,
        inner_frac: float = 0.75,
        total_extent_m: float = 32.0,
    ) -> None:
        self.total = total
        self.inner_frac = inner_frac
        self.total_extent_m = total_extent_m

    def cell_centers_m(self, device: torch.device | str | None = None) -> Tensor:
        """Return ``(total,)`` metric positions of cell centres on this axis.

        The positions are monotonically increasing and sorted: outer-left, inner,
        outer-right.  They are NOT uniformly spaced — inner cells are denser.
        """
        n_inner = int(self.total * self.inner_frac)
        n_outer = self.total - n_inner

        half = self.total_extent_m * self.inner_frac / 2.0

        # Inner cells: evenly spaced in [-half, +half]
        r_inner = (2.0 * half) / n_inner
        inner = torch.linspace(
            -half + r_inner / 2.0,
            half - r_inner / 2.0,
            n_inner,
            device=device,
            dtype=torch.float32,
        )

        # Outer cells: split between left tail and right tail
        n_left = n_outer // 2
        n_right = n_outer - n_left

        lo = -self.total_extent_m / 2.0
        hi = self.total_extent_m / 2.0

        if n_left > 0:
            r_outer_l = (half - (-half)) / (n_left + 1)
            outer_left = torch.linspace(
                lo + r_outer_l / 2.0,
                -half - r_outer_l / 2.0,
                n_left,
                device=device,
                dtype=torch.float32,
            )
        else:
            outer_left = torch.empty(0, device=device, dtype=torch.float32)

        if n_right > 0:
            r_outer_r = (hi - half) / (n_right + 1)
            outer_right = torch.linspace(
                half + r_outer_r / 2.0,
                hi - r_outer_r / 2.0,
                n_right,
                device=device,
                dtype=torch.float32,
            )
        else:
            outer_right = torch.empty(0, device=device, dtype=torch.float32)

        return torch.cat([outer_left, inner, outer_right])
