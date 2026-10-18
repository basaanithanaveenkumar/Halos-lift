"""Depth accuracy per range bin (are errors near or far?)."""

from __future__ import annotations

from torch import Tensor


class RangeErrorMetric:
    """Per-range-bin depth absolute relative error.

    WHY range-stratified error:
        Monocular depth errors are not uniform across depth ranges.  Nearby
        objects are generally predicted more accurately than distant ones
        because they occupy more pixels and have stronger texture cues.

        Reporting abs_rel per range bin (e.g. 0–10 m, 10–30 m, 30–80 m)
        reveals whether a model's limitations are in near-range (safety-critical
        for robotics) or far-range (less critical for manipulation).

        This metric is particularly important for Halos-lift because the
        lifting pipeline may be specialised for a specific depth range.

    Args:
        range_bins: edges of the depth ranges in metres.
                    Default: [0, 10, 30, 80] → three bins: 0-10, 10-30, 30-80.
    """

    def __init__(
        self,
        range_bins: tuple[float, ...] = (0.0, 10.0, 30.0, 80.0),
    ) -> None:
        self.range_bins = range_bins
        self.n_bins = len(range_bins) - 1
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._abs_rel_sum = [0.0] * self.n_bins
        self._n_per_bin = [0] * self.n_bins

    def update(
        self,
        pred_depth: Tensor,  # any shape
        gt_depth: Tensor,  # same shape
        mask: Tensor | None = None,
    ) -> None:
        """Accumulate one batch of predictions and targets."""
        p = pred_depth.float().detach().cpu().reshape(-1)
        g = gt_depth.float().detach().cpu().reshape(-1)

        if mask is not None:
            valid = mask.bool().detach().cpu().reshape(-1)
            p, g = p[valid], g[valid]

        abs_rel = (p - g).abs() / g.clamp(min=1e-6)

        for i in range(self.n_bins):
            lo = self.range_bins[i]
            hi = self.range_bins[i + 1]
            bin_mask = (g >= lo) & (g < hi)
            n = int(bin_mask.sum())
            if n == 0:
                continue
            self._abs_rel_sum[i] += float(abs_rel[bin_mask].sum())
            self._n_per_bin[i] += n

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`."""
        out: dict[str, float] = {}
        for i in range(self.n_bins):
            lo = self.range_bins[i]
            hi = self.range_bins[i + 1]
            n = self._n_per_bin[i]
            key = f"range_abs_rel_{lo:.0f}_{hi:.0f}m"
            out[key] = self._abs_rel_sum[i] / max(n, 1)
        return out
