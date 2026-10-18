"""
Monocular depth evaluation metrics for Halos-lift.

Standard suite from Eigen et al. "Depth Map Prediction from a Single Image
using a Multi-Scale Deep Network", NeurIPS 2014 — the canonical reference
for depth evaluation on KITTI and NYUv2 benchmarks.

All sub-metrics are computed by a single ``DepthMetrics`` class to avoid
redundant computation (the same masked depth arrays are reused for every
sub-metric).

Metric guide
------------
abs_rel    — Mean relative error.  Scale-invariant; a metric of 0.1 means
             predictions are off by 10% of the true depth on average.
             Lower is better.  KITTI stereo baselines: ~0.10–0.15.

sq_rel     — Mean squared relative error.  Penalises large relative errors
             more than abs_rel.  Useful for detecting tail failures.
             Lower is better.

rmse       — Root mean squared error in depth units (e.g. metres).
             Dominated by large errors; not scale-invariant.
             Lower is better.

rmse_log   — RMSE in log-depth space.  Less sensitive to large absolute
             depths (far range), giving equal weight to near and far.
             Lower is better.

silog      — Scale-invariant log error (Eigen et al.).  Measures the
             variance of log-depth differences, independent of the overall
             scale of the prediction.  Lower is better.

d1/d2/d3   — Threshold accuracies: fraction of pixels where
             max(d_pred/d_gt, d_gt/d_pred) < τ  (τ = 1.25, 1.25², 1.25³).
             These are the most intuitive quality measures.
             Higher is better.  Good models: d1 > 0.90.

abs_rel_scaled — Like abs_rel but with per-sample median alignment: the
             prediction is scaled so median(pred) = median(gt) before
             evaluation.  Measures the quality of *relative* depth (structure)
             independent of any global scale ambiguity.
"""

from __future__ import annotations

import torch
from torch import Tensor


class DepthMetrics:
    """Accumulate all standard monocular depth metrics over a dataset.

    Args:
        min_depth: clip predictions and GT below this value (avoids log(0)).
        max_depth: clip predictions and GT above this value (removes sky/inf).

    Usage::

        metric = DepthMetrics(min_depth=1e-3, max_depth=80.0)
        for pred, gt, mask in dataloader:
            metric.update(pred, gt, mask)
        results = metric.compute()
    """

    def __init__(self, min_depth: float = 1e-3, max_depth: float = 80.0) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        # Accumulators — each is (running_sum, n_pixels)
        self._abs_rel_sum = 0.0
        self._sq_rel_sum = 0.0
        self._rmse_sum = 0.0
        self._rmse_log_sum = 0.0
        self._silog_sq_sum = 0.0  # for E[δ²] term of silog
        self._silog_lin_sum = 0.0  # for E[δ]² term of silog
        self._d1_sum = 0.0
        self._d2_sum = 0.0
        self._d3_sum = 0.0
        self._abs_rel_sc_sum = 0.0  # median-scaled abs_rel
        self._n = 0  # total valid pixel count

    def update(
        self,
        pred_depth: Tensor,  # any shape; values in depth units
        gt_depth: Tensor,  # same shape
        mask: Tensor | None = None,  # boolean; True = valid GT pixel
    ) -> None:
        """Accumulate metrics for one sample or batch.

        Args:
            pred_depth: predicted depth (same shape as gt_depth).
            gt_depth:   ground-truth depth.
            mask:       boolean mask of valid GT pixels.  If None, all pixels
                        with gt_depth > min_depth are used.
        """
        pred = pred_depth.float().detach()
        gt = gt_depth.float().detach()

        # ── validity mask ────────────────────────────────────────────────────
        if mask is None:
            valid = (gt > self.min_depth) & (gt < self.max_depth)
        else:
            valid = mask.bool() & (gt > self.min_depth) & (gt < self.max_depth)

        pred = pred[valid].clamp(self.min_depth, self.max_depth)
        gt = gt[valid].clamp(self.min_depth, self.max_depth)

        n = gt.numel()
        if n == 0:
            return

        # ── basic ratio metrics ───────────────────────────────────────────────
        # abs_rel: mean(|pred - gt| / gt) — relative absolute error
        self._abs_rel_sum += float(((pred - gt).abs() / gt).mean().item()) * n

        # sq_rel: mean((pred - gt)² / gt) — squared relative error (more
        #         sensitive to outliers than abs_rel)
        self._sq_rel_sum += float(((pred - gt).pow(2) / gt).mean().item()) * n

        # rmse: sqrt(mean((pred - gt)²)) — error in depth units
        self._rmse_sum += float((pred - gt).pow(2).mean().item()) * n

        # rmse_log: RMSE in log space — de-emphasises far-range errors
        log_diff = pred.log() - gt.log()
        self._rmse_log_sum += float(log_diff.pow(2).mean().item()) * n

        # silog components: E[δ²] and E[δ] where δ = log(pred) - log(gt)
        # silog = sqrt(E[δ²] - E[δ]²)  (scale-invariant log error)
        self._silog_sq_sum += float(log_diff.pow(2).sum().item())
        self._silog_lin_sum += float(log_diff.sum().item())

        # ── threshold accuracies ─────────────────────────────────────────────
        # max(pred/gt, gt/pred) < 1.25^k — checks if prediction is within
        # a factor of 1.25 of ground truth (scale-invariant quality)
        ratio = torch.maximum(pred / gt, gt / pred)
        self._d1_sum += float((ratio < 1.25).float().mean().item()) * n
        self._d2_sum += float((ratio < 1.25**2).float().mean().item()) * n
        self._d3_sum += float((ratio < 1.25**3).float().mean().item()) * n

        # ── median-scaled abs_rel ─────────────────────────────────────────────
        # Align prediction to GT at median before computing abs_rel.
        # This removes global scale ambiguity (useful for self-supervised models
        # that predict up-to-a-scale depth).
        scale = gt.median() / pred.median().clamp(min=1e-8)
        pred_sc = (pred * scale).clamp(self.min_depth, self.max_depth)
        self._abs_rel_sc_sum += float(((pred_sc - gt).abs() / gt).mean().item()) * n

        self._n += n

    def compute(self) -> dict[str, float]:
        """Return a flat dict of all depth metrics.

        Returns an empty dict if no pixels have been accumulated.
        """
        n = self._n
        if n == 0:
            return {}

        import math

        abs_rel = self._abs_rel_sum / n
        sq_rel = self._sq_rel_sum / n
        rmse = math.sqrt(self._rmse_sum / n)
        rmse_log = math.sqrt(self._rmse_log_sum / n)

        # silog = sqrt( E[δ²] - (E[δ])² )
        silog = math.sqrt(max(self._silog_sq_sum / n - (self._silog_lin_sum / n) ** 2, 0.0))

        d1 = self._d1_sum / n
        d2 = self._d2_sum / n
        d3 = self._d3_sum / n

        abs_rel_scaled = self._abs_rel_sc_sum / n

        return {
            # ↓ lower is better
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "silog": silog,
            "abs_rel_scaled": abs_rel_scaled,
            # ↑ higher is better
            "d1": d1,
            "d2": d2,
            "d3": d3,
        }
