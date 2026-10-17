"""Intersection-over-Union between predicted and ground-truth BEV occupancy grids."""

from __future__ import annotations

import torch
from torch import Tensor


class BEVIoUMetric:
    """IoU between predicted and GT BEV occupancy grids.

    WHY BEV IoU:
        Projecting depth into a top-down grid is the standard interface
        between perception and planning.  BEV IoU measures whether the
        occupancy map (free vs. occupied cells) is correct, which directly
        predicts downstream planning quality.

        For semantic BEV grids (e.g. road / sidewalk / dynamic objects),
        per-class mIoU reveals which categories the lifting model handles well.

    Binary mode (num_classes=1):
        IoU = TP / (TP + FP + FN)  where cells are either occupied (1) or free (0).

    Multi-class mode (num_classes > 1):
        Mean IoU over all classes, using a confusion matrix — consistent with
        the existing ``IoUMetric`` in Halos-lift/src/lifting/metrics/iou_metric.py.

    Args:
        num_classes: number of semantic classes (use 1 for binary occupancy).
        ignore_index: class index to exclude from mIoU (e.g. unknown/void).
    """

    def __init__(self, num_classes: int = 1, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        # Confusion matrix [num_classes, num_classes]
        self._conf = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    def update(
        self,
        pred:   Tensor,  # [B, H, W] long — predicted class per BEV cell
        target: Tensor,  # [B, H, W] long — GT class per BEV cell
    ) -> None:
        p = pred.long().detach().cpu().reshape(-1)
        t = target.long().detach().cpu().reshape(-1)

        # Mask out ignore_index and out-of-range values
        valid = (t != self.ignore_index) & (t >= 0) & (t < self.num_classes) & \
                (p >= 0) & (p < self.num_classes)
        p, t = p[valid], t[valid]

        # Accumulate into confusion matrix using linear index trick
        idx = t * self.num_classes + p
        count = torch.bincount(idx, minlength=self.num_classes ** 2)
        self._conf += count.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        # Per-class IoU = TP / (TP + FP + FN)
        #               = diag / (row_sum + col_sum - diag)
        tp   = self._conf.diagonal().float()
        fp   = (self._conf.sum(0) - tp).float()   # col_sum - TP
        fn   = (self._conf.sum(1) - tp).float()   # row_sum - TP
        iou  = tp / (tp + fp + fn).clamp(min=1)   # avoid /0 for absent classes

        # Only average over classes that appear in GT
        present = (self._conf.sum(1) > 0)          # [num_classes] bool
        miou = float(iou[present].mean()) if present.any() else 0.0

        out: dict[str, float] = {"bev_miou": miou}
        if self.num_classes > 1:
            for c in range(self.num_classes):
                out[f"bev_iou_class_{c}"] = float(iou[c])
        else:
            out["bev_iou"] = float(iou[0]) if present[0] else 0.0
        return out
