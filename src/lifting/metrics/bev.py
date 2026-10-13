"""
Bird's-Eye-View (BEV) evaluation metrics for Halos-lift.

Halos-lift projects monocular depth estimates into a top-down BEV grid for
downstream tasks like occupancy mapping, object detection, and path planning.
These metrics evaluate the quality of the BEV representation.

Metrics
-------
BEVIoUMetric      — Intersection-over-Union between predicted and GT BEV
                    occupancy grids (semantic or binary).
BEVObjectMetric   — Average Precision for BEV object detection
                    (axis-aligned boxes in the XZ plane).
RangeErrorMetric  — Per-range-bin depth accuracy; reveals whether errors
                    are concentrated in near vs. far regions.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor


# ─── BEVIoUMetric ─────────────────────────────────────────────────────────────

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


# ─── BEVObjectMetric ──────────────────────────────────────────────────────────

def _box_iou_2d(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute IoU between two sets of axis-aligned 2D boxes.

    Args:
        boxes_a: [N, 4] in [x_min, z_min, x_max, z_max] format.
        boxes_b: [M, 4] in [x_min, z_min, x_max, z_max] format.
    Returns:
        iou: [N, M]
    """
    # Intersection
    x_min = torch.maximum(boxes_a[:, 0:1], boxes_b[:, 0].unsqueeze(0))  # [N, M]
    z_min = torch.maximum(boxes_a[:, 1:2], boxes_b[:, 1].unsqueeze(0))
    x_max = torch.minimum(boxes_a[:, 2:3], boxes_b[:, 2].unsqueeze(0))
    z_max = torch.minimum(boxes_a[:, 3:4], boxes_b[:, 3].unsqueeze(0))

    inter_w = (x_max - x_min).clamp(min=0)
    inter_h = (z_max - z_min).clamp(min=0)
    inter   = inter_w * inter_h  # [N, M]

    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])).unsqueeze(1)
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0)
    union  = area_a + area_b - inter

    return inter / union.clamp(min=1e-8)


class BEVObjectMetric:
    """Average Precision (AP) for BEV 2D object detection.

    Evaluates axis-aligned bounding box detection in the BEV XZ plane
    (X = lateral, Z = depth / range from camera).

    WHY AP for BEV detection:
        AP integrates precision-recall over all confidence thresholds,
        giving a single score that captures both detection sensitivity
        and specificity.  AP@0.5 (IoU threshold 0.5) is the standard
        benchmark for BEV detection (nuScenes, KITTI BEV).

    Implementation:
        - Per-class AP computed via 11-point interpolation
          (consistent with PASCAL VOC 2010 evaluation).
        - IoU threshold configurable (default 0.5).

    Args:
        num_classes: number of object categories.
        iou_threshold: minimum IoU to count as a true positive.
    """

    def __init__(self, num_classes: int = 1, iou_threshold: float = 0.5) -> None:
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        # Per class: list of (score, is_tp) tuples — collected across the dataset
        self._all_scores: List[List[float]] = [[] for _ in range(self.num_classes)]
        self._all_tp:     List[List[int]]   = [[] for _ in range(self.num_classes)]
        self._n_gt:       List[int]         = [0]  * self.num_classes

    def update(
        self,
        pred_boxes:  Tensor,            # [N_pred, 4] x_min z_min x_max z_max
        pred_scores: Tensor,            # [N_pred]
        pred_labels: Tensor,            # [N_pred] long
        gt_boxes:    Tensor,            # [N_gt, 4]
        gt_labels:   Tensor,            # [N_gt] long
    ) -> None:
        pred_boxes   = pred_boxes.float().detach().cpu()
        pred_scores  = pred_scores.float().detach().cpu()
        pred_labels  = pred_labels.long().detach().cpu()
        gt_boxes     = gt_boxes.float().detach().cpu()
        gt_labels    = gt_labels.long().detach().cpu()

        for c in range(self.num_classes):
            p_mask = pred_labels == c
            g_mask = gt_labels   == c

            p_boxes  = pred_boxes[p_mask]
            p_scores = pred_scores[p_mask]
            g_boxes  = gt_boxes[g_mask]

            n_gt = g_boxes.shape[0]
            self._n_gt[c] += n_gt

            if p_boxes.shape[0] == 0:
                continue

            if n_gt == 0:
                # All detections are false positives
                for s in p_scores.tolist():
                    self._all_scores[c].append(s)
                    self._all_tp[c].append(0)
                continue

            # IoU matrix [N_pred, N_gt]
            iou = _box_iou_2d(p_boxes, g_boxes)
            # Greedy matching: highest-score predictions matched first
            matched_gt = set()
            order = p_scores.argsort(descending=True)
            for pi in order.tolist():
                best_iou, best_gt = float(iou[pi].max()), int(iou[pi].argmax())
                if best_iou >= self.iou_threshold and best_gt not in matched_gt:
                    tp = 1
                    matched_gt.add(best_gt)
                else:
                    tp = 0
                self._all_scores[c].append(float(p_scores[pi]))
                self._all_tp[c].append(tp)

    def compute(self) -> dict[str, float]:
        aps: list[float] = []
        for c in range(self.num_classes):
            if self._n_gt[c] == 0 or not self._all_scores[c]:
                aps.append(0.0)
                continue

            scores = torch.tensor(self._all_scores[c])
            tps    = torch.tensor(self._all_tp[c], dtype=torch.float32)
            order  = scores.argsort(descending=True)
            tps    = tps[order]
            fps    = 1.0 - tps

            tp_cum = tps.cumsum(0)
            fp_cum = fps.cumsum(0)

            precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-8)
            recall    = tp_cum / max(self._n_gt[c], 1)

            # 11-point interpolation
            ap = 0.0
            for r_thr in [r * 0.1 for r in range(11)]:
                mask = recall >= r_thr
                if mask.any():
                    ap += float(precision[mask].max()) / 11.0
            aps.append(ap)

        map_score = sum(aps) / max(len(aps), 1)
        out: dict[str, float] = {f"bev_map@{self.iou_threshold:.2f}": map_score}
        if self.num_classes > 1:
            for c, ap in enumerate(aps):
                out[f"bev_ap_class_{c}@{self.iou_threshold:.2f}"] = ap
        return out


# ─── RangeErrorMetric ─────────────────────────────────────────────────────────

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
        self._abs_rel_sum = [0.0] * self.n_bins
        self._n_per_bin   = [0]   * self.n_bins

    def update(
        self,
        pred_depth: Tensor,              # any shape
        gt_depth:   Tensor,              # same shape
        mask:       Optional[Tensor] = None,
    ) -> None:
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
            self._n_per_bin[i]   += n

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in range(self.n_bins):
            lo = self.range_bins[i]
            hi = self.range_bins[i + 1]
            n  = self._n_per_bin[i]
            key = f"range_abs_rel_{lo:.0f}_{hi:.0f}m"
            out[key] = self._abs_rel_sum[i] / max(n, 1)
        return out
