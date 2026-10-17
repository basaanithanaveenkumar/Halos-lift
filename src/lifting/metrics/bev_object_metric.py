"""Average precision for BEV object detection with axis-aligned boxes."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor


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
