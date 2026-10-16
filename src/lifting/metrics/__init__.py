"""
Lifting Evaluation Metrics
===========================

Comprehensive metrics for evaluating the Halos-lift monocular depth and
3D reconstruction pipeline.  All metrics follow the ``reset / update / compute``
lifecycle.

Depth Estimation (monocular)
-----------------------------
DepthMetrics — full Eigen et al. suite: abs_rel, sq_rel, rmse, rmse_log,
               silog, d1/d2/d3, abs_rel_scaled (median-aligned)

3D Geometry Quality
-------------------
ChamferDistanceMetric — bidirectional nearest-neighbour distance
FScoreMetric          — precision/recall/F-score at spatial thresholds
EPEMetric             — end-point error for scene-flow fields

Bird's-Eye-View (BEV)
----------------------
BEVIoUMetric      — semantic/binary BEV occupancy IoU
BEVObjectMetric   — BEV object detection Average Precision (AP)
RangeErrorMetric  — per-range-bin depth abs_rel error

Segmentation (existing)
-----------------------
IoUMetric — confusion-matrix based semantic segmentation mIoU
"""

from lifting.metrics.bev import (
    BEVIoUMetric,
    BEVObjectMetric,
    RangeErrorMetric,
)
from lifting.metrics.depth import DepthMetrics
from lifting.metrics.geometry import (
    ChamferDistanceMetric,
    EPEMetric,
    FScoreMetric,
)
from lifting.metrics.iou_metric import IoUMetric

__all__ = [
    # existing
    "IoUMetric",
    # depth
    "DepthMetrics",
    # geometry
    "ChamferDistanceMetric",
    "FScoreMetric",
    "EPEMetric",
    # bev
    "BEVIoUMetric",
    "BEVObjectMetric",
    "RangeErrorMetric",
]
