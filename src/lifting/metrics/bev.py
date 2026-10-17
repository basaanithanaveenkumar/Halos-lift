"""Bird's-Eye-View (BEV) evaluation metrics for Halos-lift.

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

Classes moved to their own modules (one public class per file) are re-exported here so
existing ``from lifting.metrics.bev import ...`` statements keep working."""

from __future__ import annotations

from lifting.metrics.bev_iou_metric import BEVIoUMetric
from lifting.metrics.bev_object_metric import BEVObjectMetric
from lifting.metrics.range_error_metric import RangeErrorMetric

__all__ = ["BEVIoUMetric", "BEVObjectMetric", "RangeErrorMetric"]
