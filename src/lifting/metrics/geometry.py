"""3D geometry quality metrics for Halos-lift.

These metrics evaluate the quality of 3D reconstructions or point clouds
predicted by the lifting pipeline (monocular depth → 3D).

Metrics
-------
ChamferDistanceMetric — bidirectional nearest-neighbour distance between
                        predicted and ground-truth point clouds.
FScoreMetric          — precision/recall/F-score at a spatial threshold.
EPEMetric             — end-point error for dense flow/scene-flow fields.

Design note
-----------
All metrics work on CPU tensors and batch over individual point clouds
rather than assuming a fixed cloud size — reconstruction cardinality varies
by scene complexity and depth resolution.

Classes moved to their own modules (one public class per file) are re-exported here so
existing ``from lifting.metrics.geometry import ...`` statements keep working."""

from __future__ import annotations

from lifting.metrics.chamfer_distance_metric import ChamferDistanceMetric
from lifting.metrics.epe_metric import EPEMetric
from lifting.metrics.f_score_metric import FScoreMetric

__all__ = ["ChamferDistanceMetric", "EPEMetric", "FScoreMetric"]
