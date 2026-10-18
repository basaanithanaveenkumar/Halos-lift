"""Precision, recall and F-score of a predicted point cloud at a distance threshold."""

from __future__ import annotations

from torch import Tensor

from lifting.metrics._point_distances import _pairwise_sq_dist


class FScoreMetric:
    """Precision, Recall, and F-score for 3D reconstruction at a distance threshold.

    F-score is more interpretable than Chamfer Distance because it has a
    clear geometric meaning:

        Precision = fraction of predicted points within τ of any GT point.
        Recall    = fraction of GT points within τ of any predicted point.
        F-score   = harmonic mean of precision and recall.

    WHY F-score complements CD:
        CD is sensitive to outliers (a few very distant points dominate).
        F-score at a practical threshold (e.g. 5 cm) measures whether the
        reconstruction is "usable" regardless of rare outliers.

        Threshold τ = 0.05 m (5 cm) is the standard for indoor scenes
        (3D-R2N2, OccNet benchmarks).  τ = 0.1 m for outdoor / KITTI.

    Args:
        thresholds: distance thresholds τ (same units as point cloud, e.g. metres).
    """

    def __init__(self, thresholds: tuple[float, ...] = (0.05, 0.1)) -> None:
        self.thresholds = thresholds
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._prec: dict[float, float] = {t: 0.0 for t in self.thresholds}
        self._rec: dict[float, float] = {t: 0.0 for t in self.thresholds}
        self._n = 0

    def update(self, pred_pts: Tensor, gt_pts: Tensor) -> None:
        """
        Args:
            pred_pts: [N, 3]
            gt_pts:   [M, 3]
        """
        p = pred_pts.float().detach().cpu()
        q = gt_pts.float().detach().cpu()

        dists = _pairwise_sq_dist(p, q)  # [N, M]
        min_p2q = dists.min(dim=1).values.sqrt()  # [N] — dist pred→nearest GT
        min_q2p = dists.min(dim=0).values.sqrt()  # [M] — dist GT→nearest pred

        for tau in self.thresholds:
            self._prec[tau] += float((min_p2q < tau).float().mean())
            self._rec[tau] += float((min_q2p < tau).float().mean())
        self._n += 1

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`."""
        n = max(self._n, 1)
        out: dict[str, float] = {}
        for tau in self.thresholds:
            p = self._prec[tau] / n
            r = self._rec[tau] / n
            f = 2 * p * r / max(p + r, 1e-8)
            tag = f"@{tau:.2f}"
            out[f"fscore_precision{tag}"] = p
            out[f"fscore_recall{tag}"] = r
            out[f"fscore{tag}"] = f
        return out
