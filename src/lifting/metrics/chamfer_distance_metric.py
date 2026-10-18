"""Bidirectional nearest-neighbour (Chamfer) distance between point clouds."""

from __future__ import annotations

from torch import Tensor

from lifting.metrics._point_distances import _pairwise_sq_dist


class ChamferDistanceMetric:
    """Bidirectional Chamfer Distance between predicted and GT point clouds.

    Chamfer Distance (CD) is the standard geometric quality metric for 3D
    reconstruction.  It measures the average nearest-neighbour distance
    *in both directions*:

        CD(P, Q) = mean_{p∈P} min_{q∈Q} ||p−q||²
                 + mean_{q∈Q} min_{p∈P} ||p−q||²

    WHY bidirectional:
        - P→Q term (completeness): penalises predicted points that are far
          from any GT point — catches spurious geometry.
        - Q→P term (accuracy):     penalises GT points with no nearby
          prediction — catches missing geometry / holes.
        Both are necessary: optimising only one term collapses to a trivial
        solution (all predictions at the GT centroid, or vice versa).

    Lower is better.  Units: same as the point cloud (e.g. metres²).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._p_to_q_sum = 0.0  # pred→GT direction
        self._q_to_p_sum = 0.0  # GT→pred direction
        self._n = 0

    def update(self, pred_pts: Tensor, gt_pts: Tensor) -> None:
        """
        Args:
            pred_pts: [N, 3] predicted point cloud.
            gt_pts:   [M, 3] ground-truth point cloud.
        """
        p = pred_pts.float().detach().cpu()
        q = gt_pts.float().detach().cpu()

        dists = _pairwise_sq_dist(p, q)  # [N, M]
        p_to_q = float(dists.min(dim=1).values.mean())  # avg min dist pred→GT
        q_to_p = float(dists.min(dim=0).values.mean())  # avg min dist GT→pred

        self._p_to_q_sum += p_to_q
        self._q_to_p_sum += q_to_p
        self._n += 1

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`."""
        n = max(self._n, 1)
        p2q = self._p_to_q_sum / n
        q2p = self._q_to_p_sum / n
        return {
            "chamfer_dist": p2q + q2p,  # full bidirectional CD
            "chamfer_p_to_q": p2q,  # pred→GT (accuracy)
            "chamfer_q_to_p": q2p,  # GT→pred (completeness)
        }
