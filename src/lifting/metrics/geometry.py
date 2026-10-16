"""
3D geometry quality metrics for Halos-lift.

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
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor

# ─── ChamferDistanceMetric ───────────────────────────────────────────────────

def _pairwise_sq_dist(a: Tensor, b: Tensor) -> Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
        a: [N, 3]
        b: [M, 3]
    Returns:
        dists: [N, M]
    """
    # ||a - b||² = ||a||² + ||b||² - 2·a·bᵀ
    aa = (a ** 2).sum(1, keepdim=True)   # [N, 1]
    bb = (b ** 2).sum(1, keepdim=True)   # [M, 1]
    return (aa + bb.T - 2.0 * (a @ b.T)).clamp(min=0.0)


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

        dists = _pairwise_sq_dist(p, q)                  # [N, M]
        p_to_q = float(dists.min(dim=1).values.mean())   # avg min dist pred→GT
        q_to_p = float(dists.min(dim=0).values.mean())   # avg min dist GT→pred

        self._p_to_q_sum += p_to_q
        self._q_to_p_sum += q_to_p
        self._n += 1

    def compute(self) -> dict[str, float]:
        n = max(self._n, 1)
        p2q = self._p_to_q_sum / n
        q2p = self._q_to_p_sum / n
        return {
            "chamfer_dist":         p2q + q2p,   # full bidirectional CD
            "chamfer_p_to_q":       p2q,         # pred→GT (accuracy)
            "chamfer_q_to_p":       q2p,         # GT→pred (completeness)
        }


# ─── FScoreMetric ─────────────────────────────────────────────────────────────

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
        self._prec: dict[float, float] = {t: 0.0 for t in self.thresholds}
        self._rec:  dict[float, float] = {t: 0.0 for t in self.thresholds}
        self._n = 0

    def update(self, pred_pts: Tensor, gt_pts: Tensor) -> None:
        """
        Args:
            pred_pts: [N, 3]
            gt_pts:   [M, 3]
        """
        p = pred_pts.float().detach().cpu()
        q = gt_pts.float().detach().cpu()

        dists = _pairwise_sq_dist(p, q)                      # [N, M]
        min_p2q = dists.min(dim=1).values.sqrt()             # [N] — dist pred→nearest GT
        min_q2p = dists.min(dim=0).values.sqrt()             # [M] — dist GT→nearest pred

        for tau in self.thresholds:
            self._prec[tau] += float((min_p2q < tau).float().mean())
            self._rec[tau]  += float((min_q2p < tau).float().mean())
        self._n += 1

    def compute(self) -> dict[str, float]:
        n = max(self._n, 1)
        out: dict[str, float] = {}
        for tau in self.thresholds:
            p = self._prec[tau] / n
            r = self._rec[tau]  / n
            f = 2 * p * r / max(p + r, 1e-8)
            tag = f"@{tau:.2f}"
            out[f"fscore_precision{tag}"] = p
            out[f"fscore_recall{tag}"]    = r
            out[f"fscore{tag}"]           = f
        return out


# ─── EPEMetric ────────────────────────────────────────────────────────────────

class EPEMetric:
    """End-Point Error for dense 3D flow / scene-flow fields.

    EPE is the Euclidean distance between predicted and ground-truth 3D
    displacement vectors, averaged over all valid points:

        EPE = mean ||Δp_pred − Δp_gt||₂

    WHY EPE for scene flow:
        Scene flow (3D optical flow) is used in Halos-lift for dynamic scene
        understanding.  EPE directly measures the error in metres, making it
        easy to interpret (0.1 m = 10 cm per-point displacement error).

        EPE < 0.05 m is generally considered good for dynamic scenes.

    Also reports:
        epe_1px — fraction of points with EPE < 0.05 m (equivalent to 1-pixel
                  accuracy at typical depth resolution).
        epe_3px — fraction of points with EPE < 0.15 m.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total_epe = 0.0
        self._n = 0
        self._below_005 = 0
        self._below_015 = 0

    def update(
        self,
        pred_flow: Tensor,              # [N, 3] or [B, N, 3]
        gt_flow:   Tensor,              # same shape
        mask:      Optional[Tensor] = None,  # [N] or [B, N] boolean
    ) -> None:
        p = pred_flow.float().detach().cpu()
        g = gt_flow.float().detach().cpu()

        epe = (p - g).norm(p=2, dim=-1)  # [N] or [B, N]

        if mask is not None:
            epe = epe[mask.bool()]

        epe = epe.reshape(-1)
        self._total_epe  += float(epe.sum())
        self._n          += epe.numel()
        self._below_005  += int((epe < 0.05).sum())
        self._below_015  += int((epe < 0.15).sum())

    def compute(self) -> dict[str, float]:
        n = max(self._n, 1)
        return {
            "epe_mean":  self._total_epe / n,
            "epe_1px":   self._below_005 / n,   # fraction within 5 cm
            "epe_3px":   self._below_015 / n,   # fraction within 15 cm
        }
