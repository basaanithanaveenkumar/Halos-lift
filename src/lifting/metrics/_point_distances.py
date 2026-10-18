"""Private helpers shared by the geometry modules."""

from __future__ import annotations

from torch import Tensor


def _pairwise_sq_dist(a: Tensor, b: Tensor) -> Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
        a: [N, 3]
        b: [M, 3]
    Returns:
        dists: [N, M]
    """
    # ||a - b||² = ||a||² + ||b||² - 2·a·bᵀ
    aa = (a**2).sum(1, keepdim=True)  # [N, 1]
    bb = (b**2).sum(1, keepdim=True)  # [M, 1]
    return (aa + bb.T - 2.0 * (a @ b.T)).clamp(min=0.0)
