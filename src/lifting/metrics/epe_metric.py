"""End-point error for dense flow and scene-flow fields."""

from __future__ import annotations

from torch import Tensor


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
        """Clear all accumulated statistics."""
        self._total_epe = 0.0
        self._n = 0
        self._below_005 = 0
        self._below_015 = 0

    def update(
        self,
        pred_flow: Tensor,  # [N, 3] or [B, N, 3]
        gt_flow: Tensor,  # same shape
        mask: Tensor | None = None,  # [N] or [B, N] boolean
    ) -> None:
        """Accumulate one batch of predictions and targets."""
        p = pred_flow.float().detach().cpu()
        g = gt_flow.float().detach().cpu()

        epe = (p - g).norm(p=2, dim=-1)  # [N] or [B, N]

        if mask is not None:
            epe = epe[mask.bool()]

        epe = epe.reshape(-1)
        self._total_epe += float(epe.sum())
        self._n += epe.numel()
        self._below_005 += int((epe < 0.05).sum())
        self._below_015 += int((epe < 0.15).sum())

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`."""
        n = max(self._n, 1)
        return {
            "epe_mean": self._total_epe / n,
            "epe_1px": self._below_005 / n,  # fraction within 5 cm
            "epe_3px": self._below_015 / n,  # fraction within 15 cm
        }
