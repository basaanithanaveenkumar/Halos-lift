"""Matplotlib views of TPV feature planes, label planes and camera images."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor

from lifting.lifters.tpvformer.tpv_planes import TPVPlanes

#: background, vehicle, pedestrian - the synthetic world's classes.
CLASS_COLORS = ("#f2f2f2", "#d62728", "#1f5bd6")
CLASS_NAMES = ("empty", "vehicle", "pedestrian")
#: Same orientation as the README GIFs: BEV with x (forward) up and y (left) to the left,
#: side / front planes with height z pointing up.
PLANE_TITLES = ("xy · top / BEV  (x ↑, y ←)", "xz · side  (x →, z ↑)", "yz · front  (y →, z ↑)")


class PlanePlotter:
    """Draws the three planes of a triplane in one fixed layout::

        +-----------+-----------------+
        |           |  xz  (x →, z ↑) |
        |  xy (BEV) +-----------------+
        |           |  yz  (y →, z ↑) |
        +-----------+-----------------+

    * :meth:`features` - a learned ``TPVPlanes`` as RGB, using one PCA basis shared by the
      three planes (so equal colours mean similar features across planes).
    * :meth:`labels` - class-id planes (ground truth or predictions), one row per entry.
    * :meth:`scalar_planes` - any three scalar maps on one shared colour scale.
    * :meth:`cameras` - the raw camera images of one scene.

    Tensors are stored ``xy (X, Y)``, ``xz (X, Z)``, ``yz (Y, Z)``; :meth:`oriented` turns them
    into the displayed orientation. All figures use a white background.

    Args:
        row_height: figure height in inches per row.
    """

    def __init__(self, row_height: float = 3.0) -> None:
        self.row_height = row_height
        self.class_cmap = ListedColormap(CLASS_COLORS)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def oriented(plane: int, array: np.ndarray) -> np.ndarray:
        """Map a stored ``(H, W[, 3])`` plane to its display orientation."""
        if plane == 0:  # (X, Y) -> x up, y to the left
            return array[::-1, ::-1]
        return np.swapaxes(array, 0, 1)[::-1]  # (A, Z) -> z up, A to the right

    @staticmethod
    def _pca_rgb(planes: Sequence[Tensor]) -> list[np.ndarray]:
        """Project ``(C, H, W)`` planes onto their joint top-3 principal components -> RGB."""
        flat = torch.cat([p.flatten(1).T for p in planes]).double()  # (sum HW, C)
        flat = flat - flat.mean(0, keepdim=True)
        # Joint PCA basis -> colours are comparable across the three planes.
        _, _, v = torch.linalg.svd(flat, full_matrices=False)
        k = min(3, v.shape[0])
        proj = flat @ v[:k].T
        if k < 3:
            proj = torch.cat([proj, proj.new_zeros(proj.shape[0], 3 - k)], 1)
        lo, hi = proj.quantile(0.02, 0), proj.quantile(0.98, 0)
        proj = ((proj - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
        out, start = [], 0
        for p in planes:
            h, w = p.shape[-2:]
            out.append(proj[start : start + h * w].reshape(h, w, 3).float().numpy())
            start += h * w
        return out

    def _figure(self, title: str, rows: int, extra_width: float = 0.0):
        """Figure with ``rows`` x (xy | xz over yz) axes; returns ``fig, [[ax_xy, ax_xz, ax_yz], ...]``."""
        # Width scales with the row height so the equal-aspect panels fill the figure.
        width = 7.6 * min(1.0, self.row_height / 3.0) + extra_width
        height = self.row_height * rows + 0.9
        fig = plt.figure(figsize=(width, height), facecolor="white")
        grid = fig.add_gridspec(rows, 2, width_ratios=(1.0, 1.55), wspace=0.08, hspace=0.35)
        # Fixed margins in inches (title on top, legend at the bottom) whatever the number of rows.
        fig.subplots_adjust(top=1 - 0.55 / height, bottom=0.35 / height, left=0.07, right=0.97)
        axes = []
        for r in range(rows):
            right = grid[r, 1].subgridspec(2, 1, hspace=0.55)
            axes.append(
                [fig.add_subplot(grid[r, 0]), fig.add_subplot(right[0]), fig.add_subplot(right[1])]
            )
        fig.suptitle(title, fontsize=11, fontweight="bold")
        return fig, axes

    @staticmethod
    def _decorate(ax, title: str | None) -> None:
        ax.set_xticks([]), ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=8.5)

    # ------------------------------------------------------------------ public API
    def features(
        self, planes: TPVPlanes, title: str = "feature planes (joint PCA → RGB)", index: int = 0
    ) -> None:
        """Show sample ``index`` of a ``TPVPlanes`` as three RGB images."""
        tensors = [p[index].detach().cpu().float() for p in (planes.xy, planes.xz, planes.yz)]
        fig, axes = self._figure(title, 1)
        for plane, (ax, img) in enumerate(zip(axes[0], self._pca_rgb(tensors), strict=True)):
            ax.imshow(self.oriented(plane, img), interpolation="nearest")
            self._decorate(ax, PLANE_TITLES[plane])
        plt.show()

    def labels(
        self,
        rows: Sequence[tuple[str, Sequence[Tensor]]],
        title: str = "label planes",
        index: int = 0,
    ) -> None:
        """Show one or more rows of ``(xy, xz, yz)`` class-id planes.

        Args:
            rows: ``[(row_name, (xy, xz, yz)), ...]`` with batched ``(B, H, W)`` tensors.
            title: figure title.
            index: which sample of the batch.
        """
        fig, axes = self._figure(title, len(rows))
        for r, (row_name, planes) in enumerate(rows):
            for plane, (ax, tensor) in enumerate(zip(axes[r], planes, strict=True)):
                img = self.oriented(plane, tensor[index].cpu().numpy())
                ax.imshow(
                    img,
                    cmap=self.class_cmap,
                    vmin=0,
                    vmax=len(CLASS_COLORS) - 1,
                    interpolation="nearest",
                )
                self._decorate(ax, PLANE_TITLES[plane] if r == 0 else None)
            axes[r][0].set_ylabel(row_name, fontsize=9.5, fontweight="bold")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=col, ec="#888") for col in CLASS_COLORS]
        fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
        plt.show()

    def scalar_planes(self, planes: Sequence[Tensor], title: str, cmap: str = "viridis") -> None:
        """Three stored-orientation ``(H, W)`` scalar maps on one shared colour scale."""
        arrays = [p.detach().cpu().float().numpy() for p in planes]
        vmin, vmax = min(a.min() for a in arrays), max(a.max() for a in arrays)
        fig, axes = self._figure(title, 1, extra_width=0.8)
        image = None
        for plane, (ax, arr) in enumerate(zip(axes[0], arrays, strict=True)):
            image = ax.imshow(
                self.oriented(plane, arr), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            self._decorate(ax, PLANE_TITLES[plane])
        fig.colorbar(image, ax=[ax for row in axes for ax in row], fraction=0.03, pad=0.02)
        plt.show()

    def cameras(self, images: Tensor, index: int = 0, title: str = "camera images") -> None:
        """``images``: ``(B, N, 3, H, W)`` in ``[0, 1]``."""
        imgs = images[index].cpu()
        n = imgs.shape[0]
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 1.7), facecolor="white")
        for i, ax in enumerate(np.atleast_1d(axes)):
            ax.imshow(imgs[i].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_title(f"cam {i}", fontsize=8.5)
            ax.axis("off")
        fig.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.show()
