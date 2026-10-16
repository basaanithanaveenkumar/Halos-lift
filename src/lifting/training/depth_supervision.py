"""Explicit depth supervision for depth-based lifters (BEVDepth-style), as a task decorator.

Depth-based lifters (``depth_warp``, ``lift_splat`` and their triplane versions) predict a per-pixel
categorical depth distribution with a :class:`~lifting.lifters.depth_head.DepthHead`. Trained only
through a BEV / occupancy loss, that distribution barely moves away from uniform in short runs
(see ``notebooks/triplane/02_depth_warp_triplane.ipynb``), so the frustum spreads features along
the whole ray like Simple-BEV. BEVDepth (Li et al., AAAI 2023, arXiv:2206.10092) fixes this with a
cross-entropy between the predicted bins and the LiDAR depth; here the rendered ``batch.depth``
plays the LiDAR's role.

:class:`DepthSupervisedTask` wraps any :class:`Task` and adds that loss **without touching the
model or the lifters**: forward hooks on every ``DepthHead`` capture its probabilities during the
wrapped task's forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lifting.data import SceneBatch
from lifting.exceptions import InvalidConfigError
from lifting.geometry import DepthBins
from lifting.lifters.depth_head import DepthHead
from lifting.training.task import Task

#: Key under which :meth:`DepthSupervisedTask.forward` stores the captured depth predictions.
DEPTH_PREDICTIONS = "depth_predictions"


def nearest_surface_depth(depth: Tensor, size: tuple[int, int]) -> Tensor:
    """``(B*N, h, w)`` metric depth of the nearest surface inside each feature pixel's patch.

    Min-pooling (as in BEVDepth) when the image is an integer multiple of ``size``, nearest
    sampling otherwise. Pixels with no hit keep ``inf``.
    """
    h, w = size
    big, wide = depth.shape[-2:]
    if big % h == 0 and wide % w == 0:
        return -F.max_pool2d(-depth[:, None], kernel_size=(big // h, wide // w))[:, 0]
    return F.interpolate(depth[:, None], size=(h, w), mode="nearest")[:, 0]


def depth_targets(depth: Tensor, size: tuple[int, int], bins: DepthBins) -> tuple[Tensor, Tensor]:
    """Ground-truth bin index and validity mask at feature resolution.

    Args:
        depth: ``(B*N, H, W)`` metric depth (``inf`` where a ray hits nothing).
        size: ``(h, w)`` resolution of the depth head's output.
        bins: discretisation used by the lifter.

    Returns:
        ``(B*N, h, w)`` long bin indices and a boolean mask of pixels whose depth lies inside
        ``bins``. Each feature pixel takes the **nearest** surface of the image patch it covers
        (:func:`nearest_surface_depth`): the closest surface is what occludes the rest of it.
    """
    pooled = nearest_surface_depth(depth, size)
    valid = bins.contains(pooled)
    index = ((pooled - bins.min_depth) / bins.step).floor().clamp(0, bins.num_bins - 1)
    index = torch.where(valid, index, torch.zeros_like(index))
    return index.long(), valid


class DepthSupervisedTask(Task):
    """Decorates a task with a depth cross-entropy on every ``DepthHead`` of the model.

    ``loss = task.loss + weight * mean_over_heads( CE(depth_probs, bin(batch.depth)) )``; only
    pixels whose ground-truth depth lies inside the lifter's :class:`DepthBins` count.
    Predictions and targets are those of the wrapped task, so evaluation is unchanged.

    Example::

        task = DepthSupervisedTask(OccupancyTask(3, aux_weight=0.5), weight=1.0)
        trainer = Trainer(model, task)

    Args:
        task: the task to decorate (BEV segmentation, occupancy, ...).
        weight: weight of the depth loss.
        default_bins: bins for a ``DepthHead`` whose owning module has no ``depth_bins``
            attribute (lifters built from the registry always have one).
    """

    def __init__(
        self, task: Task, weight: float = 1.0, default_bins: DepthBins | None = None
    ) -> None:
        if weight < 0:
            raise InvalidConfigError(f"weight must be >= 0, got {weight}")
        self.task = task
        self.weight = weight
        self.default_bins = default_bins or DepthBins()
        self.num_classes = task.num_classes

    # ------------------------------------------------------------------ Task API
    def forward(self, model: nn.Module, batch: SceneBatch) -> dict:
        """Wrapped forward; adds ``outputs[DEPTH_PREDICTIONS] = [(probs, bins), ...]``."""
        captured: list[tuple[Tensor, DepthBins]] = []
        handles = [
            head.register_forward_hook(
                lambda _m, _inp, out, bins=bins: captured.append((out[0], bins))
            )
            for head, bins in self._depth_heads(model)
        ]
        try:
            outputs = self.task.forward(model, batch)
        finally:  # never leave hooks behind, even if the forward pass fails
            for handle in handles:
                handle.remove()
        if not captured:
            raise InvalidConfigError(
                "DepthSupervisedTask needs a model with a DepthHead (depth_warp, lift_splat ...)"
            )
        return {**outputs, DEPTH_PREDICTIONS: captured}

    def loss(self, outputs: dict, batch: SceneBatch) -> Tensor:
        base = self.task.loss(outputs, batch)
        if self.weight == 0:
            return base
        return base + self.weight * self.depth_loss(outputs, batch)

    def predict(self, outputs: dict) -> Tensor:
        return self.task.predict(outputs)

    def target(self, batch: SceneBatch) -> Tensor:
        return self.task.target(batch)

    # ------------------------------------------------------------------ depth loss
    def depth_loss(self, outputs: dict, batch: SceneBatch) -> Tensor:
        """Mean over captured heads of the masked negative log-likelihood of the true bin."""
        depth = batch.depth.flatten(0, 1)  # (B*N, H, W)
        losses = []
        for probs, bins in outputs[DEPTH_PREDICTIONS]:
            index, valid = depth_targets(depth.to(probs.device), probs.shape[-2:], bins)
            nll = -probs.clamp_min(1e-9).log().gather(1, index[:, None])[:, 0]  # (B*N, h, w)
            losses.append((nll * valid).sum() / valid.sum().clamp_min(1))
        return torch.stack(losses).mean()

    @staticmethod
    def entropy_and_correlation(probs: Tensor, depth: Tensor, bins: DepthBins) -> tuple[float, float]:
        """Diagnostics: mean entropy of the distributions and corr(E[d], true depth).

        A uniform distribution has entropy ``ln(num_bins)``; a well-trained head has low entropy
        and a correlation near 1.
        """
        true = nearest_surface_depth(depth, probs.shape[-2:])
        valid = bins.contains(true)
        expected = (probs * bins.centers(probs.device)[None, :, None, None]).sum(1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(1).mean().item()
        if valid.sum() < 2:
            return entropy, math.nan
        corr = torch.corrcoef(torch.stack([expected[valid], true[valid]]))[0, 1].item()
        return entropy, corr

    # ------------------------------------------------------------------ helpers
    def _depth_heads(self, model: nn.Module) -> list[tuple[DepthHead, DepthBins]]:
        """Every ``DepthHead`` in ``model`` with the bins of its closest owner that has them."""
        modules = dict(model.named_modules())
        heads = []
        for name, module in modules.items():
            if not isinstance(module, DepthHead):
                continue
            bins = self.default_bins
            parts = name.split(".")
            for i in range(len(parts) - 1, -1, -1):  # walk up: lifter.core.depth_head -> lifter.core
                owner = modules.get(".".join(parts[:i]))
                if owner is not None and isinstance(getattr(owner, "depth_bins", None), DepthBins):
                    bins = owner.depth_bins
                    break
            heads.append((module, bins))
        return heads

