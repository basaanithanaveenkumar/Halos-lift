"""3D semantic occupancy task (TPVFormer)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lifting.data import SceneBatch
from lifting.training.task import Task


class OccupancyTask(Task):
    """Weighted cross-entropy on ``(B, K, X, Y, Z)`` voxel logits against ``batch.occupancy``.

    With ``aux_weight > 0`` the model's per-plane ``aux_logits`` (``build_tpvformer(...,
    aux_heads=True)``) are supervised too, against the occupancy collapsed onto each plane
    (highest class along the collapsed axis, so any occupied voxel marks its column), and the
    mean plane loss is added with that weight. The default ``0`` keeps the plain voxel loss.
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_weights: Sequence[float] | None = None,
        aux_weight: float = 0.0,
    ) -> None:
        self.num_classes = num_classes
        weights = class_weights or [1.0] + [8.0] * (num_classes - 1)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        self.aux_weight = aux_weight

    def forward(self, model: nn.Module, batch: SceneBatch) -> dict:
        return model(batch.images, batch.cameras)

    def loss(self, outputs: dict, batch: SceneBatch) -> Tensor:
        logits = outputs["voxel_logits"]
        weight = self.class_weights.to(logits.device)
        loss = F.cross_entropy(logits, batch.occupancy, weight=weight)
        if self.aux_weight > 0:
            loss = loss + self.aux_weight * self.plane_loss(outputs, batch)
        return loss

    def plane_loss(self, outputs: dict, batch: SceneBatch) -> Tensor:
        """Mean cross-entropy of the ``xy`` / ``xz`` / ``yz`` plane logits."""
        if "aux_logits" not in outputs:
            raise ValueError("aux_weight > 0 needs a model built with aux_heads=True")
        occupancy = batch.occupancy  # (B, X, Y, Z)
        targets = (occupancy.amax(3), occupancy.amax(2), occupancy.amax(1))
        weight = self.class_weights.to(occupancy.device)
        losses = [
            F.cross_entropy(logits, target, weight=weight)
            for logits, target in zip(outputs["aux_logits"], targets, strict=True)
        ]
        return torch.stack(losses).mean()

    def predict(self, outputs: dict) -> Tensor:
        return outputs["voxel_logits"].argmax(1)

    def target(self, batch: SceneBatch) -> Tensor:
        return batch.occupancy
