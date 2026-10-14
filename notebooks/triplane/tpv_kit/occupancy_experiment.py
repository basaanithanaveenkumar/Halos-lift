"""Train any triplane lifter end to end on 3D occupancy and probe its use of geometry."""

from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import torch

from lifting.training import DepthSupervisedTask, OccupancyTask, Trainer
from lifting.triplane import TriplaneOccupancyModel, build_triplane_model
from lifting.verification import ExtrinsicCorruption
from tpv_kit.experiment_result import ExperimentResult
from tpv_kit.triplane_scene import TriplaneScene


class OccupancyExperiment:
    """``encoder -> triplane lifter -> TPVAggregator`` trained with ``OccupancyTask``.

    The model comes from :func:`lifting.triplane.build_triplane_model`, the same factory a user
    would call, so the notebooks exercise the public API only. Per-plane auxiliary heads are
    enabled (``aux_weight``) so each of the three planes is supervised directly.

    Args:
        scene: data and grid.
        name: triplane lifter name in ``TRIPLANE_LIFTERS``.
        channels: encoder / plane width.
        lr: AdamW learning rate.
        aux_weight: weight of the per-plane loss.
        depth_weight: weight of an explicit depth cross-entropy (``DepthSupervisedTask``) for
            lifters with a ``DepthHead``; ``0`` trains through the occupancy loss only.
        seed: torch seed for weight initialisation.
        **lifter_kwargs: forwarded to the lifter constructor (e.g. ``num_layers=1``).
    """

    def __init__(
        self,
        scene: TriplaneScene,
        name: str,
        channels: int = 32,
        lr: float = 3e-3,
        aux_weight: float = 0.5,
        depth_weight: float = 0.0,
        seed: int = 0,
        **lifter_kwargs: Any,
    ) -> None:
        torch.manual_seed(seed)
        self.scene = scene
        self.name = name
        self.model: TriplaneOccupancyModel = build_triplane_model(
            name,
            scene.grid,
            num_classes=3,
            image_size=scene.image_size,
            num_cameras=scene.num_cameras,
            channels=channels,
            aux_heads=True,
            **lifter_kwargs,
        )
        self.task = OccupancyTask(3, aux_weight=aux_weight)
        if depth_weight > 0:
            self.task = DepthSupervisedTask(self.task, weight=depth_weight)
        self.trainer = Trainer(self.model, self.task, lr=lr)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def miou(self, corrupted: bool = False) -> float:
        """Foreground mIoU on the held-out scenes (optionally with 90-degree-wrong extrinsics)."""
        transform = ExtrinsicCorruption(90.0) if corrupted else None
        return self.trainer.evaluate(self.scene.test_batches, transform).mean_foreground()

    def run(self, steps: int = 150) -> ExperimentResult:
        """Train for ``steps`` and measure before / after / corrupted mIoU."""
        before = self.miou()
        start = time.time()
        history = self.trainer.fit(self.scene.train_batches, steps=steps)
        seconds = time.time() - start
        return ExperimentResult(
            name=self.name,
            params=self.num_params,
            seconds=seconds,
            miou_before=before,
            miou_after=self.miou(),
            miou_corrupted=self.miou(corrupted=True),
            losses=list(history.losses),
        )

    @torch.no_grad()
    def predict(self, corrupted: bool = False) -> dict:
        """Model outputs on :attr:`TriplaneScene.sample` (``planes``, ``voxel_logits``, ...)."""
        self.model.eval()
        batch = self.scene.sample
        if corrupted:
            batch = ExtrinsicCorruption(90.0)(batch)
        return self.model(batch.images, batch.cameras)

    @staticmethod
    def plot_losses(
        results: list[ExperimentResult], title: str = "training loss", window: int = 10
    ) -> None:
        """Smoothed loss curves of one or more runs on one axis."""
        fig, ax = plt.subplots(figsize=(7, 3), facecolor="white")
        for res in results:
            losses = torch.tensor(res.losses)
            kernel = torch.ones(window) / window
            smooth = torch.nn.functional.conv1d(losses.view(1, 1, -1), kernel.view(1, 1, -1)).view(
                -1
            )
            ax.plot(range(window - 1, len(res.losses)), smooth.numpy(), label=res.name, lw=1.6)
        ax.set_xlabel("step"), ax.set_ylabel("loss"), ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
        plt.tight_layout()
        plt.show()
