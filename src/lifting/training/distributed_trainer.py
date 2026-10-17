"""A :class:`Trainer` that runs data-parallel (DDP) under torchrun or Ray, with checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from lifting.data import SceneBatch
from lifting.distributed.session import DistributedSession
from lifting.exceptions import EmptyDataError
from lifting.metrics import IoUMetric
from lifting.training.task import Task
from lifting.training.trainer import Trainer
from lifting.training.training_history import TrainingHistory

#: ``on_epoch_end(epoch, global_step, mean_loss_over_ranks)``
EpochCallback = Callable[[int, int, float], None]

CHECKPOINT_PATTERN = "lift_step{step:07d}.pt"


class DistributedTrainer(Trainer):
    """:class:`Trainer` + DistributedDataParallel + epoch loop over a sharded loader + checkpoints.

    The optimisation step is the parent's (:meth:`Trainer.train_step`: task forward, loss,
    clipping, AdamW); only the model it runs is the DDP wrapper, so gradients are all-reduced
    across ranks. Single-process runs use the same class with ``world_size == 1`` (no wrapper).

    * :meth:`fit_loader` - ``steps`` optimiser steps over epochs of a DataLoader, calling
      ``sampler.set_epoch`` every epoch and a callback with the loss averaged over ranks.
    * :meth:`evaluate` / :meth:`predict` - run the **unwrapped** module: evaluation happens on
      rank 0 only, and a DDP forward would wait for the other ranks.
    * :meth:`save_checkpoint` / :meth:`load_checkpoint` - model (unwrapped, no ``module.``
      prefix), optimiser, scheduler and step, written atomically by rank 0.

    Args:
        model: network to optimise.
        task: loss / prediction strategy (any :class:`Task`, e.g. ``DepthSupervisedTask``).
        session: this process's place in the run (device, rank, DDP wrapping).
        lr, weight_decay, grad_clip: as in :class:`Trainer`.
        find_unused_parameters: forwarded to DDP (see :meth:`DistributedSession.wrap`).
    """

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        session: DistributedSession,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        grad_clip: float | None = 5.0,
        find_unused_parameters: bool = True,
    ) -> None:
        super().__init__(model, task, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
                         device=session.device)
        self.session = session
        self.module = self.model  # the plain network: evaluation, checkpoints
        self.model = session.wrap(self.module, find_unused_parameters=find_unused_parameters)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.global_step = 0
        self.epoch = 0

    # ------------------------------------------------------------------ training
    def fit_loader(
        self,
        loader: DataLoader,
        steps: int,
        on_epoch_end: EpochCallback | None = None,
    ) -> TrainingHistory:
        """Train until ``global_step == steps`` (resumes from :meth:`load_checkpoint`'s step).

        Every rank must call this with loaders of equal length per epoch (``DistributedSampler``
        with ``drop_last`` and resampled WebDataset streams both guarantee it), so all ranks
        stop at the same step.
        """
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=steps)
        while self.global_step < steps:
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)  # a different shuffle every epoch, same on all ranks
            losses: list[float] = []
            for batch in loader:
                losses.append(self.train_step(batch))
                self.scheduler.step()
                self.global_step += 1
                if self.global_step >= steps:
                    break
            if not losses:
                raise EmptyDataError("the training loader produced no batches")
            self.epoch += 1
            mean_loss = self.session.mean(sum(losses) / len(losses))
            if on_epoch_end is not None:
                on_epoch_end(self.epoch, self.global_step, mean_loss)
        return self.history

    # ------------------------------------------------------------------ evaluation (rank 0)
    @torch.no_grad()
    def evaluate(
        self,
        batches: Iterable[SceneBatch],
        transform: Callable[[SceneBatch], SceneBatch] | None = None,
    ) -> IoUMetric:
        wrapped, self.model = self.model, self.module
        try:
            return super().evaluate(batches, transform)
        finally:
            self.model = wrapped

    @torch.no_grad()
    def predict(self, batch: SceneBatch) -> torch.Tensor:
        wrapped, self.model = self.model, self.module
        try:
            return super().predict(batch)
        finally:
            self.model = wrapped

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, directory: str | Path, keep_last: int = 3) -> Path | None:
        """Rank 0 writes ``lift_step<step>.pt`` (atomically) and keeps the newest ``keep_last``.

        Returns the path on rank 0 and ``None`` on the other ranks.
        """
        if not self.session.is_main:
            return None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / CHECKPOINT_PATTERN.format(step=self.global_step)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "losses": list(self.history.losses),
            },
            tmp,
        )
        tmp.replace(path)  # a crash never leaves a truncated checkpoint behind
        if keep_last > 0:
            for old in self.checkpoints(directory)[:-keep_last]:
                old.unlink()
        return path

    def load_checkpoint(self, path: str | Path, steps: int) -> None:
        """Restore model, optimiser, scheduler (cosine over ``steps``) and counters."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=steps)
        if state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.history.losses[:] = list(state["losses"])

    @staticmethod
    def checkpoints(directory: str | Path) -> list[Path]:
        """Checkpoints in ``directory``, oldest first."""
        return sorted(Path(directory).glob("lift_step*.pt"))
