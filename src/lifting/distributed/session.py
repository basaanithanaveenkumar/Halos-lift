"""Process group, device and data sharding for single-process, torchrun and Ray Train runs."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler, IterableDataset


@dataclass(frozen=True)
class DistributedSession:
    """Where this process sits in a (possibly) distributed run, and the helpers that depend on it.

    * **Single process** (``WORLD_SIZE`` unset or 1): rank 0 of 1; :meth:`wrap` and
      :meth:`sampler` are no-ops.
    * **torchrun**: ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` are set; :meth:`start` creates the
      process group (NCCL with CUDA, Gloo otherwise) and :meth:`close` destroys it.
    * **Ray Train**: Ray has already created the process group; :meth:`start` joins it.

    Attributes:
        rank: global rank.
        local_rank: rank on this node (selects the CUDA device).
        world_size: number of processes.
        device: device this rank computes on.
        owns_group: whether :meth:`start` created the process group.
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    owns_group: bool = False

    # ------------------------------------------------------------------ construction
    @classmethod
    def start(cls, device: str | torch.device = "cpu") -> DistributedSession:
        """Join (or create) the process group and choose this rank's device.

        Args:
            device: requested device; with CUDA and several ranks every rank uses
                ``cuda:<local_rank>``.
        """
        owns = False
        if (
            int(os.environ.get("WORLD_SIZE", 1)) > 1
            and dist.is_available()
            and not dist.is_initialized()
        ):
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
            owns = True
        initialised = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if initialised else 0
        world = dist.get_world_size() if initialised else 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        resolved = torch.device(device)
        if resolved.type == "cuda" and world > 1:
            resolved = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(resolved)
        return cls(rank, local_rank, world, resolved, owns)

    # ------------------------------------------------------------------ properties
    @property
    def is_main(self) -> bool:
        """Rank 0 prints, evaluates and writes checkpoints."""
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        """Whether more than one process takes part in this job."""
        return self.world_size > 1

    # ------------------------------------------------------------------ model / data
    def wrap(self, module: nn.Module, find_unused_parameters: bool = True) -> nn.Module:
        """``DistributedDataParallel(module)`` when distributed, else ``module`` itself.

        ``find_unused_parameters`` defaults to ``True`` because several lifters own parameters a
        given configuration never touches (e.g. a pyramid level the lifter does not read); DDP
        would otherwise fail on the first backward pass.
        """
        if not self.distributed:
            return module
        ids = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(
            module, device_ids=ids, find_unused_parameters=find_unused_parameters
        )

    def sampler(self, dataset: Dataset, shuffle: bool, seed: int = 0) -> DistributedSampler | None:
        """A :class:`DistributedSampler` giving this rank a disjoint 1/world_size of a map-style
        dataset, or ``None`` when not distributed or for streams (they split themselves)."""
        if not self.distributed or isinstance(dataset, IterableDataset):
            return None
        return DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle, seed=seed
        )

    # ------------------------------------------------------------------ collectives
    def mean(self, value: float) -> float:
        """Average a scalar over all ranks (every rank must call it)."""
        if not self.distributed:
            return value
        t = torch.tensor([value], dtype=torch.float64, device=self._collective_device())
        dist.all_reduce(t)
        return t.item() / self.world_size

    def barrier(self) -> None:
        """Block until every rank reaches this point (no-op for a single process)."""
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        """Destroy the process group if this session created it (torchrun)."""
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()

    def _collective_device(self) -> torch.device:
        # NCCL only reduces CUDA tensors; Gloo only CPU ones.
        return self.device if dist.get_backend() == "nccl" else torch.device("cpu")
