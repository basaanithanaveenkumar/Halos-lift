"""DataLoaders of :class:`SceneBatch` for any scene dataset, optionally via WebDataset shards."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from lifting.data import SceneBatch
from lifting.distributed.session import DistributedSession
from lifting.exceptions import InvalidConfigError


class SceneLoaderBuilder:
    """Builds the training and evaluation loaders of a (possibly distributed) run.

    **Training** (:meth:`train_loader`):

    * default: the map-style dataset with a :class:`~torch.utils.data.DistributedSampler` when
      several ranks train (each rank reads a disjoint slice, reshuffled every epoch via
      ``loader.sampler.set_epoch``), plain shuffling otherwise;
    * ``webdataset.enabled``: the dataset is written once to WebDataset tar shards by rank 0
      (HaleBlocks' :class:`~hale_core.data.webdataset.WebDatasetAdapter`; samples round-trip
      exactly, including ``inf`` sky depth) and every rank streams them. Later runs reuse the
      shards while the ``fingerprint`` matches.

    **Evaluation** (:meth:`eval_loader`): always the map-style dataset, in order, without
    sharding - evaluation runs on rank 0 so every scene is scored exactly once.

    Args:
        session: rank / world size of this process.
        batch_size: scenes per batch *per rank*.
        num_workers: DataLoader worker processes.
        webdataset: a HaleBlocks ``WebDatasetConfig`` (``None`` or ``enabled: false`` = off).
        seed: shuffling seed shared by all ranks.
    """

    def __init__(
        self,
        session: DistributedSession,
        batch_size: int,
        num_workers: int = 0,
        webdataset: Any | None = None,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise InvalidConfigError(f"batch_size must be >= 1, got {batch_size}")
        self.session = session
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.webdataset = webdataset
        self.seed = seed

    @property
    def uses_webdataset(self) -> bool:
        """Whether training data is streamed from WebDataset shards."""
        return self.webdataset is not None and bool(self.webdataset.enabled)

    def train_loader(self, dataset: Dataset, name: str, fingerprint: str) -> DataLoader:
        """Shuffled training batches for this rank.

        Args:
            dataset: map-style dataset of scene dicts (``SyntheticSceneDataset`` or any dataset
                producing the same keys).
            name: shard directory name (unique per split / source).
            fingerprint: identity of the dataset's content; shards are rebuilt when it changes.
        """
        if self.uses_webdataset:
            from hale_core.data.webdataset import WebDatasetAdapter  # lazy: optional extra

            stream = WebDatasetAdapter(self.webdataset).prepare(
                dataset,
                name,
                training=True,
                loader_workers=self.num_workers,
                fingerprint=fingerprint,
            )
            return self._loader(stream, sampler=None, shuffle=False, drop_last=True)
        sampler = self.session.sampler(dataset, shuffle=True, seed=self.seed)
        return self._loader(dataset, sampler=sampler, shuffle=sampler is None, drop_last=True)

    def eval_loader(self, dataset: Dataset) -> DataLoader:
        """Every scene once, in order (run on rank 0)."""
        return self._loader(dataset, sampler=None, shuffle=False, drop_last=False)

    def _loader(self, dataset, *, sampler, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=SceneBatch.collate,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )
