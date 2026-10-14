"""Distributed training: torchrun / Ray Train sessions, sharded and WebDataset scene loaders.

* :class:`DistributedSession` - rank, device, DDP wrapping and data sharding of this process.
* :class:`SceneLoaderBuilder` - training / evaluation loaders of :class:`~lifting.data.SceneBatch`,
  optionally streamed from WebDataset shards (``--webdataset``).
* :func:`launch_on_ray` - run a training function on Ray Train workers (``--ray``).

Ray and WebDataset come from HaleBlocks and are optional: ``pip install 'halos-lift[distributed]'``.
Nothing here imports them until the corresponding feature is used.
"""

from lifting.distributed.ray_launch import launch_on_ray, resolve_entry
from lifting.distributed.scene_loader import SceneLoaderBuilder
from lifting.distributed.session import DistributedSession

__all__ = ["DistributedSession", "SceneLoaderBuilder", "launch_on_ray", "resolve_entry"]
