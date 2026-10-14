"""Run a training function on Ray Train workers (HaleBlocks' ``RayLauncher``)."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from typing import Any


def resolve_entry(entry: str) -> Callable[[argparse.Namespace], Any]:
    """``"package.module:function"`` -> the function.

    Workers resolve the training function from this reference instead of receiving a pickled
    copy, so nothing from the driver process (loggers, open files, CUDA state) is serialised.
    """
    module_name, sep, name = entry.rpartition(":")
    if not sep or not module_name or not name:
        raise ValueError(f"entry must look like 'package.module:function', got {entry!r}")
    return getattr(importlib.import_module(module_name), name)


def _ray_worker(loop_config: dict[str, Any]) -> None:
    """Executed in every Ray worker: the normal training function (DDP via DistributedSession)."""
    resolve_entry(loop_config["entry"])(loop_config["args"])


def launch_on_ray(entry: str, args: argparse.Namespace, ray_config: Any) -> Any:
    """Run ``entry(args)`` on ``ray_config.num_workers`` Ray Train workers.

    Args:
        entry: importable ``"package.module:function"`` taking the parsed ``args``.
        args: plain ``Namespace`` shipped to every worker.
        ray_config: HaleBlocks ``RayConfig`` (cluster address, workers, GPUs, storage ...).

    Returns:
        HaleBlocks' ``RayRunResult``: last reported metrics and checkpoint directory. Inside a
        worker, :meth:`DistributedSession.start` joins the process group Ray created.
    """
    from hale_core.distributed import RayLauncher  # lazy: optional extra

    resolve_entry(entry)  # fail fast on the driver if the reference is wrong
    return RayLauncher(ray_config).launch(_ray_worker, {"entry": entry, "args": args})
