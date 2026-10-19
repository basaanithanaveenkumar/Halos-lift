"""``lifting-train``: train any lifter on synthetic scenes - one process, torchrun or Ray Train.

Examples::

    lifting-train --lifter bevformer --steps 2000                       # one process
    torchrun --nproc_per_node 4 -m lifting.train_cli --lifter lift_splat  # DDP on one node
    lifting-train --lifter tpvformer --model triplane --ray --ray-num-workers 8 --ray-use-gpu
    lifting-train --lifter depth_warp --depth-weight 0.05 --webdataset --webdataset-root /fast/shards

``--model bev`` trains BEV segmentation (``build_bev_model`` + ``BEVSegmentationTask``);
``--model triplane`` trains 3D semantic occupancy with the triplane version of the lifter
(``build_triplane_model`` + ``OccupancyTask``). ``--depth-weight`` adds a depth cross-entropy for
lifters with a ``DepthHead`` (``DepthSupervisedTask``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from lifting.data import SyntheticSceneDataset
from lifting.distributed import DistributedSession, SceneLoaderBuilder, launch_on_ray
from lifting.geometry import GridSpec
from lifting.lifters import LIFTERS
from lifting.models import build_bev_model
from lifting.training import BEVSegmentationTask, DepthSupervisedTask, OccupancyTask
from lifting.training.distributed_trainer import DistributedTrainer

#: Arguments that change the training *samples*; they form the WebDataset fingerprint.
DATA_ARGUMENTS = ("train_scenes", "resolution", "seed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a lifting model on synthetic multi-camera scenes.")
    model = p.add_argument_group("model")
    model.add_argument("--lifter", default="simple_bev", choices=LIFTERS.names())
    model.add_argument("--model", default="bev", choices=("bev", "triplane"),
                       help="bev: BEV segmentation; triplane: 3D occupancy with the triplane lifter")
    model.add_argument("--channels", type=int, default=32)
    model.add_argument("--num-classes", type=int, default=3, help="background + object classes")
    model.add_argument("--resolution", type=int, nargs=3, default=(32, 32, 4), metavar=("X", "Y", "Z"))

    data = p.add_argument_group("data")
    data.add_argument("--train-scenes", type=int, default=512)
    data.add_argument("--val-scenes", type=int, default=64)
    data.add_argument("--batch-size", type=int, default=4, help="scenes per batch per rank")
    data.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per rank")

    train = p.add_argument_group("training")
    train.add_argument("--steps", type=int, default=300, help="optimiser steps (per rank)")
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--depth-weight", type=float, default=0.0,
                       help="depth cross-entropy weight for depth-based lifters (0 = off)")
    train.add_argument("--aux-weight", type=float, default=0.5,
                       help="per-plane auxiliary loss weight (--model triplane)")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--eval-every", type=int, default=1, help="evaluate + checkpoint every N epochs")
    train.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/lift"))
    train.add_argument("--keep-ckpts", type=int, default=3, help="newest checkpoints kept (0 = all)")
    train.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    train.add_argument("--json", type=Path, default=None, help="write the final metrics here")

    ray = p.add_argument_group("Ray Train (optional, needs halos-lift[ray])")
    ray.add_argument("--ray", action="store_true", help="launch on Ray Train workers")
    ray.add_argument("--ray-address", default=None, help="'auto' or ray://host:10001 (default: local Ray)")
    ray.add_argument("--ray-num-workers", type=int, default=1, help="training processes (one per GPU)")
    ray.add_argument("--ray-use-gpu", action="store_true", help="one GPU per worker")
    ray.add_argument("--ray-cpus-per-worker", type=float, default=None)
    ray.add_argument("--ray-storage-path", default="ray_results",
                     help="where Ray persists reported checkpoints (local, NFS or s3://)")
    ray.add_argument("--ray-max-failures", type=int, default=0, help="worker-group restarts")

    wds = p.add_argument_group("WebDataset (optional, needs halos-lift[webdataset])")
    wds.add_argument("--webdataset", action="store_true",
                     help="write the training scenes to tar shards once, then stream them")
    wds.add_argument("--webdataset-root", default="data/webdataset")
    wds.add_argument("--webdataset-samples-per-shard", type=int, default=64,
                     help="keep >= ranks x DataLoader workers shards")
    wds.add_argument("--webdataset-convert-workers", type=int, default=0)
    wds.add_argument("--webdataset-shuffle-buffer", type=int, default=256)
    wds.add_argument("--webdataset-overwrite", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------- configuration
def ray_config(args: argparse.Namespace) -> Any:
    """HaleBlocks ``RayConfig`` for the ``--ray*`` flags."""
    from hale_core.config.sections.ray import RayConfig  # lazy: optional extra

    resources: dict[str, float] = {}
    if args.ray_cpus_per_worker is not None:
        resources["CPU"] = args.ray_cpus_per_worker
    if args.ray_use_gpu:
        resources["GPU"] = 1
    return RayConfig(
        enabled=True,
        address=args.ray_address,
        num_workers=args.ray_num_workers,
        use_gpu=args.ray_use_gpu,
        resources_per_worker=resources,
        storage_path=args.ray_storage_path,
        max_failures=args.ray_max_failures,
    )


def webdataset_config(args: argparse.Namespace) -> Any | None:
    """HaleBlocks ``WebDatasetConfig`` for the ``--webdataset*`` flags (``None`` when off)."""
    if not args.webdataset:
        return None
    from hale_core.config.sections.webdataset import WebDatasetConfig  # lazy: optional extra

    return WebDatasetConfig(
        enabled=True,
        root=args.webdataset_root,
        samples_per_shard=args.webdataset_samples_per_shard,
        convert_workers=args.webdataset_convert_workers,
        shuffle_buffer=args.webdataset_shuffle_buffer,
        overwrite=args.webdataset_overwrite,
        seed=args.seed,
    )


def data_fingerprint(args: argparse.Namespace) -> str:
    """Hash of the arguments that change the training scenes."""
    spec = {name: getattr(args, name) for name in DATA_ARGUMENTS}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=list).encode()).hexdigest()


def build_model_and_task(args: argparse.Namespace, grid: GridSpec, rig) -> tuple[torch.nn.Module, Any]:
    """The model for ``--model`` / ``--lifter`` and the matching task (+ depth supervision)."""
    common = dict(grid=grid, num_classes=args.num_classes, image_size=rig.image_size,
                  num_cameras=rig.num_cameras, channels=args.channels)
    if args.model == "triplane":
        from lifting.triplane import build_triplane_model  # lazy: keeps BEV-only runs light

        model = build_triplane_model(args.lifter, aux_heads=args.aux_weight > 0, **common)
        task: Any = OccupancyTask(args.num_classes, aux_weight=args.aux_weight)
    else:
        model = build_bev_model(args.lifter, **common)
        task = BEVSegmentationTask(args.num_classes)
    if args.depth_weight > 0:
        task = DepthSupervisedTask(task, weight=args.depth_weight)
    return model, task


# ---------------------------------------------------------------------------- training
def train(args: argparse.Namespace) -> dict[str, float]:
    """Train on this process (one rank of a torchrun / Ray run, or alone). Returns rank 0's
    final metrics (``{}`` on the other ranks)."""
    session = DistributedSession.start(args.device)
    torch.manual_seed(args.seed)  # identical initial weights on every rank (DDP also broadcasts)
    grid = GridSpec(resolution=tuple(args.resolution))
    train_set = SyntheticSceneDataset(args.train_scenes, grid, seed=args.seed + 1)
    val_set = SyntheticSceneDataset(args.val_scenes, grid, seed=args.seed + 2)
    model, task = build_model_and_task(args, grid, train_set.rig)
    trainer = DistributedTrainer(model, task, session, lr=args.lr, weight_decay=args.weight_decay,
                                 grad_clip=args.grad_clip)

    builder = SceneLoaderBuilder(session, args.batch_size, args.num_workers,
                                 webdataset=webdataset_config(args), seed=args.seed)
    fingerprint = data_fingerprint(args)
    loader = builder.train_loader(train_set, f"synthetic-train-{fingerprint[:12]}", fingerprint)
    val_loader = builder.eval_loader(val_set)

    reporter = None
    if args.ray:
        from hale_core.distributed import RayReporter  # lazy: optional extra

        reporter = RayReporter()
    resume = args.resume or _restored_ray_checkpoint(args)
    if resume is not None:
        trainer.load_checkpoint(resume, args.steps)
        _log(session, f"resumed from {resume} at step {trainer.global_step}")

    final: dict[str, float] = {}

    def on_epoch_end(epoch: int, step: int, loss: float) -> None:
        metrics = {"epoch": float(epoch), "step": float(step), "train/loss": loss}
        done = step >= args.steps
        checkpoint = None
        if epoch % args.eval_every == 0 or done:
            if session.is_main:
                metrics["val/miou"] = trainer.evaluate(val_loader).mean_foreground()
                checkpoint = trainer.save_checkpoint(args.ckpt_dir, keep_last=args.keep_ckpts)
            session.barrier()  # the checkpoint is complete before anyone reports it
        _log(session, f"epoch {epoch} | step {step} | "
             + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items() if "/" in k))
        if reporter is not None:  # every rank reports; rank 0 attaches the checkpoint
            reporter.report(metrics, checkpoint_dir=args.ckpt_dir if checkpoint else None)
        final.update(metrics)

    try:
        trainer.fit_loader(loader, args.steps, on_epoch_end=on_epoch_end)
    finally:
        session.close()
    return final if session.is_main else {}


def _restored_ray_checkpoint(args: argparse.Namespace) -> Path | None:
    """After a worker failure Ray restarts the group from the last reported checkpoint."""
    if not args.ray:
        return None
    from hale_core.distributed import RayReporter  # lazy: optional extra

    restored = RayReporter.latest_checkpoint_dir()
    checkpoints = DistributedTrainer.checkpoints(restored) if restored else []
    return checkpoints[-1] if checkpoints else None


def _log(session: DistributedSession, message: str) -> None:
    """Log ``message`` once per job (rank 0 only), so DDP / Ray workers do not repeat it."""
    if session.is_main:
        logger.info("[lifting-train] {}", message)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ray:
        # Workers import lifting.train_cli and call train(args); nothing else is pickled.
        result = launch_on_ray("lifting.train_cli:train", args, ray_config(args))
        metrics = dict(result.metrics)
        logger.info("[lifting-train] ray: {}  checkpoint={}", metrics, result.checkpoint_path)
    else:
        metrics = train(args)
    if args.json is not None and metrics:
        args.json.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
