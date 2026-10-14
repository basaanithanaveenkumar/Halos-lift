"""``lifting-train`` end to end: one process, resume, DDP over Gloo, WebDataset and Ray Train."""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from lifting.train_cli import main, parse_args, train
from lifting.training import DistributedTrainer

HAS_HALE_CORE = importlib.util.find_spec("hale_core") is not None
HAS_WEBDATASET = HAS_HALE_CORE and importlib.util.find_spec("webdataset") is not None
HAS_RAY = HAS_HALE_CORE and importlib.util.find_spec("ray") is not None

TINY = ["--train-scenes", "16", "--val-scenes", "4", "--resolution", "8", "8", "2", "--channels", "8",
        "--batch-size", "2", "--device", "cpu"]


def tiny_args(tmp: Path, *extra: str):
    return parse_args([*TINY, "--ckpt-dir", str(tmp / "ckpt"), *extra])


def test_single_process_trains_evaluates_and_checkpoints(tmp_path: Path) -> None:
    metrics = train(tiny_args(tmp_path, "--steps", "16", "--eval-every", "1", "--keep-ckpts", "0"))
    assert metrics["step"] == 16 and metrics["epoch"] == 2 and "val/miou" in metrics
    names = [p.name for p in DistributedTrainer.checkpoints(tmp_path / "ckpt")]
    assert names == ["lift_step0000008.pt", "lift_step0000016.pt"]
    state = torch.load(tmp_path / "ckpt" / names[-1], weights_only=True)
    assert not any(k.startswith("module.") for k in state["model"])


def test_resume_continues_the_schedule(tmp_path: Path) -> None:
    train(tiny_args(tmp_path, "--steps", "8"))
    metrics = train(tiny_args(tmp_path, "--steps", "16", "--resume",
                              str(tmp_path / "ckpt" / "lift_step0000008.pt")))
    assert metrics["step"] == 16 and metrics["epoch"] == 2


@pytest.mark.parametrize("model, lifter, extra", [
    ("triplane", "tpvformer", []),
    ("bev", "lift_splat", ["--depth-weight", "0.1"]),
])
def test_models_tasks_and_depth_supervision(tmp_path: Path, model: str, lifter: str, extra) -> None:
    metrics = train(tiny_args(tmp_path, "--steps", "8", "--model", model, "--lifter", lifter, *extra))
    assert metrics["step"] == 8 and torch.isfinite(torch.tensor(metrics["train/loss"]))


def test_cli_writes_json(tmp_path: Path) -> None:
    assert main([*TINY, "--steps", "8", "--ckpt-dir", str(tmp_path), "--json", str(tmp_path / "m.json")]) == 0
    assert '"val/miou"' in (tmp_path / "m.json").read_text()


@pytest.mark.skipif(not HAS_WEBDATASET, reason="needs halos-lift[webdataset]")
def test_webdataset_converts_once_and_streams(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    args = tiny_args(tmp_path, "--steps", "16", "--webdataset", "--webdataset-root", str(shards),
                     "--webdataset-samples-per-shard", "4")
    first = train(args)
    (directory,) = [p for p in shards.iterdir() if p.is_dir()]
    tars = sorted(directory.glob("shard-*.tar"))
    assert len(tars) == 4
    stamps = [t.stat().st_mtime_ns for t in tars]
    second = train(args)  # reuses the shards
    assert [t.stat().st_mtime_ns for t in tars] == stamps
    assert first["step"] == second["step"] == 16


# ---------------------------------------------------------------------------- DDP
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank(rank: int, world: int, port: int, tmp: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    metrics = train(tiny_args(Path(tmp), "--steps", "8"))
    torch.save(metrics, Path(tmp) / f"metrics{rank}.pt")


def test_ddp_two_ranks_shard_data_and_stay_in_sync(tmp_path: Path) -> None:
    mp.spawn(_rank, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    m0, m1 = (torch.load(tmp_path / f"metrics{r}.pt") for r in (0, 1))
    assert m1 == {}  # only rank 0 returns metrics
    # 16 scenes / 2 ranks / batch 2 = 4 steps per epoch -> 8 steps = 2 epochs.
    assert m0["epoch"] == 2 and m0["step"] == 8 and "val/miou" in m0
    assert [p.name for p in DistributedTrainer.checkpoints(tmp_path / "ckpt")][-1] == "lift_step0000008.pt"


# ---------------------------------------------------------------------------- Ray Train
@pytest.mark.skipif(not HAS_RAY, reason="needs halos-lift[ray]")
def test_ray_two_workers_report_metrics_and_checkpoint(tmp_path: Path, monkeypatch) -> None:
    from lifting.distributed import launch_on_ray
    from lifting.train_cli import ray_config

    src = str(Path(__file__).resolve().parents[2] / "src")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
    args = tiny_args(tmp_path, "--steps", "8", "--ray", "--ray-num-workers", "2",
                     "--ray-storage-path", str(tmp_path / "ray"))
    config = ray_config(args)
    config.runtime_env = {"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}}
    result = launch_on_ray("lifting.train_cli:train", args, config)
    assert result.ok, result.error
    assert result.metrics["step"] == 8 and "val/miou" in result.metrics
    assert any(name.startswith("lift_step") for name in os.listdir(result.checkpoint_path))
