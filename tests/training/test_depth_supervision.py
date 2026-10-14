"""DepthSupervisedTask: hooks every DepthHead, adds a depth CE, and actually teaches depth."""

from __future__ import annotations

import math

import pytest
import torch

from lifting.data import SceneBatch, SyntheticSceneDataset
from lifting.geometry import DepthBins, GridSpec
from lifting.lifters.depth_head import DepthHead
from lifting.models import build_bev_model
from lifting.training import BEVSegmentationTask, DepthSupervisedTask, OccupancyTask, Trainer
from lifting.training.depth_supervision import DEPTH_PREDICTIONS, depth_targets
from lifting.triplane import build_triplane_model

GRID = GridSpec(resolution=(16, 16, 4))


def make_batch(start: int = 0, size: int = 2) -> SceneBatch:
    data = SyntheticSceneDataset(64, GRID, seed=0)
    return SceneBatch.collate([data[start + i] for i in range(size)])


def test_depth_targets_min_pool_and_mask() -> None:
    bins = DepthBins(1.0, 9.0, 8)  # 1 m bins
    depth = torch.full((1, 4, 4), float("inf"))
    depth[0, :2, :2] = torch.tensor([[5.5, 2.2], [7.0, 3.9]])  # nearest surface of patch = 2.2
    depth[0, 2:, 2:] = 0.5  # closer than min_depth -> invalid
    index, valid = depth_targets(depth, (2, 2), bins)
    assert valid.tolist() == [[[True, False], [False, False]]]
    assert index[0, 0, 0].item() == 1  # 2.2 m falls in bin [2, 3)


@pytest.mark.parametrize("lifter", ["depth_warp", "lift_splat"])
def test_hooks_capture_bev_and_triplane_depth_heads(lifter: str) -> None:
    batch = make_batch()
    for model, base in (
        (build_bev_model(lifter, GRID, channels=16), BEVSegmentationTask(3)),
        (build_triplane_model(lifter, GRID, channels=16), OccupancyTask(3)),
    ):
        task = DepthSupervisedTask(base, weight=1.0)
        outputs = task.forward(model, batch)
        (probs, bins), = outputs[DEPTH_PREDICTIONS]
        assert probs.shape[:2] == (batch.images.shape[0] * batch.images.shape[1], bins.num_bins)
        assert bins is next(m for m in model.modules() if hasattr(m, "depth_bins")).depth_bins
        loss = task.loss(outputs, batch)
        assert torch.isfinite(loss) and loss > base.loss(outputs, batch)
        # Hooks are removed after every forward pass.
        assert all(not m._forward_hooks for m in model.modules() if isinstance(m, DepthHead))


def test_model_without_depth_head_is_rejected() -> None:
    model = build_bev_model("simple_bev", GRID, channels=16)
    with pytest.raises(ValueError, match="DepthHead"):
        DepthSupervisedTask(BEVSegmentationTask(3)).forward(model, make_batch())


def test_depth_loss_teaches_depth() -> None:
    """A few supervised steps make the depth distribution sharper and better correlated."""
    torch.manual_seed(0)
    model = build_bev_model("depth_warp", GRID, channels=16)
    task = DepthSupervisedTask(BEVSegmentationTask(3), weight=1.0)
    trainer = Trainer(model, task, lr=3e-3)
    probe = make_batch(40, 4)

    def diagnostics() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            (probs, bins), = task.forward(model, probe)[DEPTH_PREDICTIONS]
        return task.entropy_and_correlation(probs, probe.depth.flatten(0, 1), bins)

    entropy0, _ = diagnostics()
    trainer.fit([make_batch(i, 4) for i in range(0, 32, 4)], steps=60, schedule=False)
    entropy1, corr1 = diagnostics()
    assert entropy1 < entropy0 - 0.3
    assert not math.isnan(corr1) and corr1 > 0.3
