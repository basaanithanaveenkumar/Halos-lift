"""Complete triplane models: occupancy and BEV, forward contracts and trainability."""

from __future__ import annotations

import pytest
import torch

from lifting.data import SceneBatch
from lifting.geometry import GridSpec
from lifting.training import BEVSegmentationTask, OccupancyTask, Trainer
from lifting.triplane import (
    TRIPLANE_LIFTERS,
    TriplaneOccupancyModel,
    build_triplane_bev_model,
    build_triplane_model,
)

IMAGE = (32, 64)


def small(name: str) -> dict:
    kwargs: dict = {"num_layers": 1} if name in {"bevformer", "tpvformer", "alpamayo"} else {}
    return kwargs


@pytest.mark.parametrize("name", TRIPLANE_LIFTERS.names())
def test_occupancy_model_outputs(name: str, batch: SceneBatch, grid: GridSpec) -> None:
    model = build_triplane_model(
        name, grid, 3, IMAGE, channels=16, aux_heads=True, **small(name)
    ).eval()
    pts = grid.voxel_centers()[2:4, 5:7, 1:3].reshape(1, -1, 3).expand(2, -1, -1)
    with torch.no_grad():
        out = model(batch.images, batch.cameras, points=pts)
    assert out["voxel_logits"].shape == (2, 3, *grid.resolution)
    assert torch.allclose(
        out["point_logits"], out["voxel_logits"][:, :, 2:4, 5:7, 1:3].reshape(2, 3, -1), atol=1e-4
    )
    nx, ny, nz = grid.resolution
    assert [a.shape[-2:] for a in out["aux_logits"]] == [(nx, ny), (nx, nz), (ny, nz)]


def test_occupancy_model_rejects_bev_lifter(grid: GridSpec) -> None:
    from lifting.encoders import SimpleConvEncoder
    from lifting.lifters import BilinearSamplingLifter, TPVAggregator

    with pytest.raises(TypeError):
        TriplaneOccupancyModel(
            SimpleConvEncoder(8), BilinearSamplingLifter(grid, 8, 8), TPVAggregator(grid, 8, 3)
        )


@pytest.mark.parametrize("name", TRIPLANE_LIFTERS.names())
def test_occupancy_training_reduces_loss(name: str, batch: SceneBatch, grid: GridSpec) -> None:
    torch.manual_seed(0)
    model = build_triplane_model(name, grid, 3, IMAGE, channels=16, aux_heads=True, **small(name))
    trainer = Trainer(model, OccupancyTask(3, aux_weight=0.5), lr=3e-3)
    history = trainer.fit([batch], steps=30)
    first, last = history.losses[0], sum(history.losses[-5:]) / 5
    assert last < 0.6 * first, f"{name}: loss {first:.3f} -> {last:.3f}"
    # the BEV head is frozen in occupancy models and must stay untouched
    assert all(p.grad is None for p in model.lifter.bev_head.parameters())


@pytest.mark.parametrize("name", ["simple_bev", "tiim", "bevformer"])
def test_bev_model_training_reduces_loss(name: str, batch: SceneBatch, grid: GridSpec) -> None:
    torch.manual_seed(0)
    model = build_triplane_bev_model(name, grid, 3, IMAGE, channels=16, **small(name))
    trainer = Trainer(model, BEVSegmentationTask(3), lr=3e-3)
    history = trainer.fit([batch], steps=30)
    assert sum(history.losses[-5:]) / 5 < 0.6 * history.losses[0]
    assert trainer.predict(batch).shape == (2, *grid.bev_shape)
