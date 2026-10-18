"""Per-plane auxiliary supervision for TPVFormer occupancy."""

import pytest
import torch
import torch.nn.functional as F

from lifting import GridSpec, build_tpvformer
from lifting.data import SceneBatch
from lifting.training import OccupancyTask, Trainer


def test_aux_heads_are_off_by_default(batch: SceneBatch, grid: GridSpec) -> None:
    model = build_tpvformer(grid, num_classes=3, channels=16, num_layers=1)
    assert model.aux_heads is None
    assert "aux_logits" not in model(batch.images, batch.cameras)


def test_aux_logits_match_plane_shapes(batch: SceneBatch, grid: GridSpec) -> None:
    model = build_tpvformer(grid, num_classes=3, channels=16, num_layers=1, aux_heads=True)
    nx, ny, nz = grid.resolution
    xy, xz, yz = model(batch.images, batch.cameras)["aux_logits"]
    assert xy.shape == (2, 3, nx, ny)
    assert xz.shape == (2, 3, nx, nz)
    assert yz.shape == (2, 3, ny, nz)


def test_aux_loss_adds_to_voxel_loss(batch: SceneBatch, grid: GridSpec) -> None:
    model = build_tpvformer(grid, num_classes=3, channels=16, num_layers=1, aux_heads=True)
    outputs = model(batch.images, batch.cameras)
    plain = OccupancyTask(3).loss(outputs, batch)
    task = OccupancyTask(3, aux_weight=0.5)
    assert torch.isclose(task.loss(outputs, batch), plain + 0.5 * task.plane_loss(outputs, batch))
    assert task.plane_loss(outputs, batch) > 0


def test_plane_targets_collapse_the_right_axis(grid: GridSpec) -> None:
    occ = torch.zeros(1, *grid.resolution, dtype=torch.long)
    occ[0, 3, 5, 2] = 2
    logits = tuple(
        F.one_hot(t, 3).permute(0, 3, 1, 2).float() * 100
        for t in (occ.amax(3), occ.amax(2), occ.amax(1))
    )

    class _Batch:
        occupancy = occ

    # perfectly confident, perfectly correct plane logits -> ~zero loss
    loss = OccupancyTask(3).plane_loss({"aux_logits": logits}, _Batch())
    assert loss < 1e-3


def test_aux_weight_needs_aux_heads(batch: SceneBatch, grid: GridSpec) -> None:
    model = build_tpvformer(grid, num_classes=3, channels=16, num_layers=1)
    with pytest.raises(ValueError, match="aux_heads"):
        OccupancyTask(3, aux_weight=0.5).loss(model(batch.images, batch.cameras), batch)


def test_occupancy_training_with_aux_loss(batch: SceneBatch, grid: GridSpec) -> None:
    torch.manual_seed(0)
    model = build_tpvformer(grid, num_classes=3, channels=16, num_layers=1, aux_heads=True)
    trainer = Trainer(model, OccupancyTask(3, aux_weight=0.5), lr=3e-3)
    history = trainer.fit([batch], steps=40)
    assert sum(history.losses[-5:]) / 5 < 0.5 * history.losses[0]
    assert all(p.grad is not None for p in model.aux_heads.parameters())
