"""BEV + side two-plane TPV (``bev_side``): optional front plane, one reused perspective volume."""

from __future__ import annotations

import pytest
import torch

from lifting.encoders import SimpleConvEncoder
from lifting.exceptions import InvalidConfigError
from lifting.geometry import Cameras, GridSpec
from lifting.lifters import TPVPlanes
from lifting.triplane import (
    AxisAttentionPool,
    BEVSideMixer,
    BEVSideTriplaneLifter,
    VoxelVolumeSource,
    build_triplane_lifter,
    build_triplane_model,
)

H, W, C = 32, 64, 16


def make(
    grid: GridSpec, cameras: Cameras, **kwargs
) -> tuple[BEVSideTriplaneLifter, list[torch.Tensor]]:
    enc = SimpleConvEncoder(C, num_levels=2)
    lifter = build_triplane_lifter("bev_side", grid, C, (H, W), cameras.num_cameras, enc, **kwargs)
    b, n = cameras.batch_size, cameras.num_cameras
    feats = [torch.randn(b, n, C, H // s, W // s) for s in enc.strides]
    return lifter, feats


# ---------------------------------------------------------------------------- TPVPlanes without yz
def test_two_plane_tpv_math() -> None:
    xy, xz = torch.randn(2, 3, 5, 6), torch.randn(2, 3, 5, 4)
    planes = TPVPlanes(xy, xz)
    assert planes.names == ("xy", "xz") and not planes.has_front and planes.resolution == (5, 6, 4)
    torch.testing.assert_close(planes.to_voxels(), xy[..., None] + xz[:, :, :, None, :])
    tokens = planes.to_tokens()
    assert tokens.shape == (2, 5 * 6 + 5 * 4, 3)
    again = TPVPlanes.from_tokens(tokens, (5, 6, 4), names=planes.names)
    assert torch.equal(again.xy, xy) and torch.equal(again.xz, xz) and again.yz is None
    up = planes.upsample(2)
    assert up.yz is None and up.xy.shape == (2, 3, 10, 12)
    # A point query reads xy(x, y) + xz(x, z): exact at a cell centre.
    point = torch.tensor([[[-1 + 3 / 5, -1 + 5 / 6, -1 + 1 / 4]]]).expand(
        2, -1, -1
    )  # cell (1, 2, 0)
    torch.testing.assert_close(planes.sample_points(point)[..., 0], xy[:, :, 1, 2] + xz[:, :, 1, 0])


def test_three_plane_behaviour_is_unchanged() -> None:
    xy, xz, yz = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 5), torch.randn(1, 2, 4, 5)
    planes = TPVPlanes(xy, xz, yz)
    assert planes.names == ("xy", "xz", "yz") and planes.has_front
    vox = planes.to_voxels()
    torch.testing.assert_close(vox[0, :, 1, 2, 3], xy[0, :, 1, 2] + xz[0, :, 1, 3] + yz[0, :, 2, 3])


# ---------------------------------------------------------------------------- the lifter
def test_planes_and_bev(grid: GridSpec, cameras: Cameras) -> None:
    lifter, feats = make(grid, cameras)
    nx, ny, nz = grid.resolution
    planes = lifter.lift_planes(feats, cameras)
    assert lifter.planes == planes.names == ("xy", "xz") and planes.yz is None
    assert planes.xy.shape == (2, C, nx, ny) and planes.xz.shape == (2, C, nx, nz)
    assert lifter(feats, cameras).shape == (2, C, nx, ny)
    assert isinstance(lifter.volume, VoxelVolumeSource) and lifter.uses_geometry


def test_the_cameras_are_read_once_and_both_planes_reuse_it(
    grid: GridSpec, cameras: Cameras
) -> None:
    lifter, feats = make(grid, cameras)
    feats = [f.requires_grad_() for f in feats]  # simple_bev's volume itself has no parameters
    lifter.eval()
    calls = []
    original = lifter.volume.lift_voxels

    def counted(*args, **kwargs):
        out = original(*args, **kwargs)
        out.retain_grad()
        calls.append(out)
        return out

    lifter.volume.lift_voxels = counted
    planes = lifter.lift_planes(feats, cameras)
    assert len(calls) == 1  # one perspective sampling for both planes
    volume = calls[0]
    planes.xy.sum().backward(retain_graph=True)
    assert volume.grad.abs().sum() > 0
    volume.grad = None
    planes.xz.sum().backward()
    assert volume.grad.abs().sum() > 0
    with torch.no_grad():
        torch.testing.assert_close(
            lifter.planes_from_volume(volume.detach()).xy, planes.xy.detach()
        )


@pytest.mark.parametrize("volume", ["simple_bev", "depth_warp", "lift_splat", "tiim"])
@pytest.mark.parametrize("pooling", ["attention", "fold"])
def test_every_volume_source_and_pooling_trains(
    volume: str, pooling: str, grid: GridSpec, cameras: Cameras
) -> None:
    lifter, feats = make(grid, cameras, volume=volume, pooling=pooling)
    feats = [f.requires_grad_() for f in feats]
    planes = lifter.lift_planes(feats, cameras)
    (planes.xy.square().mean() + planes.xz.square().mean()).backward()
    # Every parameter on the plane path learns (the BEV head is only used by ``forward``).
    missing = [
        n for n, p in lifter.named_parameters() if p.grad is None and not n.startswith("bev_head")
    ]
    assert not missing, (volume, missing)
    assert feats[lifter.level].grad.abs().sum() > 0


def test_mix_layers_couple_the_planes(grid: GridSpec, cameras: Cameras) -> None:
    lifter, feats = make(grid, cameras, mix_layers=1)
    volume = lifter.lift_volume(feats, cameras).detach()
    bev = lifter.pool_height(volume).detach().requires_grad_()
    side = lifter.pool_lateral(volume.permute(0, 1, 2, 4, 3)).detach().requires_grad_()
    new_bev, new_side = lifter.mixers[0](bev, side)
    new_side[:, :, 3].sum().backward()  # side row x = 3 ...
    assert bev.grad[:, :, 3].abs().sum() > 0  # ... reads the BEV row x = 3
    independent = BEVSideTriplaneLifter(grid, C, C, mix_layers=0)
    assert len(independent.mixers) == 0


def test_attention_pool_weights() -> None:
    pool = AxisAttentionPool(8, 5, 6, num_heads=2)
    volume = torch.randn(2, 8, 3, 4, 5)
    out = pool(volume)
    assert out.shape == (2, 6, 3, 4)
    torch.testing.assert_close(pool.last_weights.sum(-1), torch.ones(2, 2, 3, 4))
    flipped = pool(volume.flip(-1))  # the position embedding makes the pool order-aware
    assert not torch.allclose(out, flipped)


def test_mixer_shapes() -> None:
    mixer = BEVSideMixer(8, (5, 6, 4), num_heads=2)
    bev, side = mixer(torch.randn(2, 8, 5, 6), torch.randn(2, 8, 5, 4))
    assert bev.shape == (2, 8, 5, 6) and side.shape == (2, 8, 5, 4)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"volume": "bevformer"}, "voxel volume"),
        ({"volume": "bev_side"}, "itself"),
        ({"pooling": "max"}, "pooling"),
        ({"num_heads": 3}, "num_heads"),
    ],
)
def test_invalid_configs(grid: GridSpec, kwargs: dict, match: str) -> None:
    with pytest.raises(InvalidConfigError, match=match):
        BEVSideTriplaneLifter(grid, C, C, **kwargs)


def test_occupancy_model_has_no_front_aux_head(grid: GridSpec, cameras: Cameras, batch) -> None:
    model = build_triplane_model("bev_side", grid, 3, (H, W), channels=C, aux_heads=True)
    assert sum(p.numel() for p in model.aux_heads[2].parameters()) == 0
    out = model(batch.images, cameras)
    assert out["aux_logits"][2] is None and out["planes"].yz is None


def test_fewer_plane_cells_than_three_plane_tpv(grid: GridSpec) -> None:
    nx, ny, nz = grid.resolution
    two = sum(h * w for h, w in TPVPlanes.plane_shapes(grid.resolution, ("xy", "xz")))
    three = sum(h * w for h, w in TPVPlanes.plane_shapes(grid.resolution))
    assert three - two == ny * nz


def test_storyboard_drops_the_front_plane() -> None:
    from lifting.triplane.animation import STORYBOARDS

    board = STORYBOARDS.get("bev_side")()
    assert board.planes == (0, 1)
    steps = board.steps()
    frame = board.render_frame(steps, len(steps) - 1, 1.0)
    assert frame.size == board.SIZE
