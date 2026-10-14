"""Every triplane lifter: shapes, gradients, BEV consistency and dependence on calibration."""

from __future__ import annotations

import pytest
import torch

from lifting.encoders import SimpleConvEncoder
from lifting.geometry import Cameras, GridSpec
from lifting.lifters import LIFTERS, TPVPlanes
from lifting.triplane import (
    TRIPLANE_LIFTERS,
    BEVLifterTriplaneAdapter,
    MLPViewTriplaneLifter,
    build_triplane_lifter,
)
from tests.helpers import rotation_z

H, W = 32, 64
C = 16


def make(name: str, grid: GridSpec, cameras: Cameras, **kwargs):
    enc = SimpleConvEncoder(C, num_levels=2)
    lifter = build_triplane_lifter(name, grid, C, (H, W), cameras.num_cameras, enc, **kwargs)
    return lifter, enc


def features(enc: SimpleConvEncoder, cameras: Cameras, grad: bool = False) -> list[torch.Tensor]:
    b, n = cameras.batch_size, cameras.num_cameras
    return [torch.randn(b, n, C, H // s, W // s, requires_grad=grad) for s in enc.strides]


@pytest.mark.parametrize("name", TRIPLANE_LIFTERS.names())
def test_planes_bev_shapes_and_gradients(name: str, grid: GridSpec, cameras: Cameras) -> None:
    lifter, enc = make(name, grid, cameras)
    lifter.train()
    feats = features(enc, cameras, grad=True)
    planes = lifter.lift_planes(feats, cameras)
    nx, ny, nz = grid.resolution
    b = cameras.batch_size
    assert isinstance(planes, TPVPlanes)
    assert planes.xy.shape == (b, C, nx, ny)
    assert planes.xz.shape == (b, C, nx, nz)
    assert planes.yz.shape == (b, C, ny, nz)
    assert planes.resolution == grid.resolution
    for p in (planes.xy, planes.xz, planes.yz):
        assert torch.isfinite(p).all()
    loss = sum(p.square().mean() for p in (planes.xy, planes.xz, planes.yz))
    loss.backward()
    assert sum(f.grad.abs().sum() for f in feats if f.grad is not None) > 0


@pytest.mark.parametrize("name", TRIPLANE_LIFTERS.names())
def test_forward_is_bev_of_the_planes(name: str, grid: GridSpec, cameras: Cameras) -> None:
    lifter, enc = make(name, grid, cameras)
    lifter.eval()
    feats = features(enc, cameras)
    with torch.no_grad():
        bev = lifter(feats, cameras)
        again = lifter.bev_head(lifter.lift_planes(feats, cameras))
    assert bev.shape == (cameras.batch_size, C, *grid.bev_shape)
    assert torch.allclose(bev, again, atol=1e-5)


@pytest.mark.parametrize("name", [n for n in TRIPLANE_LIFTERS.names() if n != "mlp_view"])
def test_geometric_triplane_lifters_depend_on_calibration(
    name: str, grid: GridSpec, cameras: Cameras
) -> None:
    lifter, enc = make(name, grid, cameras)
    lifter.eval()
    assert lifter.uses_geometry
    feats = features(enc, cameras)
    with torch.no_grad():
        a = lifter.lift_planes(feats, cameras)
        r = lifter.lift_planes(feats, cameras.transformed(rotation_z(90)))
    for pa, pr in zip((a.xy, a.xz, a.yz), (r.xy, r.xz, r.yz), strict=True):
        assert not torch.allclose(pa, pr, atol=1e-3)


def test_mlp_view_triplane_ignores_calibration(grid: GridSpec, cameras: Cameras) -> None:
    lifter, enc = make("mlp_view", grid, cameras)
    assert isinstance(lifter, MLPViewTriplaneLifter) and not lifter.uses_geometry
    lifter.eval()
    feats = features(enc, cameras)
    with torch.no_grad():
        a = lifter.lift_planes(feats, cameras)
        r = lifter.lift_planes(feats, cameras.transformed(rotation_z(90)))
    assert torch.equal(a.xy, r.xy) and torch.equal(a.xz, r.xz) and torch.equal(a.yz, r.yz)


def test_mlp_view_triplane_rejects_wrong_camera_count(grid: GridSpec, cameras: Cameras) -> None:
    lifter, enc = make("mlp_view", grid, cameras)
    feats = [f[:, :3] for f in features(enc, cameras)]
    with pytest.raises(ValueError):
        lifter.lift_planes(feats, cameras)


@pytest.mark.parametrize("base", LIFTERS.names())
def test_bev_adapter_wraps_every_bev_lifter(base: str, grid: GridSpec, cameras: Cameras) -> None:
    lifter, enc = make("bev_adapter", grid, cameras, base=base)
    assert isinstance(lifter, BEVLifterTriplaneAdapter)
    assert lifter.uses_geometry == LIFTERS.get(base).uses_geometry
    lifter.eval()
    with torch.no_grad():
        planes = lifter.lift_planes(features(enc, cameras), cameras)
    assert planes.resolution == grid.resolution


def test_factory_injects_context(grid: GridSpec, cameras: Cameras) -> None:
    lifter, _ = make("mlp_view", grid, cameras)
    assert lifter.num_cameras == cameras.num_cameras
    lifter, _ = make("bev_adapter", grid, cameras, base="mlp_view")
    assert len(lifter.core.view_mlps) == cameras.num_cameras
    lifter, _ = make("lift_splat", grid, cameras)
    reach = max(abs(v) for v in (*grid.x_bounds, *grid.y_bounds))
    assert lifter.depth_bins.max_depth >= reach


def test_unknown_triplane_lifter(grid: GridSpec, cameras: Cameras) -> None:
    with pytest.raises(KeyError):
        make("does_not_exist", grid, cameras)
