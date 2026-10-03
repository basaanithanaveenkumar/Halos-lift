import pytest
import torch

from lifting import GridSpec, build_bev_model
from lifting.data import SceneBatch
from lifting.lifters import BEVToTPV, PlaneAuxHead, TPVAggregator, TPVPlanes


@pytest.mark.parametrize("bev_size", [(16, 16), (8, 12), (5, 7)])
def test_bev_to_tpv_shapes(bev_size: tuple[int, int]) -> None:
    conv = BEVToTPV(8, 6, resolution=(16, 12, 4))
    planes = conv(torch.randn(2, 8, *bev_size))
    assert isinstance(planes, TPVPlanes)
    assert planes.xy.shape == (2, 6, 16, 12)
    assert planes.xz.shape == (2, 6, 16, 4)
    assert planes.yz.shape == (2, 6, 12, 4)
    assert planes.resolution == (16, 12, 4)


def test_bev_to_tpv_gradients_and_single_sample() -> None:
    conv = BEVToTPV(4, 4, resolution=(8, 8, 3)).train()
    bev = torch.randn(1, 4, 8, 8, requires_grad=True)
    planes = conv(bev)
    sum(p.sum() for p in (planes.xy, planes.xz, planes.yz)).backward()
    assert bev.grad is not None and bev.grad.abs().sum() > 0
    assert all(p.grad is not None for p in conv.parameters())


def test_bev_to_tpv_columns_follow_the_matching_bev_axis() -> None:
    """``xz`` must depend on the X position of the BEV map and ``yz`` on its Y position."""
    conv = BEVToTPV(2, 2, resolution=(8, 8, 3)).eval()
    bev = torch.zeros(1, 2, 8, 8)
    bev[..., 2, :] = 1.0  # a stripe along Y at x == 2
    base = conv(torch.zeros_like(bev))
    planes = conv(bev)
    assert (planes.xz - base.xz).abs().amax(dim=(1, 3))[0, 2] > 0  # reacts at x == 2
    assert torch.allclose(planes.yz, conv(bev.flip(2)).yz)  # yz pools X away


def test_bev_lifter_output_feeds_tpv_aggregator(batch: SceneBatch, grid: GridSpec) -> None:
    model = build_bev_model("simple_bev", grid, 3, (32, 64), channels=16)
    feats = [
        f.view(2, 4, *f.shape[1:]) for f in model.encoder(batch.images.flatten(0, 1))
    ]
    bev = model.lifter(feats, batch.cameras)
    planes = BEVToTPV(16, 16, grid.resolution)(bev)
    out = TPVAggregator(grid, 16, 3)(planes)
    assert out["voxel_logits"].shape == (2, 3, *grid.resolution)


@pytest.mark.parametrize("channels", [16, 6, 5, 1])
def test_plane_aux_head_shape(channels: int) -> None:
    out = PlaneAuxHead(channels, 3)(torch.randn(2, channels, 7, 5))
    assert out.shape == (2, 3, 7, 5)
