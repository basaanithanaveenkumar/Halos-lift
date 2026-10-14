"""Mip-NeRF 360 parametrizer (Alpamayo) and the reference points built on it."""

from __future__ import annotations

import pytest
import torch

from lifting.geometry import GridSpec
from lifting.lifters.alpamayo import AlpamayoReferencePoints, MipNeRF360Parametrizer

TOTALS = [1, 2, 4, 6, 8, 16, 32, 96, 200]


@pytest.mark.parametrize("total", TOTALS)
@pytest.mark.parametrize("inner_frac", [0.6, 0.75, 0.9, 1.0])
def test_centres_strictly_increasing_and_inside_bounds(total: int, inner_frac: float) -> None:
    p = MipNeRF360Parametrizer(total, inner_frac, total_extent_m=32.0)
    centres = p.cell_centers_m()
    assert centres.shape == (total,)
    assert (centres[1:] > centres[:-1]).all()
    assert centres.min() > -16.0 and centres.max() < 16.0


@pytest.mark.parametrize("total", TOTALS)
def test_edges_span_exactly_the_extent(total: int) -> None:
    p = MipNeRF360Parametrizer(total, 0.75, total_extent_m=40.0, center_m=5.0)
    edges = p.cell_edges_m()
    assert edges.shape == (total + 1,)
    assert edges[0].item() == pytest.approx(-15.0, abs=1e-4)
    assert edges[-1].item() == pytest.approx(25.0, abs=1e-4)
    centres = p.cell_centers_m()
    assert ((centres > edges[:-1]) & (centres < edges[1:])).all()


@pytest.mark.parametrize("total", [8, 16, 32, 96])
def test_inner_zone_denser_than_uniform_and_tails_widen(total: int) -> None:
    p = MipNeRF360Parametrizer(total, 0.75, total_extent_m=32.0)
    widths = p.cell_edges_m().diff()
    uniform = 32.0 / total
    half = p.n_outer // 2
    inner = widths[half : half + p.n_inner]
    assert torch.allclose(inner, inner[0].expand_as(inner), atol=1e-5)  # linear inside
    assert inner[0] < uniform - 1e-4  # extra density near the ego vehicle
    right_tail = widths[half + p.n_inner :]
    assert (right_tail.diff() > 0).all()  # cells widen with distance
    assert torch.allclose(widths, widths.flip(0), atol=1e-5)  # symmetric tails
    # Width is continuous at the zone boundary: the first outer cell is only slightly wider.
    assert right_tail[0] < 2 * inner[0]


def test_tiny_axis_with_odd_outer_count_becomes_uniform() -> None:
    p = MipNeRF360Parametrizer(4, 0.75, total_extent_m=4.0, center_m=1.0)
    assert (p.n_inner, p.n_outer) == (4, 0)
    assert torch.allclose(p.cell_centers_m(), torch.tensor([-0.5, 0.5, 1.5, 2.5]))


@pytest.mark.parametrize("bad", [{"inner_frac": 0.5}, {"inner_frac": 0.0}, {"total": 0},
                                 {"total_extent_m": 0.0}])
def test_invalid_configuration_raises(bad: dict) -> None:
    kwargs = {"total": 32, "inner_frac": 0.75, "total_extent_m": 32.0, **bad}
    with pytest.raises(ValueError):
        MipNeRF360Parametrizer(**kwargs)


def test_reference_points_respect_asymmetric_bounds() -> None:
    grid = GridSpec(z_bounds=(-1.0, 3.0), x_bounds=(0.0, 40.0), resolution=(16, 16, 4))
    refs = AlpamayoReferencePoints(grid, num_anchors=(4, 8, 8))
    for plane in range(3):
        pts = refs.anchor_points(plane)
        for axis, (lo, hi) in enumerate(grid.bounds):
            assert pts[..., axis].min() > lo and pts[..., axis].max() < hi, (plane, axis)
    # The xz plane's z queries are the four height cells of [-1, 3], not centred on 0.
    z = refs.anchor_points(1)[:, 0, 2].unique()
    assert torch.allclose(z, torch.tensor([-0.5, 0.5, 1.5, 2.5]))


@pytest.mark.parametrize("total", [4, 8, 16, 32])
def test_metric_to_index_inverts_cell_centres(total: int) -> None:
    p = MipNeRF360Parametrizer(total, 0.75, total_extent_m=24.0, center_m=2.0)
    index = p.metric_to_index(p.cell_centers_m())
    assert torch.allclose(index, torch.arange(total, dtype=torch.float32), atol=1e-4)
