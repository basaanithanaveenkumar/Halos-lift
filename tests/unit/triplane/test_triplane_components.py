"""Building blocks of the triplane package and exactness of the reused lifter stages."""

from __future__ import annotations

import torch

from lifting.geometry import Cameras, DepthBins, GridSpec
from lifting.layers import HeightCompressor
from lifting.lifters import BEVFormerLifter, BilinearSamplingLifter, TPVPlanes
from lifting.triplane import (
    AlpamayoTriplaneAdapter,
    BEVFormerTriplaneLifter,
    PlaneProjection,
    PlaneQueryEncoder,
    PolarRayTriplaneLifter,
    SimpleBEVTriplaneLifter,
    TriplaneBEVHead,
    VoxelToTriplane,
)

H, W = 32, 64


def random_planes(grid: GridSpec, c: int = 4, b: int = 2) -> TPVPlanes:
    nx, ny, nz = grid.resolution
    return TPVPlanes(
        torch.randn(b, c, nx, ny), torch.randn(b, c, nx, nz), torch.randn(b, c, ny, nz)
    )


def test_voxel_to_triplane_collapses_the_right_axis(grid: GridSpec) -> None:
    """A voxel perturbation at (i, j, k) may only change plane cells near (i,j), (i,k), (j,k)."""
    torch.manual_seed(0)
    to_planes = VoxelToTriplane(3, grid, 5).eval()
    nx, ny, nz = grid.resolution
    base = torch.zeros(1, 3, nx, ny, nz)
    bumped = base.clone()
    i, j, k = 3, 11, 2
    bumped[0, :, i, j, k] = 5.0
    with torch.no_grad():
        p0, p1 = to_planes(base), to_planes(bumped)

    def changed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs().amax(dim=(0, 1)) > 1e-6

    def near(mask: torch.Tensor, r: int, c: int) -> None:
        rows, cols = torch.nonzero(mask, as_tuple=True)
        assert mask.any()
        assert (rows - r).abs().max() <= 1 and (cols - c).abs().max() <= 1  # 3x3 conv footprint

    near(changed(p0.xy, p1.xy), i, j)
    near(changed(p0.xz, p1.xz), i, k)
    near(changed(p0.yz, p1.yz), j, k)


def test_triplane_bev_head_is_height_compressed_voxel_sum(grid: GridSpec) -> None:
    head = TriplaneBEVHead(4, grid.resolution[2], 6).eval()
    planes = random_planes(grid)
    ref = HeightCompressor(4, grid.resolution[2], 6).eval()
    ref.load_state_dict(head.compressor.state_dict())
    with torch.no_grad():
        assert torch.allclose(head(planes), ref(planes.to_voxels()))


def test_plane_projection_shapes(grid: GridSpec) -> None:
    out = PlaneProjection(4, 7)(random_planes(grid))
    assert out.xy.shape[1] == out.xz.shape[1] == out.yz.shape[1] == 7
    assert out.resolution == grid.resolution


def test_plane_query_encoder_reference_points_are_cell_centres() -> None:
    enc = PlaneQueryEncoder((16, 4), 16, num_layers=1, num_anchors=8)
    ref = enc.reference_points().view(16, 4, 2)
    assert torch.allclose(ref[5, 2], torch.tensor([(2 + 0.5) / 4, (5 + 0.5) / 16]))


def test_simple_bev_triplane_voxels_equal_simple_bev(grid: GridSpec, cameras: Cameras) -> None:
    feats = torch.rand(cameras.batch_size, cameras.num_cameras, 4, H, W)
    tri = SimpleBEVTriplaneLifter(grid, 4, 4)
    ref = BilinearSamplingLifter(grid, 4, 4)
    assert torch.equal(tri.lift_voxels([feats], cameras), ref.unproject(feats, cameras))


def test_bevformer_triplane_xy_anchors_are_bevformer_pillars(
    grid: GridSpec, cameras: Cameras
) -> None:
    tri = BEVFormerTriplaneLifter(grid, 16, 16, num_layers=1, num_anchors=(4, 8, 8))
    ref_cam, visible = BEVFormerLifter(grid, 16, 16, num_anchors=4).camera_reference_points(cameras)
    (xy_ref, xy_vis), xz, yz = tri.references.camera_reference_points(cameras)
    assert torch.allclose(xy_ref, ref_cam) and torch.equal(xy_vis, visible)
    nx, ny, nz = grid.resolution
    assert xz[0].shape[2:4] == (nx * nz, 8) and yz[0].shape[2:4] == (ny * nz, 8)
    assert [e.plane_shape for e in tri.encoders] == [(nx, ny), (nx, nz), (ny, nz)]


def test_polar_triplane_voxels_read_range_and_elevation(grid: GridSpec, cameras: Cameras) -> None:
    """Ramp features: the range half must return each voxel's (column, depth), the elevation
    half its (u, v) pixel - for voxels seen by exactly one camera."""
    bins = DepthBins(1.0, 25.0, 24)
    lifter = PolarRayTriplaneLifter(grid, 2, 8, depth_bins=bins, num_heads=1)
    b, n = cameras.batch_size, cameras.num_cameras
    cols = ((torch.arange(W) + 0.5) / W).view(1, 1, 1, 1, W).expand(b, n, 1, bins.num_bins, W)
    deps = bins.centers().view(1, 1, 1, -1, 1).expand(b, n, 1, -1, W)
    polar = torch.cat([cols, deps], 2)  # (B, N, 2, D, W)
    v, u = torch.meshgrid((torch.arange(H) + 0.5) / H, (torch.arange(W) + 0.5) / W, indexing="ij")
    image = torch.stack([u, v]).expand(b, n, 2, H, W).clone()
    vox = lifter.polar_to_voxels(polar, image, cameras).flatten(2).transpose(1, 2)  # (B, P, 4)

    pts = grid.voxel_centers().view(1, -1, 3).expand(b, -1, -1)
    uv, depth, valid = cameras.project(pts)
    uv_n = cameras.normalize_uv(uv)
    lo = torch.tensor([0.5 / W, 0.5 / H])
    inner = ((uv_n >= lo) & (uv_n <= 1 - lo)).all(-1)
    ok = (
        valid
        & inner
        & (depth > bins.min_depth + bins.step / 2)
        & (depth < bins.max_depth - bins.step / 2)
    )
    single = (valid.sum(1) == 1) & ok.any(1)
    cam = ok.float().argmax(1)
    exp_uv = torch.gather(uv_n, 1, cam[:, None, :, None].expand(-1, 1, -1, 2))[:, 0]
    exp_d = torch.gather(depth, 1, cam[:, None]).squeeze(1)
    assert single.sum() > 50
    assert torch.allclose(vox[..., 0][single], exp_uv[..., 0][single], atol=1e-4)
    assert torch.allclose(vox[..., 1][single], exp_d[single], atol=1e-3)
    assert torch.allclose(vox[..., 2:][single], exp_uv[single], atol=1e-4)


def test_alpamayo_adapter_patchify(grid: GridSpec, cameras: Cameras) -> None:
    lifter = AlpamayoTriplaneAdapter(grid, 16, 8, num_layers=1)
    feats = [torch.randn(cameras.batch_size, cameras.num_cameras, 16, 8, 16)]
    with torch.no_grad():
        planes = lifter.lift_planes(feats, cameras)
        tokens = lifter.patchify_planes(planes, patch_size=2)
    nx, ny, nz = grid.resolution
    assert tokens.shape == (cameras.batch_size, (nx * ny + nx * nz + ny * nz) // 4, 8 * 4)


def test_alpamayo_planes_are_resampled_onto_the_uniform_grid() -> None:
    """Native planes are non-uniform; lift_planes must index the grid's uniform cells."""
    grid = GridSpec(resolution=(16, 16, 4))
    adapter = AlpamayoTriplaneAdapter(grid, 4, 4, num_layers=1)
    centres = [adapter.core.references.parametrizer(a).cell_centers_m() for a in range(3)]
    nx, ny, nz = grid.resolution
    # A "ramp" native plane: every cell stores its own metric x (rows) coordinate.
    native = TPVPlanes(
        centres[0].view(1, 1, nx, 1).expand(1, 1, nx, ny).clone(),
        centres[0].view(1, 1, nx, 1).expand(1, 1, nx, nz).clone(),
        centres[1].view(1, 1, ny, 1).expand(1, 1, ny, nz).clone(),
    )
    uniform = adapter.to_uniform(native)
    ux, uy = grid.axis_centers(0), grid.axis_centers(1)
    p = adapter.core.references.parametrizer(0)
    inner = ux.abs() < p.inner_radius_m - p.cell_edges_m().diff().max()  # away from the tails
    # Linear inside the inner zone: resampling recovers the uniform centres exactly there.
    assert torch.allclose(uniform.xy[0, 0, :, 0][inner], ux[inner], atol=1e-4)
    assert torch.allclose(uniform.yz[0, 0, :, 0][inner], uy[inner], atol=1e-4)
    # Everywhere: within a fraction of a cell (piecewise-linear interpolation of the tails).
    assert (uniform.xy[0, 0, :, 0] - ux).abs().max() < 0.5 * grid.voxel_size[0]
    # A uniform layout (inner_frac=1) makes the resampling the identity.
    flat = AlpamayoTriplaneAdapter(grid, 4, 4, num_layers=1, inner_frac=1.0)
    planes = TPVPlanes(
        torch.randn(2, 4, nx, ny), torch.randn(2, 4, nx, nz), torch.randn(2, 4, ny, nz)
    )
    out = flat.to_uniform(planes)
    assert all(
        torch.allclose(a, b, atol=1e-5)
        for a, b in zip((out.xy, out.xz, out.yz), (planes.xy, planes.xz, planes.yz), strict=True)
    )
