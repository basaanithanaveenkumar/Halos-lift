"""Reference points for Alpamayo triplane attention using Mip-NeRF 360 non-linear coords."""

from __future__ import annotations

import torch
from torch import Tensor

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.alpamayo.mipnerf360 import MipNeRF360Parametrizer
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes

#: The axis orthogonal to each plane, in ``xy, xz, yz`` order.
_NORMAL_AXIS = (2, 1, 0)


class AlpamayoReferencePoints:
    """Reference points for Alpamayo triplane attention using Mip-NeRF 360 coords.

    Structurally identical to :class:`TPVReferencePoints` but the metric positions of
    query and anchor points are determined by the Mip-NeRF 360 non-linear parametrizer
    instead of a uniform grid.  This places denser queries near the ego vehicle.

    Args:
        grid: 3D region; plane resolutions follow ``grid.resolution``.
        num_anchors: anchors per query along the plane normal per plane (xy, xz, yz).
        inner_frac: fraction of cells in the inner high-resolution zone (default 0.75).
    """

    def __init__(
        self,
        grid: GridSpec,
        num_anchors: tuple[int, int, int] = (4, 8, 8),
        inner_frac: float = 0.75,
    ) -> None:
        self.grid = grid
        self.num_anchors = num_anchors
        self.inner_frac = inner_frac
        # One parametrizer per spatial axis (x, y, z). ``center_m`` keeps the cells inside the
        # grid bounds for axes that are not symmetric about 0 (the default z range is [-1, 3]).
        self._params = [
            MipNeRF360Parametrizer(
                total=grid.resolution[i],
                inner_frac=inner_frac,
                total_extent_m=float(grid.bounds[i][1] - grid.bounds[i][0]),
                center_m=float(grid.bounds[i][0] + grid.bounds[i][1]) / 2.0,
            )
            for i in range(3)
        ]

    def parametrizer(self, axis: int) -> MipNeRF360Parametrizer:
        """The non-linear parametrizer of ``axis`` (0 = x, 1 = y, 2 = z)."""
        return self._params[axis]

    @property
    def plane_shapes(self) -> list[tuple[int, int]]:
        return TPVPlanes.plane_shapes(self.grid.resolution)

    def _axis_centers(self, axis: int, device: torch.device | str | None = None) -> Tensor:
        """Non-linear metric cell centres along ``axis``."""
        return self._params[axis].cell_centers_m(device)

    def anchor_points(
        self, plane: int, device: torch.device | str | None = None
    ) -> Tensor:
        """``(Q_plane, A, 3)`` metric anchors for every query on ``plane``.

        The two in-plane axes use non-linear cell centres; anchors along the normal
        axis are uniformly sampled across the full extent so the pillar structure is
        preserved (avoids bunching anchors in the inner zone of the normal axis only).
        """
        normal = _NORMAL_AXIS[plane]
        a, b = (i for i in range(3) if i != normal)

        coords_a = self._axis_centers(a, device)  # (N_a,)
        coords_b = self._axis_centers(b, device)  # (N_b,)

        # Anchor points are uniformly spread along the normal axis
        lo, hi = self.grid.bounds[normal]
        num_a = self.num_anchors[plane]
        step = (hi - lo) / num_a
        coords_normal = torch.arange(num_a, device=device, dtype=torch.float32) * step + lo + step / 2.0

        # Build mesh: (N_a, N_b, A, 3)
        mesh_ab = torch.stack(
            torch.meshgrid(coords_a, coords_b, indexing="ij"), dim=-1
        )  # (N_a, N_b, 2)

        q = mesh_ab.shape[0] * mesh_ab.shape[1]
        # Expand anchors: (N_a, N_b, A, 3)
        pts = torch.zeros(mesh_ab.shape[0], mesh_ab.shape[1], num_a, 3, device=device)
        pts[..., a] = mesh_ab[..., 0:1].expand(-1, -1, num_a)
        pts[..., b] = mesh_ab[..., 1:2].expand(-1, -1, num_a)
        pts[..., normal] = coords_normal[None, None, :]

        return pts.reshape(q, num_a, 3)

    def camera_reference_points(
        self, cameras: Cameras
    ) -> list[tuple[Tensor, Tensor]]:
        """Per plane: normalised projections ``(B, N, Q, A, 2)`` and visibility ``(B, N, Q, A)``.

        Same interface as :class:`TPVReferencePoints.camera_reference_points`.
        """
        b, n = cameras.batch_size, cameras.num_cameras
        out = []
        for plane in range(3):
            anchors = self.anchor_points(plane, cameras.device)  # (Q, A, 3)
            q, a, _ = anchors.shape
            uv, _, valid = cameras.project(anchors.view(1, -1, 3).expand(b, -1, -1))
            out.append((cameras.normalize_uv(uv).view(b, n, q, a, 2), valid.view(b, n, q, a)))
        return out

    def hybrid_reference_points(
        self, device: torch.device | str | None = None
    ) -> Tensor:
        """``(Q_total, 3, 2)`` cross-view attention reference points.

        Each query points at the same 3D location on each of the three planes.
        Uses normalised [0, 1] cell coordinates (independent of the non-linear spacing),
        so it matches the :class:`TPVFormerLayer` hybrid attention interface exactly.
        """
        nx, ny, nz = self.grid.resolution

        def norm_centers(n: int, device: torch.device | str | None) -> Tensor:
            return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n

        half = torch.tensor(0.5, device=device)
        xyz_per_plane = []
        for (h, w), normal in zip(self.plane_shapes, _NORMAL_AXIS, strict=True):
            r, c = torch.meshgrid(norm_centers(h, device), norm_centers(w, device), indexing="ij")
            r, c = r.reshape(-1), c.reshape(-1)
            axes = [i for i in range(3) if i != normal]
            coords = [half.expand_as(r)] * 3
            coords[axes[0]], coords[axes[1]] = r, c
            xyz_per_plane.append(torch.stack(coords, -1))

        xyz = torch.cat(xyz_per_plane, 0)  # (Q_total, 3) in [0, 1]
        x, y, z = xyz.unbind(-1)
        return torch.stack(
            [torch.stack([y, x], -1), torch.stack([z, x], -1), torch.stack([z, y], -1)], dim=1
        )
