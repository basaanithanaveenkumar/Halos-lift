"""Tri-perspective-view (TPV) feature planes, with an optional front plane."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

#: Plane names in their canonical order (top / BEV, side, front).
PLANE_NAMES: tuple[str, str, str] = ("xy", "xz", "yz")


@dataclass
class TPVPlanes:
    """Orthogonal feature planes describing a 3D scene (Huang et al., *TPVFormer*, 2023).

    The top (``xy``) and side (``xz``) planes are always present. The front plane (``yz``) is
    optional: two-plane lifters such as ``bev_side`` leave it ``None``, and every helper below
    (voxel broadcast, point sampling, upsampling, tokenisation) then uses only the planes that
    exist. With all three planes the behaviour is exactly the original TPV sum.

    Attributes:
        xy: ``(B, C, X, Y)`` top (BEV) plane.
        xz: ``(B, C, X, Z)`` side plane.
        yz: ``(B, C, Y, Z)`` front plane, or ``None`` for a two-plane (BEV + side) TPV.
    """

    xy: Tensor
    xz: Tensor
    yz: Tensor | None = None

    # ------------------------------------------------------------------ structure
    @property
    def names(self) -> tuple[str, ...]:
        """Names of the planes that are present, in canonical order."""
        return tuple(name for name in PLANE_NAMES if getattr(self, name) is not None)

    @property
    def has_front(self) -> bool:
        """Whether the ``yz`` (front) plane is present."""
        return self.yz is not None

    def items(self) -> Iterator[tuple[str, Tensor]]:
        """``(name, plane)`` for every present plane, in canonical order."""
        for name in self.names:
            yield name, getattr(self, name)

    def tensors(self) -> tuple[Tensor, ...]:
        """The present planes, in canonical order."""
        return tuple(plane for _, plane in self.items())

    @property
    def resolution(self) -> tuple[int, int, int]:
        """Voxel resolution ``(X, Y, Z)`` the planes describe."""
        return self.xy.shape[2], self.xy.shape[3], self.xz.shape[3]

    @staticmethod
    def plane_shapes(
        resolution: tuple[int, int, int], names: Sequence[str] = PLANE_NAMES
    ) -> list[tuple[int, int]]:
        """Spatial shape ``(H, W)`` of each plane in ``names`` (default ``xy``, ``xz``, ``yz``)."""
        nx, ny, nz = resolution
        shapes = {"xy": (nx, ny), "xz": (nx, nz), "yz": (ny, nz)}
        return [shapes[name] for name in names]

    # ------------------------------------------------------------------ tokens
    @classmethod
    def from_tokens(
        cls, tokens: Tensor, resolution: tuple[int, int, int], names: Sequence[str] = PLANE_NAMES
    ) -> TPVPlanes:
        """Split ``(B, sum(H*W), C)`` tokens of the planes ``names`` back into planes."""
        shapes = cls.plane_shapes(resolution, names)
        parts = tokens.split([h * w for h, w in shapes], dim=1)
        b, _, c = tokens.shape
        planes = {
            name: p.transpose(1, 2).reshape(b, c, h, w)
            for name, p, (h, w) in zip(names, parts, shapes, strict=True)
        }
        return cls(**planes)

    def to_tokens(self) -> Tensor:
        """Concatenate the present planes into one ``(B, sum(H*W), C)`` token sequence."""
        return torch.cat([p.flatten(2).transpose(1, 2) for p in self.tensors()], dim=1)

    # ------------------------------------------------------------------ geometry
    def map(self, fn) -> TPVPlanes:
        """Apply ``fn`` to every present plane (missing planes stay ``None``)."""
        return TPVPlanes(**{name: fn(plane) for name, plane in self.items()})

    def upsample(self, scale: int) -> TPVPlanes:
        """Return planes bilinearly enlarged by the integer factor ``scale``."""
        if scale == 1:
            return self
        return self.map(
            lambda p: F.interpolate(p, scale_factor=scale, mode="bilinear", align_corners=False)
        )

    def to_voxels(self) -> Tensor:
        """Broadcast-sum the planes into a ``(B, C, X, Y, Z)`` volume.

        ``f = xy + xz + yz`` with a front plane, ``f = xy + xz`` without one.
        """
        voxels = self.xy[..., None] + self.xz[:, :, :, None, :]
        if self.yz is not None:
            voxels = voxels + self.yz[:, :, None, :, :]
        return voxels

    def sample_points(self, points_normalized: Tensor) -> Tensor:
        """Features of arbitrary 3D points.

        Args:
            points_normalized: ``(B, P, 3)`` coordinates in ``[-1, 1]`` (see ``GridSpec.to_normalized``).

        Returns:
            ``(B, C, P)`` sum of the bilinear samples of every present plane.
        """
        x, y, z = points_normalized.unbind(-1)

        def sample(plane: Tensor, col: Tensor, row: Tensor) -> Tensor:
            grid = torch.stack([col, row], -1)[:, None]
            return F.grid_sample(plane, grid, mode="bilinear", align_corners=False).squeeze(2)

        out = sample(self.xy, y, x) + sample(self.xz, z, x)
        if self.yz is not None:
            out = out + sample(self.yz, z, y)
        return out
