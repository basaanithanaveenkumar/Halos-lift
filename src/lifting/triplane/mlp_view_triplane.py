"""Geometry-free MLP view transform -> TPV (the negative control for triplane lifting)."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.layers import ConvNormAct
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter


@TRIPLANE_LIFTERS.register("mlp_view")
class MLPViewTriplaneLifter(BaseTriplaneLifter):
    """VPN-style view parsing, one fully connected map per (camera, plane).

    Every camera's ``h x w`` feature cells are mapped by a ``Linear(h*w, H_p*W_p)`` to the cells
    of each plane ``p`` and summed over cameras. Calibration is **ignored** (``uses_geometry =
    False``), so the planes cannot follow a rotated rig: compare any geometric triplane lifter
    against this one to prove it really uses the cameras.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        feature_size: ``(h, w)`` of the pyramid level used.
        num_cameras: number of cameras (one MLP per camera and plane).
        level: pyramid level (default: coarsest).
    """

    uses_geometry = False

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        feature_size: tuple[int, int] = (3, 6),
        num_cameras: int = 4,
        level: int = -1,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.level = level
        self.num_cameras = num_cameras
        cells = feature_size[0] * feature_size[1]
        self.plane_shapes = TPVPlanes.plane_shapes(grid.resolution)
        self.view_mlps = nn.ModuleList(
            nn.ModuleList(nn.Linear(cells, h * w) for _ in range(num_cameras))
            for h, w in self.plane_shapes
        )
        self.outputs = nn.ModuleList(
            nn.Sequential(
                ConvNormAct(in_channels, out_channels, 3), nn.Conv2d(out_channels, out_channels, 1)
            )
            for _ in self.plane_shapes
        )

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        feats = features[self.level]  # B N C h w
        b, n, c = feats.shape[:3]
        if n != self.num_cameras:
            raise ValueError(
                f"MLPViewTriplaneLifter was built for {self.num_cameras} cameras, got {n}"
            )
        flat = feats.flatten(3)
        planes = []
        for (h, w), mlps, output in zip(
            self.plane_shapes, self.view_mlps, self.outputs, strict=True
        ):
            plane = sum(mlp(flat[:, i]) for i, mlp in enumerate(mlps))
            planes.append(output(plane.view(b, c, h, w)))
        return TPVPlanes(*planes)
