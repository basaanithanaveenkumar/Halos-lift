"""Triplane (TPV) lifting for every lifter. Importing this package registers all of them.

Every BEV lifter in :data:`lifting.lifters.LIFTERS` has a triplane counterpart with the
**same name** in :data:`TRIPLANE_LIFTERS`. A triplane lifter returns three orthogonal planes
(``xy`` = BEV, ``xz``, ``yz``) from :meth:`~BaseTriplaneLifter.lift_planes` and is still a
BEV lifter through ``forward``. The existing lifters are reused unchanged (composition).

| name          | class                       | how the three planes are produced                         |
|---------------|-----------------------------|-----------------------------------------------------------|
| ``simple_bev``| SimpleBEVTriplaneLifter     | bilinear voxel unprojection -> collapse Z / Y / X         |
| ``depth_warp``| DepthWarpTriplaneLifter     | depth-weighted frustum pulled per voxel -> collapse       |
| ``lift_splat``| LiftSplatTriplaneLifter     | depth x context point cloud splatted -> collapse          |
| ``tiim``      | PolarRayTriplaneLifter      | polar rays (range) + image rows (height) per voxel        |
| ``bevformer`` | BEVFormerTriplaneLifter     | one BEVFormer query encoder per plane                     |
| ``tpvformer`` | TPVFormerTriplaneLifter     | TPVFormer (planes + cross-view hybrid attention)          |
| ``alpamayo``  | AlpamayoTriplaneAdapter     | TPVFormer layers on Mip-NeRF 360 non-uniform planes       |
| ``mlp_view``  | MLPViewTriplaneLifter       | geometry-free MLPs per (camera, plane): negative control  |
| ``bev_adapter``| BEVLifterTriplaneAdapter   | any ``LIFTERS`` entry -> BEV -> learned BEVToTPV          |

Example::

    from lifting.triplane import TRIPLANE_LIFTERS, build_triplane_model
    lifter = TRIPLANE_LIFTERS.build("lift_splat", grid=grid, in_channels=64, out_channels=64)
    planes = lifter.lift_planes(features, cameras)          # TPVPlanes(xy, xz, yz)
    bev = lifter(features, cameras)                         # (B, 64, X, Y)
    model = build_triplane_model("bevformer", grid, num_classes=3)
"""

from lifting.triplane.alpamayo_triplane import AlpamayoTriplaneAdapter
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.bev_lifter_adapter import BEVLifterTriplaneAdapter
from lifting.triplane.bevformer_triplane import BEVFormerTriplaneLifter
from lifting.triplane.depth_warp_triplane import DepthWarpTriplaneLifter
from lifting.triplane.factory import (
    build_triplane_bev_model,
    build_triplane_lifter,
    build_triplane_model,
)
from lifting.triplane.lift_splat_triplane import LiftSplatTriplaneLifter
from lifting.triplane.mlp_view_triplane import MLPViewTriplaneLifter
from lifting.triplane.plane_compressor import VoxelToTriplane
from lifting.triplane.plane_projection import PlaneProjection
from lifting.triplane.plane_query_encoder import PlaneQueryEncoder
from lifting.triplane.polar_ray_triplane import PolarRayTriplaneLifter
from lifting.triplane.simple_bev_triplane import SimpleBEVTriplaneLifter
from lifting.triplane.tpvformer_triplane import TPVFormerTriplaneLifter
from lifting.triplane.triplane_bev_head import TriplaneBEVHead
from lifting.triplane.triplane_occupancy_model import TriplaneOccupancyModel

__all__ = [
    "TRIPLANE_LIFTERS",
    "AlpamayoTriplaneAdapter",
    "BEVFormerTriplaneLifter",
    "BEVLifterTriplaneAdapter",
    "BaseTriplaneLifter",
    "DepthWarpTriplaneLifter",
    "LiftSplatTriplaneLifter",
    "MLPViewTriplaneLifter",
    "PlaneProjection",
    "PlaneQueryEncoder",
    "PolarRayTriplaneLifter",
    "SimpleBEVTriplaneLifter",
    "TPVFormerTriplaneLifter",
    "TriplaneBEVHead",
    "TriplaneOccupancyModel",
    "VoxelToTriplane",
    "build_triplane_bev_model",
    "build_triplane_lifter",
    "build_triplane_model",
]
