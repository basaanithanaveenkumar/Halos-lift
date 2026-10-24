"""BEV + side: a two-plane TPV (no front view) built from one reused perspective volume."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from torch import Tensor, nn

from lifting.exceptions import InvalidConfigError
from lifting.geometry import Cameras, GridSpec
from lifting.layers import HeightCompressor
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.triplane.axis_attention_pool import AxisAttentionPool
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.bev_side_mixer import BEVSideMixer
from lifting.triplane.voxel_volume_source import VoxelVolumeSource

Pooling = Literal["attention", "fold"]


@TRIPLANE_LIFTERS.register("bev_side")
class BEVSideTriplaneLifter(BaseTriplaneLifter):
    """Top (``xy``) and side (``xz``) planes only - the front (``yz``) plane is dropped.

    1. **Perspective features, sampled once.** A voxel-volume lifter (``volume``:
       ``simple_bev``, ``depth_warp``, ``lift_splat`` or ``tiim``) reads the cameras and returns
       a ``(B, C, X, Y, Z)`` volume. This is the only place the images are touched.
    2. **Both planes reuse that volume.** The BEV plane pools it over height ``Z``, the side
       plane over the lateral axis ``Y`` (:class:`AxisAttentionPool`, or ``pooling="fold"``
       for the channel-folding compressor of ``simple_bev``). No plane samples the cameras
       again, unlike ``bevformer`` / ``tpvformer`` which run one image cross-attention per plane.
    3. **BEV <-> side mixing.** ``mix_layers`` :class:`BEVSideMixer` layers exchange
       information along the forward axis ``x`` the two planes share.

    The planes are returned as ``TPVPlanes(xy, xz, yz=None)``: voxel features are
    ``xy(x, y) + xz(x, z)``, so the BEV head, the occupancy aggregator and point queries work
    unchanged. Dropping the front plane removes the ``Y x Z`` plane and its compressor; the
    side plane still carries height, and the BEV plane carries the lateral layout, so every
    voxel is still addressed on all three axes.

    Args:
        grid, in_channels, out_channels: see :class:`BaseTriplaneLifter`.
        volume: name of the voxel-volume lifter in :data:`TRIPLANE_LIFTERS`.
        volume_kwargs: extra constructor arguments of the volume lifter (e.g. ``level``).
        pooling: ``attention`` (:class:`AxisAttentionPool`) or ``fold``.
        mix_layers: BEV <-> side mixing layers (0 = independent planes).
        num_heads: heads of the attention pooling and of the mixer.
        dropout: mixer attention dropout.
    """

    planes: ClassVar[tuple[str, ...]] = ("xy", "xz")

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        volume: str = "simple_bev",
        volume_kwargs: Mapping[str, Any] | None = None,
        pooling: Pooling = "attention",
        mix_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        if volume == "bev_side":
            raise InvalidConfigError("bev_side cannot use itself as the volume lifter")
        source = TRIPLANE_LIFTERS.build(
            volume,
            grid=grid,
            in_channels=in_channels,
            out_channels=in_channels,
            **dict(volume_kwargs or {}),
        )
        if not isinstance(source, VoxelVolumeSource):
            raise InvalidConfigError(
                f"volume={volume!r} does not expose a voxel volume (lift_voxels / voxel_channels); "
                "use simple_bev, depth_warp, lift_splat or tiim"
            )
        # Only the volume is reused: the source's own plane compressors and BEV head are never
        # called, so they are dropped to keep every parameter on the gradient path.
        source.to_planes = nn.Identity()
        source.bev_head = nn.Identity()
        self.volume_name = volume
        self.volume = source
        self.uses_geometry = source.uses_geometry
        channels = source.voxel_channels
        _, ny, nz = grid.resolution
        if pooling == "attention":
            self.pool_height = AxisAttentionPool(channels, nz, out_channels, num_heads)
            self.pool_lateral = AxisAttentionPool(channels, ny, out_channels, num_heads)
        elif pooling == "fold":
            self.pool_height = HeightCompressor(channels, nz, out_channels)
            self.pool_lateral = HeightCompressor(channels, ny, out_channels)
        else:
            raise InvalidConfigError(f"pooling must be 'attention' or 'fold', got {pooling!r}")
        self.pooling = pooling
        self.mixers = nn.ModuleList(
            BEVSideMixer(out_channels, grid.resolution, num_heads, dropout)
            for _ in range(mix_layers)
        )

    @property
    def level(self) -> int:
        """Feature-pyramid level the volume lifter reads."""
        return self.volume.level

    def lift_volume(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """``(B, C, X, Y, Z)`` perspective features sampled from the cameras (the shared step)."""
        return self.volume.lift_voxels(features, cameras)

    def planes_from_volume(self, voxels: Tensor) -> TPVPlanes:
        """BEV and side planes from an already-lifted volume (no camera access)."""
        bev = self.pool_height(voxels)  # collapse Z -> (B, C, X, Y)
        side = self.pool_lateral(voxels.permute(0, 1, 2, 4, 3))  # collapse Y -> (B, C, X, Z)
        for mixer in self.mixers:
            bev, side = mixer(bev, side)
        return TPVPlanes(xy=bev, xz=side)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        """Lift per-camera features into the BEV and side planes (``yz`` is ``None``).

        Args:
            features: feature pyramid from the image encoder.
            cameras: the rig geometry.

        Returns:
            ``TPVPlanes(xy, xz, yz=None)``.
        """
        return self.planes_from_volume(self.lift_volume(features, cameras))

    @classmethod
    def resolve_kwargs(cls, kwargs: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
        """Forward the build context (``depth_bins`` ...) to the *volume* lifter's constructor."""
        out = dict(kwargs)
        volume_cls = TRIPLANE_LIFTERS.get(out.get("volume", "simple_bev"))
        params = inspect.signature(volume_cls).parameters
        volume_kwargs = dict(out.get("volume_kwargs") or {})
        for key, value in context.items():
            if key in params and key not in volume_kwargs:
                volume_kwargs[key] = value
        out["volume_kwargs"] = volume_kwargs
        return out
