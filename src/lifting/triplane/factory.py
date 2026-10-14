"""One-line constructors for triplane lifters and complete triplane models."""

from __future__ import annotations

from typing import Any

from lifting.decoders import BEVDecoder, SegmentationHead
from lifting.encoders import ImageEncoder, SimpleConvEncoder
from lifting.geometry import GridSpec
from lifting.lifters.tpvformer.plane_aux_head import PlaneAuxHead
from lifting.lifters.tpvformer.tpv_aggregator import TPVAggregator
from lifting.models.bev_segmentation_model import BEVSegmentationModel

# Reuse the BEV factory's context helpers so BEV and TPV lifters are sized identically.
from lifting.models.factory import _depth_bins_for, _feature_size
from lifting.triplane.base import TRIPLANE_LIFTERS, BaseTriplaneLifter
from lifting.triplane.triplane_occupancy_model import TriplaneOccupancyModel


def build_triplane_lifter(
    name: str,
    grid: GridSpec,
    channels: int,
    image_size: tuple[int, int],
    num_cameras: int,
    encoder: ImageEncoder,
    **kwargs: Any,
) -> BaseTriplaneLifter:
    """Build a registered triplane lifter, injecting the context it asks for.

    Mirrors :func:`lifting.models.build_lifter`: ``feature_size``, ``num_cameras`` and
    ``depth_bins`` are derived from the encoder / rig / grid unless given in ``kwargs``.
    """
    cls = TRIPLANE_LIFTERS.get(name)
    level = kwargs.get("level", dict(kwargs.get("base_kwargs") or {}).get("level", -1))
    context = {
        "feature_size": _feature_size(image_size, encoder.strides[level]),
        "num_cameras": num_cameras,
        "depth_bins": _depth_bins_for(grid),
    }
    kwargs = cls.resolve_kwargs(kwargs, context)
    return cls(grid=grid, in_channels=channels, out_channels=channels, **kwargs)


def build_triplane_model(
    lifter: str = "tpvformer",
    grid: GridSpec | None = None,
    num_classes: int = 3,
    image_size: tuple[int, int] = (48, 96),
    num_cameras: int = 4,
    channels: int = 64,
    encoder: ImageEncoder | None = None,
    scale: int = 1,
    aux_heads: bool = False,
    **lifter_kwargs: Any,
) -> TriplaneOccupancyModel:
    """3D semantic occupancy (+ point queries) with ANY triplane lifter.

    Example::

        model = build_triplane_model("lift_splat", GridSpec(), num_classes=3)
        out = model(images, cameras, points=lidar)  # planes, voxel_logits, point_logits
    """
    grid = grid or GridSpec()
    encoder = encoder or SimpleConvEncoder(out_channels=channels, num_levels=2)
    c = encoder.out_channels
    lift = build_triplane_lifter(lifter, grid, c, image_size, num_cameras, encoder, **lifter_kwargs)
    aggregator = TPVAggregator(grid, c, num_classes, scale=scale)
    heads = [PlaneAuxHead(c, num_classes) for _ in range(3)] if aux_heads else None
    return TriplaneOccupancyModel(encoder, lift, aggregator, heads)


def build_triplane_bev_model(
    lifter: str = "tpvformer",
    grid: GridSpec | None = None,
    num_classes: int = 3,
    image_size: tuple[int, int] = (48, 96),
    num_cameras: int = 4,
    channels: int = 64,
    encoder: ImageEncoder | None = None,
    **lifter_kwargs: Any,
) -> BEVSegmentationModel:
    """BEV segmentation where the BEV map is read off the triplane (``lifter.forward``).

    Same signature and output as :func:`lifting.build_bev_model`, so triplane and BEV lifters
    can be benchmarked side by side with the existing trainer / verifier.
    """
    grid = grid or GridSpec()
    encoder = encoder or SimpleConvEncoder(out_channels=channels, num_levels=2)
    c = encoder.out_channels
    lift = build_triplane_lifter(lifter, grid, c, image_size, num_cameras, encoder, **lifter_kwargs)
    return BEVSegmentationModel(
        encoder, lift, BEVDecoder(c, channels), SegmentationHead(channels, num_classes)
    )
