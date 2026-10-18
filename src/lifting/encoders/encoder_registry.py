"""The :data:`ENCODERS` registry: image encoders selectable by name from a config."""

from __future__ import annotations

from typing import Any

from lifting.encoders.base import ImageEncoder
from lifting.encoders.resnet_encoder import ResNetEncoder
from lifting.encoders.simple_conv_encoder import SimpleConvEncoder
from lifting.registry import Registry

#: Pattern: Registry / Factory. ``ENCODERS.build({"type": "resnet", "arch": "resnet18"})``.
ENCODERS: Registry[ImageEncoder] = Registry("encoder")

ENCODERS.register("simple_conv")(SimpleConvEncoder)
ENCODERS.register("resnet")(ResNetEncoder)


@ENCODERS.register("hale_backbone")
def _build_hale_backbone(**kwargs: Any) -> ImageEncoder:
    """Build a ``hale_core`` backbone encoder; hale_core is imported only when requested."""
    from lifting.encoders.hale_backbone_encoder import build_hale_backbone_encoder

    return build_hale_backbone_encoder(**kwargs)
