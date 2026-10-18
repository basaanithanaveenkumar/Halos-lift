"""Image encoders producing multi-scale feature pyramids.

Encoders
--------
SimpleConvEncoder  — lightweight strided-conv encoder for quick experiments
ResNetEncoder      — torchvision ResNet + FPN (requires torchvision)
HaleBackboneEncoder — any hale_core registered backbone (ConvNeXt, SwinV2, ViT,
                       ResNet, MobileOne, EfficientNet, InternImage, CoAtNet,
                       MaxViT, DINOv2, EVA-02, DeiT III, etc.) + optional FPN neck
"""

from typing import Any

from lifting.encoders.base import ImageEncoder
from lifting.encoders.encoder_registry import ENCODERS
from lifting.encoders.fpn import FPN
from lifting.encoders.resnet_encoder import ResNetEncoder
from lifting.encoders.simple_conv_encoder import SimpleConvEncoder


# HaleBackboneEncoder wraps hale_core vision backbones. It is imported lazily so the package
# imports without hale_core; the module loads only when the encoder is requested.
def build_hale_backbone_encoder(**kwargs: Any) -> ImageEncoder:
    """Build a HaleBackboneEncoder — see lifting.encoders.hale_backbone_encoder."""
    from lifting.encoders.hale_backbone_encoder import build_hale_backbone_encoder as _build

    return _build(**kwargs)


__all__ = [
    "ENCODERS",
    "FPN",
    "ImageEncoder",
    "ResNetEncoder",
    "SimpleConvEncoder",
    "build_hale_backbone_encoder",
]
