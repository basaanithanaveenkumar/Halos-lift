"""
HaleBackboneEncoder — bridge from hale_core vision backbones to Halos-lift's ImageEncoder.

Why this adapter exists
------------------------
Halos-lift's ImageEncoder contract:
  forward(images: Tensor [B*N, 3, H, W]) -> list[Tensor]  (multi-scale feature maps)

hale_core.nn.vision backbones produce BackboneOutput which carries feature_maps
with the same list structure.  This thin adapter:
  1. Normalises images (same ImageNet stats as the base class).
  2. Calls the hale_core backbone.
  3. Returns only the feature map levels that Halos-lift needs.

Usage
-----
In your config YAML:

    encoder:
      type: hale_backbone
      backbone_name: convnext_base     # any registered backbone
      img_size: 224
      out_channels: 128                # FPN output width
      num_levels: 3                    # how many scales (finest 3 = strides 8,16,32)
      freeze_encoder: false
      freeze_up_to_stage: -1

Or in Python:

    encoder = HaleBackboneEncoder(
        backbone_name="swinv2_tiny",
        out_channels=128,
        num_levels=3,
        img_size=224,
    )
    feature_list = encoder(images)   # list of [B*N, 128, H_i, W_i]

Architecture
------------
  Input images (unnormalised [0,1])
       ↓
  ImageNet normalisation (inherited from ImageEncoder base)
       ↓
  hale_core backbone (ConvNeXt / SwinV2 / ViT / etc.)
       ↓  BackboneOutput.feature_maps  [finest → coarsest]
  FPN neck (optional, matches out_channels)
       ↓
  list of num_levels feature tensors

For ViT-based backbones (single-scale), the single feature map is tiled to
fill all requested levels (coarse → finer via upsampling) or repeated.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lifting.encoders.base import ImageEncoder
from lifting.encoders.fpn import FPN
from lifting.exceptions import InvalidConfigError


class HaleBackboneEncoder(ImageEncoder):
    """Wraps any hale_core registered vision backbone as a Halos-lift ImageEncoder.

    Args:
        backbone_name:
            Name registered in hale_core.nn.vision.registry.
            Examples: "convnext_base", "swinv2_tiny", "resnet50", "vit_large".

        img_size:
            Input spatial resolution forwarded to the backbone constructor.

        out_channels:
            Channel width of each returned feature map level (FPN output).

        num_levels:
            How many feature pyramid levels to return.  Levels correspond to
            the coarsest feature_maps (high-stride features).
            Max is backbone.feature_dims count.

        freeze_encoder:
            If True, backbone parameters are frozen (no gradient).

        freeze_up_to_stage:
            If >= 0, freeze backbone stages 0..N.  -1 = no stage freezing.

        drop_rate:
            Dropout on backbone's pooled embedding (does not affect feature maps).
    """

    def __init__(
        self,
        backbone_name:       str   = "convnext_base",
        img_size:            int   = 224,
        out_channels:        int   = 128,
        num_levels:          int   = 3,
        freeze_encoder:      bool  = False,
        freeze_up_to_stage:  int   = -1,
        drop_rate:           float = 0.0,
    ) -> None:
        super().__init__()

        # ── Build backbone ────────────────────────────────────────────────────
        # Lazy import so Halos-lift does not depend on hale_core at import time
        # unless this encoder is actually instantiated.
        from hale_core.nn.vision import build_vision_backbone

        self.backbone = build_vision_backbone(
            backbone_name,
            img_size=img_size,
            in_chans=3,
            drop_rate=drop_rate,
        )
        self.backbone_name = backbone_name

        # Apply freezing
        if freeze_encoder:
            self.backbone.freeze_all()
        elif freeze_up_to_stage >= 0:
            self.backbone.freeze_up_to_stage(freeze_up_to_stage)

        # ── Resolve feature channel counts ────────────────────────────────────
        # feature_dims: list[int] finest→coarsest, set by each backbone class.
        backbone_feature_dims = self.backbone.feature_dims
        n_available = len(backbone_feature_dims)

        if num_levels > n_available:
            raise InvalidConfigError(
                f"backbone '{backbone_name}' has only {n_available} feature levels, "
                f"but num_levels={num_levels} was requested"
            )

        # Use the *last* num_levels levels (coarsest strides — most semantically rich)
        self._selected_dims = backbone_feature_dims[-num_levels:]
        self._n_levels      = num_levels
        self._n_available   = n_available

        # ── FPN neck to unify channel widths ─────────────────────────────────
        # If backbone has a single scale (ViT), replicate it to num_levels
        # first via bilinear up/down-sampling, then pass through FPN.
        self._is_single_scale = n_available == 1

        if self._is_single_scale:
            # ViT produces one scale; we map it to num_levels via 1×1 convs
            single_dim = backbone_feature_dims[0]
            self.scale_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(single_dim, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                for _ in range(num_levels)
            ])
        else:
            # Multi-scale backbone → FPN merges lateral connections
            self.neck = FPN(self._selected_dims, out_channels)

        self.out_channels = out_channels
        # Strides: assume standard ResNet-style [8, 16, 32, 64, ...] starting from stride 8
        self.strides = tuple(8 * 2**i for i in range(num_levels))

    # ── Core interface ────────────────────────────────────────────────────────

    def extract(self, images: Tensor) -> list[Tensor]:
        """Extract multi-scale features from normalised images.

        Args:
            images: [B*N, 3, H, W] normalised float tensors.

        Returns:
            List of num_levels feature tensors [B*N, out_channels, H_i, W_i],
            ordered finest → coarsest (matching strides 8, 16, 32, ...).
        """
        out = self.backbone(images)

        if self._is_single_scale:
            # ViT gives one spatial feature map [B, D, H/p, W/p]; create a pyramid
            # by halving resolution progressively (mimic an FPN's top-down path).
            single_feat = out.feature_maps[0]  # [B, D, H/p, W/p]
            levels: list[Tensor] = []
            for i, adapter in enumerate(self.scale_adapters):
                # Scale factor: finest level (i=0) keeps original resolution,
                # coarser levels are downsampled by 2^i.
                if i == 0:
                    feat = single_feat
                else:
                    scale = 0.5 ** i
                    feat = F.interpolate(
                        single_feat, scale_factor=scale, mode="bilinear",
                        align_corners=False, recompute_scale_factor=False,
                    )
                levels.append(adapter(feat))
            return levels
        else:
            # Multi-scale backbone: select the last num_levels feature maps.
            selected = out.feature_maps[-self._n_levels:]
            return self.neck(selected)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count total parameters (backbone + neck)."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        return (
            f"HaleBackboneEncoder("
            f"backbone={self.backbone_name}, "
            f"out_channels={self.out_channels}, "
            f"num_levels={self._n_levels}, "
            f"strides={self.strides}, "
            f"params={self.num_parameters():,})"
        )


def build_hale_backbone_encoder(
    backbone_name:      str   = "convnext_base",
    img_size:           int   = 224,
    out_channels:       int   = 128,
    num_levels:         int   = 3,
    freeze_encoder:     bool  = False,
    freeze_up_to_stage: int   = -1,
) -> HaleBackboneEncoder:
    """Factory function for use in Halos-lift model builders.

    Example::

        from lifting.encoders.hale_backbone_encoder import build_hale_backbone_encoder

        encoder = build_hale_backbone_encoder("swinv2_tiny", out_channels=256)
        feats = encoder(images)   # list of 3 feature maps
    """
    return HaleBackboneEncoder(
        backbone_name=backbone_name,
        img_size=img_size,
        out_channels=out_channels,
        num_levels=num_levels,
        freeze_encoder=freeze_encoder,
        freeze_up_to_stage=freeze_up_to_stage,
    )
