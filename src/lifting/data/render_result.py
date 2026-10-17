"""Images and depth maps produced by :class:`RayCastRenderer`."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class RenderResult:
    """Rendered views.

    Attributes:
        images: ``(N, 3, H, W)`` RGB in ``[0, 1]``.
        depth: ``(N, H, W)`` camera-frame z of the first hit (``inf`` for sky).
        labels: ``(N, H, W)`` semantic id per pixel (0 = ground / sky).
    """

    images: Tensor
    depth: Tensor
    labels: Tensor
