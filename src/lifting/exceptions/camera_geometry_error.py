"""Camera tensors with inconsistent shapes or camera counts."""

from __future__ import annotations

from lifting.exceptions.lifting_error import LiftingError


class CameraGeometryError(LiftingError, ValueError):
    """Intrinsics / extrinsics / image batches disagree on shape or number of cameras.

    Also a :class:`ValueError` for compatibility with existing call sites.
    """
