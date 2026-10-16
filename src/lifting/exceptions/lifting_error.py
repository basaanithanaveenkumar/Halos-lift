"""Root of the Halos-lift exception hierarchy."""

from __future__ import annotations


class LiftingError(Exception):
    """Base class of every error ``lifting`` raises on purpose.

    Catch ``LiftingError`` to separate expected, actionable failures (bad configuration,
    inconsistent camera tensors, empty data) from programming errors. It derives from
    :class:`Exception`, not from HaleBlocks' ``HaleCoreError``: Halos-lift depends only on
    torch, numpy and pillow, and HaleBlocks is an optional extra.
    """
