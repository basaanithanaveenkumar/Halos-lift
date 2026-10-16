"""A component was configured with invalid hyper-parameters."""

from __future__ import annotations

from lifting.exceptions.lifting_error import LiftingError


class InvalidConfigError(LiftingError, ValueError):
    """Invalid constructor arguments or option values (e.g. ``embed_dims`` not divisible by
    ``num_heads``, an unknown ``reduce`` mode).

    Also a :class:`ValueError`, so callers written against the original exceptions still match.
    """
