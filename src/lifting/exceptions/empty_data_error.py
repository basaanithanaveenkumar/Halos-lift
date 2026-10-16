"""A training or evaluation input produced no data."""

from __future__ import annotations

from lifting.exceptions.lifting_error import LiftingError


class EmptyDataError(LiftingError, ValueError):
    """An iterable of batches, a loader or a dataset mixture turned out to be empty.

    Also a :class:`ValueError` for compatibility with existing call sites.
    """
