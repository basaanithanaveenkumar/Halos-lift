"""An optional dependency needed by the requested feature is not installed."""

from __future__ import annotations

from lifting.exceptions.lifting_error import LiftingError


class MissingDependencyError(LiftingError, ImportError):
    """Raised instead of a bare :class:`ImportError` when an optional extra is missing.

    Args:
        package: import name that failed (e.g. ``"torchvision"``).
        extra: the ``halos-lift`` extra that provides it (e.g. ``"vision"``).
    """

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(f"{package!r} is not installed; run `pip install 'halos-lift[{extra}]'`")
