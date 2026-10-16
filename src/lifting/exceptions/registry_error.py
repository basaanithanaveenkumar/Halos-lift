"""Unknown or duplicate registry names."""

from __future__ import annotations

from lifting.exceptions.lifting_error import LiftingError


class RegistryError(LiftingError, KeyError):
    """A registry lookup failed (unknown name) or a name was registered twice.

    Also a :class:`KeyError`, the exception the original registry raised.
    """

    def __str__(self) -> str:  # KeyError quotes its message; keep it readable
        return str(self.args[0]) if self.args else ""
