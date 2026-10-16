"""Name -> class registry so new components plug in without editing core code.

The API mirrors HaleBlocks' mmdet-style ``hale_core.registry.Registry`` (``register_module``,
``build(cfg)`` with a ``type`` key, parent fallback), so components move between the two code
bases unchanged, while Halos-lift keeps its torch/numpy/pillow-only dependency footprint.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar

from loguru import logger

from lifting.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps string names to classes (or factories). Pattern: Registry / Factory.

    Components register next to their definition; callers build them by name or from a
    config mapping, so adding a lifter never edits a factory ``if/elif`` chain (open/closed).

    Args:
        kind: what is registered (``"lifter"``, ``"storyboard"`` …), used in messages.
        parent: optional registry consulted when a name is not found here.

    Example::

        LIFTERS = Registry("lifter")

        @LIFTERS.register("my_lifter")
        class MyLifter(BaseLifter): ...

        lifter = LIFTERS.build("my_lifter", grid=grid, in_channels=64, out_channels=64)
        lifter = LIFTERS.build({"type": "my_lifter", "grid": grid, "in_channels": 64,
                                "out_channels": 64})
    """

    def __init__(self, kind: str, parent: Registry[T] | None = None) -> None:
        self.kind = kind
        self.parent = parent
        self._entries: dict[str, Callable[..., T]] = {}

    # ------------------------------------------------------------------ registration
    def register(self, name: str, force: bool = False) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator registering a class or factory under ``name``.

        Args:
            name: registry key.
            force: replace an existing entry instead of raising.

        Raises:
            RegistryError: if ``name`` is already registered and ``force`` is false.
        """

        def decorator(obj: Callable[..., T]) -> Callable[..., T]:
            self._add(name, obj, force)
            return obj

        return decorator

    def register_module(
        self,
        name: str | None = None,
        force: bool = False,
        module: Callable[..., T] | None = None,
    ) -> Any:
        """mmdet-style registration: decorator, or direct call with ``module=``.

        ``name`` defaults to the object's ``__name__``.

        Raises:
            RegistryError: on a duplicate name without ``force``.
        """
        if module is not None:
            self._add(name or module.__name__, module, force)
            return module

        def decorator(obj: Callable[..., T]) -> Callable[..., T]:
            self._add(name or obj.__name__, obj, force)
            return obj

        return decorator

    def _add(self, name: str, obj: Callable[..., T], force: bool) -> None:
        if name in self._entries and not force:
            raise RegistryError(f"{self.kind} {name!r} is already registered")
        self._entries[name] = obj
        logger.trace("registered {} {}", self.kind, name)

    # ------------------------------------------------------------------ lookup
    def get(self, name: str) -> Callable[..., T]:
        """Return the entry for ``name``, falling back to the parent registry.

        Raises:
            RegistryError: for an unknown name; the message lists close matches and every
                available name.
        """
        if name in self._entries:
            return self._entries[name]
        if self.parent is not None and name in self.parent:
            return self.parent.get(name)
        close = difflib.get_close_matches(name, self.names(), n=3)
        hint = f" Did you mean {close}?" if close else ""
        raise RegistryError(f"unknown {self.kind} {name!r}; available: {self.names()}.{hint}")

    def build(self, spec: str | Mapping[str, Any], **kwargs: Any) -> T:
        """Instantiate an entry by name, or from a config mapping with a ``type`` key.

        Keyword arguments act as defaults; keys in a config mapping override them. The mapping
        is copied, never mutated.

        Args:
            spec: registry name, or ``{"type": name, **constructor_kwargs}``.
            **kwargs: constructor keyword arguments.

        Raises:
            RegistryError: unknown name, or a mapping without ``type``.
        """
        if isinstance(spec, Mapping):
            if "type" not in spec:
                raise RegistryError(f"{self.kind} config has no 'type' key: {dict(spec)}")
            kwargs = {**kwargs, **{k: v for k, v in spec.items() if k != "type"}}
            spec = spec["type"]
        return self.get(spec)(**kwargs)

    def names(self) -> list[str]:
        """Sorted names resolvable here (own entries plus the parent's)."""
        own = set(self._entries)
        return sorted(own | set(self.parent.names()) if self.parent is not None else own)

    def __contains__(self, name: object) -> bool:
        if name in self._entries:
            return True
        return self.parent is not None and name in self.parent

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def __repr__(self) -> str:
        return f"Registry(kind={self.kind!r}, names={self.names()})"
