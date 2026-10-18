"""YAML/JSON config loading with inheritance and command-line overrides."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from lifting.exceptions import InvalidConfigError

#: Keys naming parent config(s). ``inherits`` is what the shipped ``configs/*.yaml`` use.
BASE_KEYS = ("_base_", "inherits")
DELETE_KEY = "_delete_"


class ConfigLoader:
    """Loads a config file, resolving its bases, and applies dotted overrides.

    Pattern: Facade over parsing, inheritance resolution and override application.

    * ``_base_: path`` (or ``inherits:``; a string or a list) names parents, resolved relative
      to the file that names them; later bases override earlier ones, the child overrides all.
    * Child dicts deep-merge over the base; a child dict containing ``_delete_: true`` replaces
      the base dict instead of merging.
    * A base cycle raises :class:`InvalidConfigError` naming the chain.

    Example::

        cfg = ConfigLoader.load("configs/train_synthetic.yaml", ["train.lr=1e-3", "model.channels=64"])
    """

    @classmethod
    def load(cls, path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
        """Load ``path`` with its bases and apply ``overrides``.

        Args:
            path: ``.yaml`` / ``.yml`` / ``.json`` file.
            overrides: ``dotted.key=value`` strings; values are parsed as YAML scalars/lists.

        Returns:
            The merged config (a fresh dict; bases are not mutated).

        Raises:
            InvalidConfigError: unreadable file, base cycle, non-mapping root or bad override.
            MissingDependencyError: a YAML file was given and ``pyyaml`` is not installed.
        """
        config = cls._load_resolved(Path(path).resolve(), ())
        return cls.apply_overrides(config, overrides)

    @classmethod
    def apply_overrides(cls, config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
        """Return a copy of ``config`` with ``dotted.key=value`` overrides applied.

        Intermediate dicts are created when missing. Raises :class:`InvalidConfigError` for an
        override without ``=`` or one that descends into a non-mapping value.
        """
        result = copy.deepcopy(dict(config))
        for item in overrides:
            key, sep, raw = item.partition("=")
            if not sep or not key.strip():
                raise InvalidConfigError(f"override {item!r} must look like dotted.key=value")
            *parents, leaf = key.strip().split(".")
            node = result
            for part in parents:
                child = node.setdefault(part, {})
                if not isinstance(child, dict):
                    raise InvalidConfigError(f"override {item!r}: {part!r} is not a mapping")
                node = child
            node[leaf] = cls._parse_scalar(raw)
        return result

    @classmethod
    def merge(cls, base: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
        """Deep-merge ``child`` over ``base`` honouring ``_delete_``. Time O(size of child)."""
        merged = copy.deepcopy(dict(base))
        for key, value in child.items():
            if key == DELETE_KEY:
                continue
            if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                if value.get(DELETE_KEY):
                    merged[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != DELETE_KEY}
                else:
                    merged[key] = cls.merge(merged[key], value)
            elif isinstance(value, Mapping):
                merged[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != DELETE_KEY}
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    # ------------------------------------------------------------------ internals
    @classmethod
    def _load_resolved(cls, path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
        if path in chain:
            cycle = " -> ".join(str(p) for p in (*chain, path))
            raise InvalidConfigError(f"config base cycle: {cycle}")
        raw = cls._read(path)
        bases: list[str] = []
        for key in BASE_KEYS:
            declared = raw.pop(key, None)
            if declared is None:
                continue
            bases.extend([declared] if isinstance(declared, str) else list(declared))
        merged: dict[str, Any] = {}
        for base in bases:
            merged = cls.merge(
                merged, cls._load_resolved((path.parent / base).resolve(), (*chain, path))
            )
        logger.debug("loaded config {} (bases: {})", path, bases)
        return cls.merge(merged, raw)

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            text = path.read_text()
        except OSError as err:
            raise InvalidConfigError(f"cannot read config {path}: {err}") from err
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            from lifting.config.yaml_support import load_yaml  # lazy: pyyaml is only needed here

            data = load_yaml(text, path)
        if not isinstance(data, dict):
            raise InvalidConfigError(f"config {path} must contain a mapping at the top level")
        return data

    @staticmethod
    def _parse_scalar(raw: str) -> Any:
        from lifting.config.yaml_support import load_yaml

        return load_yaml(raw.strip(), None)
