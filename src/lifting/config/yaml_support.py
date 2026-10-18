"""Lazy PyYAML access shared by the config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lifting.exceptions import InvalidConfigError, MissingDependencyError


def load_yaml(text: str, source: Path | None) -> Any:
    """Parse YAML ``text``; ``source`` (a file or ``None`` for an override) names errors.

    Raises:
        MissingDependencyError: ``pyyaml`` is not installed.
        InvalidConfigError: the text is not valid YAML.
    """
    try:
        import yaml
    except ImportError as err:
        raise MissingDependencyError("pyyaml", "config") from err
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise InvalidConfigError(f"invalid YAML in {source or 'override'}: {err}") from err
