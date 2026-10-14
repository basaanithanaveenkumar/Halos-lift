"""Render every registered storyboard to ``<out_dir>/<name>.gif``."""

from __future__ import annotations

from pathlib import Path


def render_all(out_dir: str | Path, names: list[str] | None = None) -> dict[str, Path]:
    """Write one GIF per technique and return ``{name: path}``.

    Args:
        out_dir: destination directory (created if missing).
        names: subset of :data:`STORYBOARDS` names (default: all).
    """
    # Lazy imports: registering the storyboards pulls in Pillow drawing code only when needed.
    from lifting.triplane.animation.storyboard import STORYBOARDS
    from lifting.triplane.animation.storyboards import __all__ as _registered  # noqa: F401

    out = Path(out_dir)
    return {
        name: STORYBOARDS.build(name).save(out / f"{name}.gif")
        for name in (names or STORYBOARDS.names())
    }
