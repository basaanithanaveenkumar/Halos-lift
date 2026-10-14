"""Show a technique's animated GIF and its steps as a static contact sheet."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage
from IPython.display import display

from lifting.triplane.animation import STORYBOARDS, Storyboard


class StoryboardViewer:
    """Wraps one :class:`lifting.triplane.animation.Storyboard`.

    The GIF in ``assets/triplane/`` is the same animation embedded in the README.
    :meth:`contact_sheet` re-renders the *last* frame of every step so the whole lifting
    sequence can be read at once (also in viewers that do not animate GIFs).

    Args:
        name: technique name (``STORYBOARDS`` uses the same names as ``TRIPLANE_LIFTERS``).
        assets_dir: where ``make_triplane_gifs.py`` wrote the GIFs.
    """

    def __init__(self, name: str, assets_dir: str | Path | None = None) -> None:
        self.name = name
        self.storyboard: Storyboard = STORYBOARDS.build(name)
        self.assets_dir = Path(assets_dir) if assets_dir else self._find_assets()

    @staticmethod
    def _find_assets() -> Path:
        """``assets/triplane`` of the repo, whether Jupyter runs from the repo root or a notebook dir."""
        for root in (Path.cwd(), *Path.cwd().parents):
            candidate = root / "assets" / "triplane"
            if candidate.is_dir():
                return candidate
        return Path("assets/triplane")

    @property
    def gif_path(self) -> Path:
        return self.assets_dir / f"{self.name}.gif"

    def show_gif(self) -> None:
        """Display the animated GIF, rendering it first if it does not exist yet."""
        if not self.gif_path.exists():
            self.storyboard.save(self.gif_path)
        display(IPImage(filename=str(self.gif_path)))

    def step_titles(self) -> list[str]:
        return [step.title for step in self.storyboard.steps()]

    def contact_sheet(self, columns: int = 2) -> None:
        """Every step's final frame in a grid, with the step caption underneath."""
        steps = self.storyboard.steps()
        rows = int(np.ceil(len(steps) / columns))
        fig, axes = plt.subplots(
            rows, columns, figsize=(7.2 * columns, 4.2 * rows), facecolor="white"
        )
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for idx, (ax, step) in enumerate(zip(np.atleast_1d(axes).ravel(), steps, strict=False)):
            ax.imshow(np.asarray(self.storyboard.render_frame(steps, idx, 1.0)))
            ax.set_title(f"step {idx + 1}: {step.title}", fontsize=10)
        plt.tight_layout()
        plt.show()
