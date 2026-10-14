"""Static flow chart of a lifting pipeline (renders everywhere, unlike Mermaid in .ipynb)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LEARNED = ("#f1e6f8", "#8e44ad")
GEOMETRY = ("#eaf2fb", "#1f77b4")
REUSED = ("#eaf7ea", "#2e8b57")


class PipelineDiagram:
    """Left-to-right boxes connected by arrows, colour-coded by what each stage is.

    * purple - learned (trainable parameters),
    * blue - fixed geometry / data (calibration, projection, sampling),
    * green - reused *unchanged* from the existing BEV lifter.

    Notebooks also contain the same pipeline as a Mermaid block (Jupyter >= 4.1 and GitHub
    Markdown render it); this matplotlib version is the fallback that renders in any viewer.

    Args:
        title: figure title.
    """

    def __init__(self, title: str) -> None:
        self.title = title
        self.stages: list[tuple[str, str, str]] = []

    def add(self, name: str, detail: str, kind: str = "geometry") -> PipelineDiagram:
        """Append a stage. ``kind`` is ``"learned"``, ``"geometry"`` or ``"reused"`` (fluent)."""
        if kind not in {"learned", "geometry", "reused"}:
            raise ValueError(f"unknown stage kind {kind!r}")
        self.stages.append((name, detail, kind))
        return self

    def extend(self, stages: Sequence[tuple[str, str, str]]) -> PipelineDiagram:
        for stage in stages:
            self.add(*stage)
        return self

    def show(self, box_width: float = 2.3, gap: float = 0.55) -> None:
        colors = {"learned": LEARNED, "geometry": GEOMETRY, "reused": REUSED}
        n = len(self.stages)
        total = n * box_width + (n - 1) * gap
        fig, ax = plt.subplots(figsize=(max(8.0, total * 0.8), 2.7), facecolor="white")
        for i, (name, detail, kind) in enumerate(self.stages):
            x = i * (box_width + gap)
            fc, ec = colors[kind]
            ax.add_patch(
                FancyBboxPatch(
                    (x, 0.45),
                    box_width,
                    1.35,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    fc=fc,
                    ec=ec,
                    lw=1.6,
                )
            )
            ax.text(
                x + box_width / 2,
                1.42,
                name,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
            ax.text(
                x + box_width / 2,
                0.93,
                detail,
                ha="center",
                va="center",
                fontsize=7.6,
                linespacing=1.2,
            )
            if i < n - 1:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + box_width + 0.03, 1.12),
                        (x + box_width + gap - 0.03, 1.12),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        lw=1.6,
                        color="#333",
                    )
                )
        for j, (label, kind) in enumerate(
            (
                ("learned", "learned"),
                ("geometry / fixed", "geometry"),
                ("reused unchanged", "reused"),
            )
        ):
            fc, ec = colors[kind]
            ax.add_patch(
                FancyBboxPatch((j * 2.4, 0.02), 0.28, 0.18, boxstyle="round,pad=0.01", fc=fc, ec=ec)
            )
            ax.text(j * 2.4 + 0.38, 0.11, label, fontsize=8, va="center")
        ax.text(0, 2.12, self.title, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.2, total + 0.2), ax.set_ylim(-0.05, 2.35)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
