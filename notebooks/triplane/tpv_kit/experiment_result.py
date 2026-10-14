"""The numbers one triplane training run produces."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentResult:
    """Outcome of :class:`OccupancyExperiment.run`.

    Attributes:
        name: triplane lifter name in ``TRIPLANE_LIFTERS``.
        params: trainable parameters of the whole model.
        seconds: wall-clock training time.
        miou_before: foreground mIoU of the untrained model.
        miou_after: foreground mIoU after training (correct calibration).
        miou_corrupted: foreground mIoU after training with every camera yawed by 90 degrees.
        losses: training loss per step.
    """

    name: str
    params: int
    seconds: float
    miou_before: float
    miou_after: float
    miou_corrupted: float
    losses: list[float] = field(default_factory=list)

    @property
    def uses_geometry(self) -> bool:
        """A lifter that reads pixels *through* the calibration loses most of its score."""
        return self.miou_corrupted < 0.5 * self.miou_after

    def row(self) -> str:
        """One line of the comparison table printed by the notebooks."""
        if self.miou_after < 0.1:
            verdict = "cannot localise objects"
        else:
            verdict = "uses geometry" if self.uses_geometry else "ignores geometry"
        return (
            f"{self.name:12s} {self.params / 1e3:8.0f}k {self.seconds:6.1f}s "
            f"{self.miou_before:8.3f} {self.miou_after:8.3f} {self.miou_corrupted:10.3f}   {verdict}"
        )

    @staticmethod
    def header() -> str:
        return f"{'lifter':12s} {'params':>9s} {'time':>7s} {'before':>8s} {'after':>8s} {'corrupted':>10s}   verdict"
