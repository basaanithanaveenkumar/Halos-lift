"""Per-step losses and metrics recorded by :class:`Trainer`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    """Loss per optimisation step."""

    losses: list[float] = field(default_factory=list)

    def smoothed(self, window: int = 10) -> list[float]:
        out = []
        for i in range(len(self.losses)):
            chunk = self.losses[max(0, i - window + 1) : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out
