"""One pinhole camera of the toy rig used by the triplane animations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyCamera:
    """A level camera at ``position`` looking along ``yaw`` (degrees, ``0`` = ``+x``)."""

    name: str
    position: tuple[float, float, float]
    yaw: float

    @property
    def forward(self) -> np.ndarray:
        a = math.radians(self.yaw)
        return np.array([math.cos(a), math.sin(a), 0.0])

    @property
    def right(self) -> np.ndarray:
        a = math.radians(self.yaw)
        return np.array([math.sin(a), -math.cos(a), 0.0])

    @property
    def up(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])
