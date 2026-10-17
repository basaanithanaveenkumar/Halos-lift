"""Configuration of the Waymo Open BEV mid-training dataset."""

from __future__ import annotations

from dataclasses import dataclass

from lifting.data.stages import TrainingPhase


@dataclass
class WaymoBEVConfig:
    """Configuration for the Waymo BEV mid-training dataset (proxy mode)."""

    proxy_length: int = 52_386  # ≈ Waymo train frame count
    seed: int = 10
    phase: TrainingPhase = TrainingPhase.MID_TRAIN
