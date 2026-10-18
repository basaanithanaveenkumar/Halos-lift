"""Typed ``lifting-train`` settings validated from a loaded config mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from lifting.exceptions import InvalidConfigError

#: Config section -> {config key: TrainConfig field}. Keys not listed are rejected so a typo
#: fails fast instead of silently using a default.
SECTIONS: dict[str, dict[str, str]] = {
    "model": {
        "lifter": "lifter",
        "kind": "model",
        "channels": "channels",
        "num_classes": "num_classes",
        "resolution": "resolution",
    },
    "data": {
        "train_scenes": "train_scenes",
        "val_scenes": "val_scenes",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
    },
    "train": {
        "steps": "steps",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
        "depth_weight": "depth_weight",
        "aux_weight": "aux_weight",
        "seed": "seed",
        "eval_every": "eval_every",
        "keep_ckpts": "keep_ckpts",
    },
}


@dataclass(frozen=True)
class TrainConfig:
    """Validated settings of ``lifting-train`` (the boundary type; no raw dicts past it).

    Field names equal the command-line destinations, so a config file only supplies defaults
    that explicit flags still override.

    Example::

        cfg = TrainConfig.from_mapping(ConfigLoader.load("configs/train_synthetic.yaml"))
        parser.set_defaults(**cfg.as_defaults())
    """

    lifter: str = "simple_bev"
    model: str = "bev"
    channels: int = 32
    num_classes: int = 3
    resolution: tuple[int, int, int] = (32, 32, 4)
    train_scenes: int = 512
    val_scenes: int = 64
    batch_size: int = 4
    num_workers: int = 0
    steps: int = 300
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    depth_weight: float = 0.0
    aux_weight: float = 0.5
    seed: int = 0
    eval_every: int = 1
    keep_ckpts: int = 3

    def __post_init__(self) -> None:
        if self.model not in ("bev", "triplane"):
            raise InvalidConfigError(f"model.kind must be 'bev' or 'triplane', got {self.model!r}")
        if len(self.resolution) != 3 or min(self.resolution) < 1:
            raise InvalidConfigError(
                f"model.resolution needs three positive ints, got {self.resolution}"
            )
        for name in (
            "channels",
            "num_classes",
            "train_scenes",
            "val_scenes",
            "batch_size",
            "steps",
        ):
            if getattr(self, name) < 1:
                raise InvalidConfigError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.lr <= 0:
            raise InvalidConfigError(f"train.lr must be > 0, got {self.lr}")
        if min(self.weight_decay, self.depth_weight, self.aux_weight, self.grad_clip) < 0:
            raise InvalidConfigError(
                "weight_decay, grad_clip, depth_weight and aux_weight must be >= 0"
            )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> TrainConfig:
        """Validate the ``model`` / ``data`` / ``train`` sections of a loaded config.

        Top-level sections this class does not know are ignored; unknown keys inside
        ``model`` / ``data`` / ``train`` raise (the phase configs are not training configs).

        Raises:
            InvalidConfigError: unknown key, wrong type or out-of-range value.
        """
        values: dict[str, Any] = {}
        for section, keys in SECTIONS.items():
            body = config.get(section) or {}
            if not isinstance(body, Mapping):
                raise InvalidConfigError(f"config section {section!r} must be a mapping")
            for key, value in body.items():
                if key not in keys:
                    raise InvalidConfigError(
                        f"unknown key {section}.{key}; allowed: {sorted(keys)}"
                    )
                values[keys[key]] = value
        if "resolution" in values:
            values["resolution"] = tuple(values["resolution"])
        try:
            return cls(**values)
        except TypeError as err:
            raise InvalidConfigError(f"invalid training config: {err}") from err

    def as_defaults(self) -> dict[str, Any]:
        """Field values keyed by argparse destination (for ``parser.set_defaults``)."""
        return asdict(self)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Names of all settings."""
        return tuple(f.name for f in fields(cls))
