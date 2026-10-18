"""Config loading (``_base_`` inheritance, dotted overrides) and typed training sections."""

from lifting.config.config_loader import ConfigLoader
from lifting.config.train_config import TrainConfig

__all__ = ["ConfigLoader", "TrainConfig"]
