"""Exception hierarchy of :mod:`lifting`.

.. code-block:: text

    LiftingError
    ├── InvalidConfigError     (also ValueError)  bad hyper-parameters / options
    ├── CameraGeometryError    (also ValueError)  inconsistent camera tensors
    ├── EmptyDataError         (also ValueError)  an input produced no data
    ├── RegistryError          (also KeyError)    unknown / duplicate registry name
    └── MissingDependencyError (also ImportError) optional extra not installed

Each subclass also derives from the built-in exception the code raised before the hierarchy
existed, so ``except ValueError`` / ``except KeyError`` call sites keep working.
"""

from lifting.exceptions.camera_geometry_error import CameraGeometryError
from lifting.exceptions.empty_data_error import EmptyDataError
from lifting.exceptions.invalid_config_error import InvalidConfigError
from lifting.exceptions.lifting_error import LiftingError
from lifting.exceptions.missing_dependency_error import MissingDependencyError
from lifting.exceptions.registry_error import RegistryError

__all__ = [
    "CameraGeometryError",
    "EmptyDataError",
    "InvalidConfigError",
    "LiftingError",
    "MissingDependencyError",
    "RegistryError",
]
