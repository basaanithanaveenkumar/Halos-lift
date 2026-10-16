"""Exception hierarchy and the spec-compliant :class:`lifting.registry.Registry`."""

from __future__ import annotations

import pytest
import torch

from lifting.exceptions import (
    CameraGeometryError,
    EmptyDataError,
    InvalidConfigError,
    LiftingError,
    MissingDependencyError,
    RegistryError,
)
from lifting.geometry import Cameras
from lifting.registry import Registry


class _Thing:
    def __init__(self, size: int = 1) -> None:
        self.size = size


@pytest.mark.parametrize(
    ("exc", "builtin"),
    [
        (InvalidConfigError, ValueError),
        (CameraGeometryError, ValueError),
        (EmptyDataError, ValueError),
        (RegistryError, KeyError),
        (MissingDependencyError, ImportError),
    ],
)
def test_every_error_is_a_lifting_error_and_keeps_its_builtin(exc, builtin):
    assert issubclass(exc, LiftingError) and issubclass(exc, builtin)


def test_missing_dependency_message_names_the_extra():
    err = MissingDependencyError("torchvision", "vision")
    assert err.package == "torchvision" and "halos-lift[vision]" in str(err)


def test_registry_register_get_build_by_name_and_config():
    reg: Registry[_Thing] = Registry("thing")
    reg.register("a")(_Thing)
    reg.register_module(module=_Thing)
    assert reg.build("a", size=2).size == 2
    assert reg.build({"type": "_Thing", "size": 5}, size=1).size == 5  # config wins
    assert reg.names() == ["_Thing", "a"] and len(reg) == 2 and "a" in reg


def test_registry_errors_suggest_close_matches():
    reg: Registry[_Thing] = Registry("thing")
    reg.register("simple_bev")(_Thing)
    with pytest.raises(RegistryError, match="Did you mean"):
        reg.get("simple_bevv")
    with pytest.raises(KeyError, match="already registered"):
        reg.register("simple_bev")(_Thing)
    reg.register("simple_bev", force=True)(dict)
    assert reg.get("simple_bev") is dict
    with pytest.raises(RegistryError, match="no 'type'"):
        reg.build({"size": 1})


def test_child_registry_falls_back_to_parent():
    parent: Registry[_Thing] = Registry("thing")
    parent.register("base")(_Thing)
    child: Registry[_Thing] = Registry("thing", parent=parent)
    child.register("extra")(_Thing)
    assert child.get("base") is _Thing and child.names() == ["base", "extra"]
    assert "extra" not in parent


def test_cameras_reject_inconsistent_shapes():
    with pytest.raises(CameraGeometryError):
        Cameras(
            intrinsics=torch.eye(3).expand(1, 2, 3, 3),
            cam_to_ego=torch.eye(4).expand(1, 3, 4, 4),
            image_size=(32, 32),
        )
