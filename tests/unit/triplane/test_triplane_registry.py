"""The triplane package mirrors every BEV lifter and leaves the BEV registry untouched."""

from lifting.lifters import LIFTERS, BaseLifter
from lifting.triplane import TRIPLANE_LIFTERS, BaseTriplaneLifter

BEV_NAMES = {
    "simple_bev",
    "depth_warp",
    "lift_splat",
    "bevformer",
    "tiim",
    "tpvformer",
    "mlp_view",
    "alpamayo",
}


def test_every_bev_lifter_has_a_triplane_counterpart() -> None:
    assert set(LIFTERS.names()) <= set(TRIPLANE_LIFTERS.names())
    assert set(TRIPLANE_LIFTERS.names()) == set(LIFTERS.names()) | {"bev_adapter"}


def test_importing_triplane_does_not_touch_bev_registry() -> None:
    assert set(LIFTERS.names()) == BEV_NAMES


def test_triplane_lifters_are_bev_lifters() -> None:
    for name in TRIPLANE_LIFTERS.names():
        cls = TRIPLANE_LIFTERS.get(name)
        assert issubclass(cls, BaseTriplaneLifter) and issubclass(cls, BaseLifter), name
