"""Animated GIFs that play, step by step, how each technique lifts images into a triplane.

Pillow-only (no matplotlib), so it ships with the library::

    from lifting.triplane.animation import render_all
    render_all("assets/triplane")            # one <technique>.gif per triplane lifter

Each technique is a :class:`Storyboard` (one class per file under ``storyboards/``) built from
shared steps. Geometry is exact: cameras, projections, ray casting and occupancy are computed
on a small toy scene (:class:`ToyScene`, :class:`ToyRig`) with the library's conventions.
"""

from lifting.triplane.animation.render import render_all
from lifting.triplane.animation.step import Step
from lifting.triplane.animation.storyboard import STORYBOARDS, Storyboard
from lifting.triplane.animation.storyboards import (
    AlpamayoStoryboard,
    BEVAdapterStoryboard,
    BEVFormerStoryboard,
    BEVSideStoryboard,
    DepthWarpStoryboard,
    LiftSplatStoryboard,
    MLPViewStoryboard,
    SimpleBEVStoryboard,
    TIIMStoryboard,
    TPVFormerStoryboard,
)
from lifting.triplane.animation.theme import Theme
from lifting.triplane.animation.toy_object import ToyObject
from lifting.triplane.animation.toy_rig import ToyRig
from lifting.triplane.animation.toy_scene import ToyScene

__all__ = [
    "STORYBOARDS",
    "AlpamayoStoryboard",
    "BEVAdapterStoryboard",
    "BEVSideStoryboard",
    "BEVFormerStoryboard",
    "DepthWarpStoryboard",
    "LiftSplatStoryboard",
    "MLPViewStoryboard",
    "SimpleBEVStoryboard",
    "Step",
    "Storyboard",
    "TIIMStoryboard",
    "TPVFormerStoryboard",
    "Theme",
    "ToyObject",
    "ToyRig",
    "ToyScene",
    "render_all",
]
