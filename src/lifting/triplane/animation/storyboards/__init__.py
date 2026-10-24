"""One storyboard per lifting technique; importing registers them in ``STORYBOARDS``."""

from lifting.triplane.animation.storyboards.alpamayo_storyboard import AlpamayoStoryboard
from lifting.triplane.animation.storyboards.bev_adapter_storyboard import BEVAdapterStoryboard
from lifting.triplane.animation.storyboards.bev_side_storyboard import BEVSideStoryboard
from lifting.triplane.animation.storyboards.bevformer_storyboard import BEVFormerStoryboard
from lifting.triplane.animation.storyboards.depth_warp_storyboard import DepthWarpStoryboard
from lifting.triplane.animation.storyboards.lift_splat_storyboard import LiftSplatStoryboard
from lifting.triplane.animation.storyboards.mlp_view_storyboard import MLPViewStoryboard
from lifting.triplane.animation.storyboards.simple_bev_storyboard import SimpleBEVStoryboard
from lifting.triplane.animation.storyboards.tiim_storyboard import TIIMStoryboard
from lifting.triplane.animation.storyboards.tpvformer_storyboard import TPVFormerStoryboard

__all__ = [
    "AlpamayoStoryboard",
    "BEVAdapterStoryboard",
    "BEVSideStoryboard",
    "BEVFormerStoryboard",
    "DepthWarpStoryboard",
    "LiftSplatStoryboard",
    "MLPViewStoryboard",
    "SimpleBEVStoryboard",
    "TIIMStoryboard",
    "TPVFormerStoryboard",
]
