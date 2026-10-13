"""Alpamayo triplane lifter — Mip-NeRF 360 non-linear scene parametrization.

Ivanovic et al., "Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving",
arXiv:2506.12251, 2025.  Used in Alpamayo-R1 (arXiv:2511.00088, NVIDIA, 2025).
"""

from lifting.lifters.alpamayo.alpamayo_lifter import AlpamayoTriplaneLifter
from lifting.lifters.alpamayo.mipnerf360 import MipNeRF360Parametrizer
from lifting.lifters.alpamayo.triplane_ref_points import AlpamayoReferencePoints

__all__ = [
    "AlpamayoTriplaneLifter",
    "AlpamayoReferencePoints",
    "MipNeRF360Parametrizer",
]
