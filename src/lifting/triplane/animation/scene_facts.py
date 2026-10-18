"""Quantities derived once from the toy scene and reused by every frame of a storyboard."""

from __future__ import annotations

from functools import cached_property

import numpy as np

from lifting.triplane.animation.toy_rig import ToyRig
from lifting.triplane.animation.toy_scene import ToyScene


class SceneFacts:
    """Exact answers to "what would each lifter see?" for the toy scene.

    Everything is computed with real projection and ray casting so the animations show what
    the real algorithms do - e.g. Simple-BEV's feature *smear* along camera rays.

    Args:
        scene, rig: the toy world and cameras.
        image_resolution: ray-cast resolution of the camera images ``(width, height)``.
    """

    def __init__(
        self, scene: ToyScene, rig: ToyRig, image_resolution: tuple[int, int] = (75, 33)
    ) -> None:
        self.scene = scene
        self.rig = rig
        self.image_resolution = image_resolution

    @cached_property
    def images(self) -> list[np.ndarray]:
        """Per-camera ``(H, W)`` class-id images (``-1`` sky, ``0`` ground)."""
        return self.rig.render(self.scene, *self.image_resolution)

    @cached_property
    def occupancy(self) -> np.ndarray:
        """Voxel occupancy of the toy scene."""
        return self.scene.occupancy()

    @cached_property
    def plane_labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-plane class labels collapsed from the occupancy."""
        return self.scene.plane_labels(self.occupancy)

    @cached_property
    def _voxel_hits(
        self,
    ) -> dict[tuple[int, int, int], list[tuple[int, float, float, int, float, float]]]:
        """voxel -> list of ``(cam, u, v, hit_cls, hit_depth, voxel_depth)``."""
        hits = {}
        centers = self.scene.voxel_centers()
        for idx in np.ndindex(*self.scene.resolution):
            row = []
            for cam, u, v, depth in self.rig.project(centers[idx]):
                cls, hit_depth = self.rig.cast(self.scene, cam, u, v)
                row.append((cam, u, v, cls, hit_depth, depth))
            hits[idx] = row
        return hits

    @cached_property
    def smear(self) -> np.ndarray:
        """Simple-BEV voxel features: the class seen at the voxel's pixel in any camera.

        Without depth, every voxel along a camera ray receives the same feature.
        """
        out = np.zeros(self.scene.resolution, dtype=np.int64)
        for idx, row in self._voxel_hits.items():
            seen = [cls for _, _, _, cls, _, _ in row if cls > 0]
            out[idx] = max(seen) if seen else 0
        return out

    def depth_consistent(self, tolerance: float = 0.9) -> np.ndarray:
        """Depth-aware voxel features: the class only where the predicted (true) surface depth
        matches the voxel's depth - what a well-trained depth head achieves."""
        out = np.zeros(self.scene.resolution, dtype=np.int64)
        for idx, row in self._voxel_hits.items():
            seen = [cls for _, _, _, cls, hd, vd in row if cls > 0 and abs(hd - vd) < tolerance]
            out[idx] = max(seen) if seen else 0
        return out

    def object_pixels(self, cam: int, stride: int = 4) -> list[tuple[float, float, int, float]]:
        """``(u, v, cls, depth)`` of object pixels of ``cam``, subsampled by ``stride``."""
        img = self.images[cam]
        h, w = img.shape
        out = []
        for r in range(0, h, stride):
            for c in range(0, w, stride):
                if img[r, c] > 0:
                    u, v = (c + 0.5) / w, (r + 0.5) / h
                    out.append((u, v, int(img[r, c]), self.rig.cast(self.scene, cam, u, v)[1]))
        return out

    def voxel_views(self, idx: tuple[int, int, int]) -> list[tuple[int, float, float, float]]:
        """``(cam, u, v, depth)`` for every camera that sees voxel ``idx``."""
        return [(cam, u, v, vd) for cam, u, v, _, _, vd in self._voxel_hits[idx]]
