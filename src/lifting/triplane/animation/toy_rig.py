"""Four pinhole cameras on the toy ego vehicle, with exact projection and ray casting.

Classes moved to their own modules (one public class per file) are re-exported here so
existing ``from lifting.triplane.animation.toy_rig import ...`` statements keep working."""

from __future__ import annotations

import math

import numpy as np

from lifting.triplane.animation.toy_camera import ToyCamera
from lifting.triplane.animation.toy_scene import ToyScene


class ToyRig:
    """Same conventions as :class:`lifting.geometry.Cameras`: ``project`` returns normalised
    ``(u, v)`` in ``[0, 1]`` (``v`` down) and the depth along the optical axis.

    Args:
        hfov, vfov: horizontal / vertical field of view in degrees.
    """

    def __init__(self, hfov: float = 100.0, vfov: float = 62.0, height: float = 0.9) -> None:
        self.cameras = [
            ToyCamera("front", (0.35, 0.0, height), 0.0),
            ToyCamera("left", (0.0, 0.35, height), 90.0),
            ToyCamera("back", (-0.35, 0.0, height), 180.0),
            ToyCamera("right", (0.0, -0.35, height), 270.0),
        ]
        self.tan_h = math.tan(math.radians(hfov / 2))
        self.tan_v = math.tan(math.radians(vfov / 2))

    def __len__(self) -> int:
        return len(self.cameras)

    def project(self, p) -> list[tuple[int, float, float, float]]:
        """Visible ``(camera, u, v, depth)`` of a 3D point, for every camera that sees it."""
        out = []
        for idx, cam in enumerate(self.cameras):
            d = np.asarray(p, dtype=float) - np.asarray(cam.position)
            depth = float(d @ cam.forward)
            if depth <= 0.05:
                continue
            u = 0.5 + 0.5 * float(d @ cam.right) / depth / self.tan_h
            v = 0.5 - 0.5 * float(d @ cam.up) / depth / self.tan_v
            if 0.0 <= u < 1.0 and 0.0 <= v < 1.0:
                out.append((idx, u, v, depth))
        return out

    def ray(self, cam: int, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Origin and (depth-normalised) direction of pixel ``(u, v)``: ``point = o + depth * d``."""
        c = self.cameras[cam]
        d = c.forward + (2 * u - 1) * self.tan_h * c.right + (1 - 2 * v) * self.tan_v * c.up
        return np.asarray(c.position, dtype=float), d

    def point_at(self, cam: int, u: float, v: float, depth: float) -> tuple[float, float, float]:
        o, d = self.ray(cam, u, v)
        return tuple(float(x) for x in o + depth * d)  # type: ignore[return-value]

    def cast(self, scene: ToyScene, cam: int, u: float, v: float) -> tuple[int, float]:
        """Exact ray cast: ``(class id, depth)`` of the first surface; ``0`` ground, ``-1`` sky."""
        o, d = self.ray(cam, u, v)
        best_cls, best_t = -1, math.inf
        for obj in scene.objects:
            t = _slab(o, d, obj.mins, obj.maxs)
            if t is not None and t < best_t:
                best_cls, best_t = obj.cls, t
        if d[2] < -1e-6:
            t_ground = -o[2] / d[2]
            if t_ground < best_t:
                best_cls, best_t = 0, t_ground
        return best_cls, best_t

    def render(self, scene: ToyScene, width: int, height: int) -> list[np.ndarray]:
        """``(height, width)`` class-id images (``-1`` sky, ``0`` ground) and depth, per camera."""
        images = []
        for cam in range(len(self.cameras)):
            img = np.full((height, width), -1, dtype=np.int64)
            for r in range(height):
                for c in range(width):
                    img[r, c] = self.cast(scene, cam, (c + 0.5) / width, (r + 0.5) / height)[0]
            images.append(img)
        return images


def _slab(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float | None:
    """Ray / axis-aligned box intersection distance (``None`` when missed)."""
    t0, t1 = 0.0, math.inf
    for a in range(3):
        if abs(d[a]) < 1e-9:
            if o[a] < lo[a] or o[a] > hi[a]:
                return None
            continue
        ta, tb = (lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return None
    return t0

__all__ = ["ToyCamera"]
