"""The lifting animations: exact toy geometry, one storyboard per triplane lifter, GIF export."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lifting.triplane import TRIPLANE_LIFTERS
from lifting.triplane.animation import STORYBOARDS, ToyRig, ToyScene


def test_one_storyboard_per_triplane_lifter() -> None:
    assert STORYBOARDS.names() == TRIPLANE_LIFTERS.names()


def test_toy_rig_ray_cast_round_trip() -> None:
    """Casting a pixel and projecting the hit point must give the same pixel back."""
    scene, rig = ToyScene(), ToyRig()
    hits = 0
    for cam in range(len(rig)):
        for u in np.linspace(0.05, 0.95, 7):
            for v in np.linspace(0.55, 0.95, 4):
                cls, depth = rig.cast(scene, cam, float(u), float(v))
                if cls < 0:
                    continue
                hits += 1
                p = rig.point_at(cam, float(u), float(v), depth)
                views = {c: (pu, pv, d) for c, pu, pv, d in rig.project(p)}
                pu, pv, d = views[cam]
                assert abs(pu - u) < 1e-6 and abs(pv - v) < 1e-6 and abs(d - depth) < 1e-6
    assert hits > 50


def test_toy_scene_planes_are_occupancy_collapses() -> None:
    scene = ToyScene()
    occ = scene.occupancy()
    xy, xz, yz = scene.plane_labels(occ)
    assert occ.shape == scene.resolution and set(np.unique(occ)) == {0, 1, 2, 3}
    assert (
        np.array_equal(xy, occ.max(2))
        and np.array_equal(xz, occ.max(1))
        and np.array_equal(yz, occ.max(0))
    )
    assert (occ[..., -1] == 3).any()  # only the pole reaches the top layer


@pytest.mark.parametrize("name", STORYBOARDS.names())
def test_every_step_renders(name: str) -> None:
    board = STORYBOARDS.build(name)
    steps = board.steps()
    assert len(steps) >= 5
    for idx in range(len(steps)):
        for t in (0.0, 1.0):
            frame = board.render_frame(steps, idx, t)
            assert frame.size == board.SIZE


def test_gif_export(tmp_path) -> None:
    board = STORYBOARDS.build("bev_adapter")
    frames, durations = board.render()
    path = board.save(tmp_path / "anim.gif")
    with Image.open(path) as gif:
        # Pillow merges identical consecutive frames when optimising, so compare the
        # total play time (approximately) rather than the frame count.
        assert 1 < gif.n_frames <= len(frames) == len(durations)
        assert gif.info["loop"] == 0
        total = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            total += gif.info["duration"]
    assert abs(total - sum(durations)) <= 0.05 * sum(durations)
