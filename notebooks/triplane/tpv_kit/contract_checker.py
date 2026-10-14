"""Executable checks of the ``BaseTriplaneLifter`` contract for one lifter instance."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from lifting.geometry import Cameras
from lifting.lifters import LIFTERS
from lifting.triplane import BaseTriplaneLifter


class TriplaneContractChecker:
    """Runs the five properties every triplane lifter promises and prints a pass/fail table.

    1. ``lift_planes`` returns ``xy (X, Y)``, ``xz (X, Z)``, ``yz (Y, Z)`` with ``out_channels``.
    2. ``forward`` (the BEV path) equals ``bev_head(lift_planes(...))`` - BEV is read off the planes.
    3. The voxel feature is the TPVFormer sum ``f(x, y, z) = xy(x, y) + xz(x, z) + yz(y, z)``.
    4. Gradients reach the image features (the lifter is trainable end to end).
    5. The wrapped BEV lifter (``lifter.core``) is the *unchanged* class from ``LIFTERS``.

    Args:
        lifter: any :class:`BaseTriplaneLifter`.
        features: ``(B, N, C, H_l, W_l)`` per pyramid level.
        cameras: calibration of the same batch.
    """

    def __init__(
        self, lifter: BaseTriplaneLifter, features: Sequence[Tensor], cameras: Cameras
    ) -> None:
        self.lifter = lifter
        self.features = [f.detach() for f in features]
        self.cameras = cameras
        self.results: list[tuple[str, bool, str]] = []

    def _record(self, name: str, ok: bool, detail: str) -> None:
        self.results.append((name, bool(ok), detail))

    def check_shapes(self) -> None:
        with torch.no_grad():
            planes = self.lifter.lift_planes(self.features, self.cameras)
        nx, ny, nz = self.lifter.grid.resolution
        c = self.lifter.out_channels
        b = self.features[0].shape[0]
        expected = [(b, c, nx, ny), (b, c, nx, nz), (b, c, ny, nz)]
        got = [tuple(p.shape) for p in (planes.xy, planes.xz, planes.yz)]
        self._record("plane shapes", got == expected, f"xy{got[0]} xz{got[1]} yz{got[2]}")

    def check_bev_from_planes(self) -> None:
        with torch.no_grad():
            bev = self.lifter(self.features, self.cameras)
            ref = self.lifter.bev_head(self.lifter.lift_planes(self.features, self.cameras))
        err = (bev - ref).abs().max().item()
        self._record(
            "BEV = bev_head(planes)", err < 1e-5, f"BEV {tuple(bev.shape)}, max |diff| = {err:.1e}"
        )

    def check_voxel_sum(self) -> None:
        with torch.no_grad():
            planes = self.lifter.lift_planes(self.features, self.cameras)
        vox = planes.to_voxels()
        i, j, k = (s // 2 for s in self.lifter.grid.resolution)
        by_hand = planes.xy[..., i, j] + planes.xz[..., i, k] + planes.yz[..., j, k]
        err = (vox[..., i, j, k] - by_hand).abs().max().item()
        self._record("f = xy + xz + yz", err < 1e-5, f"voxel ({i},{j},{k}) max |diff| = {err:.1e}")

    def check_gradients(self) -> None:
        feats = [f.clone().requires_grad_(True) for f in self.features]
        planes = self.lifter.lift_planes(feats, self.cameras)
        (
            planes.xy.square().mean() + planes.xz.square().mean() + planes.yz.square().mean()
        ).backward()
        reached = sum(f.grad is not None and f.grad.abs().sum().item() > 0 for f in feats)
        self.lifter.zero_grad(set_to_none=True)
        self._record(
            "gradients reach images",
            reached > 0,
            f"{reached}/{len(feats)} pyramid levels receive gradient",
        )

    def check_reuse(self) -> None:
        core = getattr(self.lifter, "core", None)
        if core is None:
            self._record(
                "reuses BEV lifter",
                True,
                f"native triplane lifter built from existing layers ({type(self.lifter).__name__})",
            )
            return
        registered = [n for n in LIFTERS.names() if type(core) is LIFTERS.get(n)]
        ok = bool(registered)
        self._record(
            "reuses BEV lifter",
            ok,
            f"core = {type(core).__name__} = LIFTERS[{registered[0]!r}]"
            if ok
            else type(core).__name__,
        )

    def run(self) -> bool:
        """Run all checks, print the table and return ``True`` if every check passed."""
        self.results.clear()
        was_training = self.lifter.training
        self.lifter.eval()
        try:
            self.check_shapes()
            self.check_bev_from_planes()
            self.check_voxel_sum()
            self.check_reuse()
        finally:
            self.lifter.train(was_training)
        self.check_gradients()
        width = max(len(n) for n, _, _ in self.results)
        for name, ok, detail in self.results:
            print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
        return all(ok for _, ok, _ in self.results)
