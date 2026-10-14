"""A small, deterministic multi-camera scene shared by every triplane notebook."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from lifting.data import CameraRig, SceneBatch, SyntheticSceneDataset
from lifting.encoders import ImageEncoder
from lifting.geometry import GridSpec


class TriplaneScene:
    """Grid + camera rig + synthetic dataset, split into fixed train / test batches.

    Every notebook builds one of these so all techniques are compared on *exactly* the same
    scenes. The default size (``16 x 16 x 4`` cells over ``32 x 32 x 4`` m, four ``48 x 96``
    cameras) trains any triplane lifter on a laptop CPU in seconds.

    Args:
        resolution: ``(X, Y, Z)`` cells of the region of interest.
        num_cameras: cameras in the rig (yaw is jittered per scene so lifters must use geometry).
        image_size: ``(H, W)`` of every camera image.
        num_train: training scenes (multiple of ``batch_size``).
        num_test: held-out scenes (multiple of ``batch_size``).
        batch_size: scenes per :class:`SceneBatch`.
        seed: dataset seed; scene ``i`` is always the same.
    """

    def __init__(
        self,
        resolution: tuple[int, int, int] = (16, 16, 4),
        num_cameras: int = 4,
        image_size: tuple[int, int] = (48, 96),
        num_train: int = 128,
        num_test: int = 32,
        batch_size: int = 4,
        seed: int = 0,
    ) -> None:
        self.grid = GridSpec(resolution=resolution)
        self.rig = CameraRig(num_cameras=num_cameras, image_size=image_size)
        self.batch_size = batch_size
        self.dataset = SyntheticSceneDataset(
            num_train + num_test, grid=self.grid, rig=self.rig, seed=seed
        )
        self.train_batches = self._batches(0, num_train)
        self.test_batches = self._batches(num_train, num_train + num_test)

    # ------------------------------------------------------------------ data
    def _batches(self, start: int, stop: int) -> list[SceneBatch]:
        """Collate scenes ``[start, stop)`` into consecutive batches."""
        bs = self.batch_size
        return [
            SceneBatch.collate([self.dataset[i + j] for j in range(bs)])
            for i in range(start, stop, bs)
        ]

    @property
    def num_cameras(self) -> int:
        return self.rig.num_cameras

    @property
    def image_size(self) -> tuple[int, int]:
        return self.rig.image_size

    @property
    def sample(self) -> SceneBatch:
        """The first held-out batch: the one every notebook visualises."""
        return self.test_batches[0]

    # ------------------------------------------------------------------ features
    @staticmethod
    def encode(encoder: ImageEncoder, batch: SceneBatch) -> list[Tensor]:
        """Run ``encoder`` on all cameras and restore the camera axis.

        Returns:
            One ``(B, N, C, H_l, W_l)`` tensor per pyramid level, fine -> coarse - the input
            contract of every lifter (exactly what ``TPVFormer.forward`` does internally).
        """
        b, n = batch.images.shape[:2]
        return [f.view(b, n, *f.shape[1:]) for f in encoder(batch.images.flatten(0, 1))]

    # ------------------------------------------------------------------ labels
    @staticmethod
    def label_planes(occupancy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project ``(B, X, Y, Z)`` class ids onto the three planes.

        The highest class along the collapsed axis wins, so any occupied voxel marks its
        column - the same targets :class:`lifting.training.OccupancyTask` uses for the
        per-plane auxiliary loss.

        Returns:
            ``xy (B, X, Y)``, ``xz (B, X, Z)``, ``yz (B, Y, Z)``.
        """
        return occupancy.amax(3), occupancy.amax(2), occupancy.amax(1)

    def describe(self) -> str:
        """One-paragraph summary printed at the top of each notebook."""
        g = self.grid
        lines = [
            f"grid      : x {g.x_bounds} y {g.y_bounds} z {g.z_bounds} m, resolution {g.resolution}",
            f"voxel     : {tuple(round(v, 2) for v in g.voxel_size)} m",
            f"cameras   : {self.num_cameras} x {self.image_size} px",
            f"scenes    : {len(self.train_batches) * self.batch_size} train / "
            f"{len(self.test_batches) * self.batch_size} test (batch {self.batch_size})",
            f"TPV cells : xy {g.resolution[0] * g.resolution[1]} + xz {g.resolution[0] * g.resolution[2]}"
            f" + yz {g.resolution[1] * g.resolution[2]} = {self.tpv_cells(g.resolution)}"
            f"  (voxels: {g.resolution[0] * g.resolution[1] * g.resolution[2]})",
        ]
        return "\n".join(lines)

    @staticmethod
    def tpv_cells(resolution: Sequence[int]) -> int:
        """Number of plane cells ``XY + XZ + YZ``."""
        nx, ny, nz = resolution
        return nx * ny + nx * nz + ny * nz
