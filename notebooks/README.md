# Learning notebooks

Step-by-step, runnable explanations of every lifter in `Halos-lift`, written for someone new to deep learning on 3D / BEV perception.
Each notebook has the same shape: the idea, the geometry, the building blocks (with tiny examples and exact tests), the full lifter, then training
and a "does it really use the camera geometry?" check. They all run on a laptop CPU (about 15-40 s of training each).

| order | notebook | lifter | one-line idea |
|---|---|---|---|
| 1 | `simple_bev_lifting_step_by_step.ipynb` | `simple_bev` | for every voxel: project into the cameras, read the feature, average |
| 2 | `depth_warp_lifting_step_by_step.ipynb` | `depth_warp` | add a per-pixel depth distribution; voxels pull from the frustum |
| 3 | `lift_splat_lifting_step_by_step.ipynb` | `lift_splat` | same frustum, but pushed (summed) into voxels |
| 4 | `tiim_polar_lifting_step_by_step.ipynb` | `tiim` | image column -> polar ray with a transformer, then resample |
| 5 | `bevformer_lifting_step_by_step.ipynb` | `bevformer` | learned BEV queries + deformable attention into the cameras |
| 6 | `tpv_lifting_step_by_step.ipynb` | `tpvformer` | three orthogonal planes (xy, xz, yz) -> 3D occupancy and point queries |
| 7 | `lifter_comparison_and_negative_control.ipynb` | all + `mlp_view` | why a geometry-free control matters, and a side-by-side shoot-out |

## Triplane (TPV) notebooks

`triplane/` has one executed notebook per lifting technique showing how it lifts into the **tri-perspective view**
(`xy` = BEV, `xz`, `yz`) through `lifting.triplane`. Each has the same structure: idea + animated GIF + detailed Mermaid
diagram + equations, every animation step as a still, every intermediate tensor with exact checks, the triplane
contract verified, end-to-end 3D-occupancy training, and the corrupted-extrinsics negative control. Start with the
overview. Shared helpers are in `triplane/tpv_kit/` (one class per file). Background: [`docs/triplane.md`](../docs/triplane.md).

| # | notebook | technique | one-line idea |
|---|---|---|---|
| 00 | [`triplane/00_triplane_overview.ipynb`](triplane/00_triplane_overview.ipynb) | all | architecture, GIF gallery, contract checks and an occupancy shoot-out of every triplane lifter |
| 01 | [`triplane/01_simple_bev_triplane.ipynb`](triplane/01_simple_bev_triplane.ipynb) | `simple_bev` | For every voxel: project its centre into every camera, bilinearly read the feature there, average over the cameras that see it - then collapse the volume along Z, Y and X. |
| 02 | [`triplane/02_depth_warp_triplane.ipynb`](triplane/02_depth_warp_triplane.ipynb) | `depth_warp` | Predict a depth distribution for every pixel, build a depth-weighted frustum per camera, and let every voxel PULL its feature from the frustum at its (u, v, depth) - then collapse along Z, Y and X. |
| 03 | [`triplane/03_lift_splat_triplane.ipynb`](triplane/03_lift_splat_triplane.ipynb) | `lift_splat` | Lift every pixel to D weighted 3D points (depth probability × context), splat (scatter-sum) the point cloud into voxels, and shoot by collapsing the volume along Z, Y and X. |
| 04 | [`triplane/04_tiim_triplane.ipynb`](triplane/04_tiim_triplane.ipynb) | `tiim` | A transformer turns every image column into a polar ray of depth cells; every voxel reads the ray at its (column, range) AND the image at its (column, row) for height - then collapse along Z, Y and X. |
| 05 | [`triplane/05_bevformer_triplane.ipynb`](triplane/05_bevformer_triplane.ipynb) | `bevformer` | One BEVFormer spatial encoder per plane: learned queries on xy, xz and yz, each owning a line of 3D anchors along its plane normal, read the cameras with deformable spatial cross-attention. |
| 06 | [`triplane/06_tpvformer_triplane.ipynb`](triplane/06_tpvformer_triplane.ipynb) | `tpvformer` | Natively a triplane lifter: XY + XZ + YZ learned query tokens alternate cross-view hybrid attention (planes talk to each other) and image cross-attention (anchors projected into the cameras). |
| 07 | [`triplane/07_alpamayo_triplane.ipynb`](triplane/07_alpamayo_triplane.ipynb) | `alpamayo` | TPVFormer layers whose plane cells follow a Mip-NeRF 360 non-linear parametrisation - dense near the ego, coarse far away - and whose planes are patchified into a fixed-length token sequence for a VLA policy. |
| 08 | [`triplane/08_mlp_view_triplane.ipynb`](triplane/08_mlp_view_triplane.ipynb) | `mlp_view` | One fully connected layer per (camera, plane) maps the image cells straight to plane cells; calibration is ignored - the control that proves every other triplane lifter really uses the cameras. |
| 09 | [`triplane/09_bev_adapter_triplane.ipynb`](triplane/09_bev_adapter_triplane.ipynb) | `bev_adapter` | Run any registered BEV lifter unchanged, then expand its BEV map into three planes with the learned BEVToTPV: xy from the map, xz / yz from pooled columns expanded along height. |

## Running them

```bash
pip install -e .            # from the repo root (Python >= 3.10)
pip install matplotlib jupyter
jupyter lab notebooks/
```

The triplane notebooks render Mermaid diagrams in JupyterLab ≥ 4.1 / VS Code (each also has a matplotlib version).
They take 10-50 s each on a laptop CPU.

The notebooks are saved with their outputs, so you can read them without running anything. Numbers (mIoU etc.) come from single short runs
and will vary a little between machines and seeds.
