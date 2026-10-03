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

## Running them

```bash
pip install -e .            # from the repo root (Python >= 3.10)
pip install matplotlib jupyter
jupyter lab notebooks/
```

The notebooks are saved with their outputs, so you can read them without running anything. Numbers (mIoU etc.) come from single short runs
and will vary a little between machines and seeds.
