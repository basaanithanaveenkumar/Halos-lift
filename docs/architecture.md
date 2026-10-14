# Architecture

Mermaid diagrams (render on GitHub). See also the [project page](../project-page/index.html)
and the [README](../README.md) for a laymen overview.

---

## 1. Full pipeline — any lifter, one interface

```mermaid
flowchart TB
  subgraph INPUT["Inputs"]
    IMGS["N camera images\n[B, N, 3, H, W]"]
    CAMS["Cameras struct\nintrinsics K  [B, N, 3, 3]\ncam_to_ego T  [B, N, 4, 4]\nimage_size (H, W)"]
    GRID["GridSpec\nx_bounds, y_bounds, z_bounds\nresolution (X, Y, Z)"]
  end

  subgraph ENCODER["ImageEncoder"]
    BACKBONE["ResNet-50 or EfficientNet\n+ Feature Pyramid Network (FPN)\noutput: list of [B, N, C_l, H_l, W_l]\nper FPN scale (fine → coarse)"]
  end

  subgraph LIFTER["BaseLifter — LIFTERS.build(name, grid, …)"]
    PROJ["Project grid voxel centres\nto each camera's image plane\nusing K and T\n→ pixel sample coordinates"]
    FEAT["Gather image features\nat projected coordinates\n(bilinear / trilinear / deformable)"]
    BEV["Compress height axis\n→ BEV feature map\n[B, C, X, Y]"]
    PROJ --> FEAT --> BEV
  end

  subgraph DECODER["BEVDecoder + SegmentationHead"]
    DECONV["Residual decoder\n(transposed convolutions)"]
    HEAD["Conv 1×1 → n_classes\n[B, n_classes, X, Y]"]
    DECONV --> HEAD
  end

  IMGS --> BACKBONE --> LIFTER
  CAMS & GRID --> LIFTER
  LIFTER --> DECODER
  HEAD --> LOGITS["Logits [B, n_classes, X, Y]\n→ CE loss vs BEV seg labels"]
```

---

## 2. Lifter internals — six strategies compared

```mermaid
flowchart TB
  subgraph SIMPLE_BEV["simple_bev — BilinearSamplingLifter"]
    SB1["For each voxel centre (x,y,z):\nproject to image plane via K·T⁻¹"]
    SB2["Bilinear-sample feature at\nnormalised pixel coordinates\nusing F.grid_sample"]
    SB3["Voxel tensor [B, C, X, Y, Z]\nmean/max pool over Z\n→ BEV [B, C, X, Y]"]
    SB1 --> SB2 --> SB3
  end

  subgraph LIFT_SPLAT["lift_splat — LiftSplatLifter"]
    LS1["Predict depth distribution\nD bins per pixel column\np(d | pixel) via softmax"]
    LS2["Outer product: context × depth\n→ frustum feature volume\n[B, N, C, D, H_feat, W_feat]"]
    LS3["Splat into voxel grid\n(QuickCumsum voxel pooling)\n→ [B, C, X, Y, Z] → BEV"]
    LS1 --> LS2 --> LS3
  end

  subgraph BEVFORMER["bevformer — BEVFormerLifter"]
    BF1["Learnable BEV queries\n[X·Y, C] — one per grid cell"]
    BF2["Deformable self-attention\nBEV queries attend to each other\n(reference points = pillar centres)"]
    BF3["Spatial cross-attention\nBEV queries attend to image features\nthrough deformable attention\n(multi-camera, multi-scale)"]
    BF4["FFN → refined BEV queries\nreshape [B, C, X, Y]"]
    BF1 --> BF2 --> BF3 --> BF4
  end

  subgraph TIIM["tiim — PolarRayLifter"]
    TI1["Column transformer\nattend along each image column"]
    TI2["Project to polar ray\nfor each camera ray direction"]
    TI3["Resample from polar\nonto Cartesian BEV grid\nvia F.grid_sample"]
    TI1 --> TI2 --> TI3
  end
```

---

## 3. TPVFormer — tri-plane 3D representation

```mermaid
flowchart TB
  subgraph PLANES["Three orthogonal feature planes"]
    XY["XY plane\n[B, C, X, Y]\ntop-down view"]
    XZ["XZ plane\n[B, C, X, Z]\nfront view"]
    YZ["YZ plane\n[B, C, Y, Z]\nside view"]
  end

  subgraph CROSSVIEW["Cross-view hybrid attention (per layer)"]
    XY2XZ["XY queries attend to\nXZ and YZ planes\n(K,V = other planes)"]
    XZ2XY["XZ queries attend to\nXY and YZ planes"]
    YZ2XY["YZ queries attend to\nXY and XZ planes"]
  end

  subgraph IMGXATTN["Image cross-attention (per layer)"]
    PIL["Pillar deformable attention:\nfor each BEV cell, sample\npoints along its vertical ray\nfrom multi-camera image features"]
  end

  AGG["TPVAggregator\nfor each voxel (x,y,z):\nv = XY[x,y] + XZ[x,z] + YZ[y,z]\n→ [B, C, X, Y, Z]"]

  subgraph OUTPUTS["Task heads"]
    OCC["OccHead\n[B, n_cls, X, Y, Z]\n3D semantic occupancy"]
    PTS["PointHead\n[B, n_cls, P]\nper-point prediction"]
  end

  PLANES --> CROSSVIEW
  CROSSVIEW --> IMGXATTN
  IMGXATTN --> AGG
  AGG --> OCC & PTS
```

---

## 4. Pure-PyTorch deformable attention

```mermaid
flowchart LR
  subgraph DEFORM2D["MSDeformableAttention2D\n(replaces mmcv CUDA kernel)"]
    REF["Reference points\n[B, Q, n_levels, 2]\n(normalised x,y per query per scale)"]
    OFFSET["Sampling offsets\nLinear(C) → [B, Q, n_heads, n_levels, n_points, 2]\n(learned Δx, Δy per head)"]
    SAMPLE_PTS["Reference + offset\n→ absolute sample coordinates"]
    GRID_SAMP["F.grid_sample on each scale\n[B, C_v, H_l, W_l]\n→ sampled features"]
    ATTN_W["Attention weights\nLinear(C) → [B, Q, n_heads, n_levels×n_points]\nsoftmax → normalised"]
    WSUM["Weighted sum over\nsampled features\n→ [B, Q, C_v]"]
    PROJ["Output projection\n[B, Q, C]"]
  end

  REF & OFFSET --> SAMPLE_PTS --> GRID_SAMP
  ATTN_W --> WSUM
  GRID_SAMP --> WSUM --> PROJ
```

---

## 5. Geometric verifier

```mermaid
flowchart TB
  subgraph DATA["SyntheticSceneDataset"]
    RENDER["Ray-cast renderer\ncheckerboard ground\nsky, vehicle and pedestrian boxes\n(random yaw + position)"]
    CAM_RIG["Random camera rig\nN cameras at random positions\n(rig rotated randomly per scene)"]
    LABELS["BEV + 3D occupancy labels\nfrom exact geometry"]
    RENDER & CAM_RIG --> LABELS
  end

  subgraph VERIFY["LiftingVerifier"]
    TRAIN_V["Train lifter end-to-end\n300 steps, BEV seg\nor 3D occupancy (TPVFormer)"]
    CHECK1{"beats_baseline\nmIoU > mlp_view + 0.15?"}
    CHECK2{"uses_geometry\nrotate rig 90° → mIoU drops ≥50%?"}
    CHECK3{"loss_decreases\nsmoothed loss falls ≥30%?"}
    PASS["PASS\nwrite GIF + report.json"]
    FAIL["FAIL\n(lifter does not use calibration)"]
  end

  DATA --> TRAIN_V --> CHECK1 & CHECK2 & CHECK3
  CHECK1 & CHECK2 & CHECK3 -->|"all yes"| PASS
  CHECK1 & CHECK2 & CHECK3 -->|"any no"| FAIL
```

---

## 6. Analytic lifting probes (test suite)

```mermaid
flowchart LR
  subgraph PIXEL_RAMP["Pixel-ramp probe"]
    PR["Set image features = pixel_x / W\n(each pixel has a unique value)"]
    PROJ_PR["Project each voxel to image"]
    EXPECT["Lifted voxel value must equal\nthe exact pixel value it projects to\n(bilinear-exact)"]
    PR --> PROJ_PR --> EXPECT
  end

  subgraph DEPTH_RAMP["Depth-ramp probe"]
    DR["Set image features = depth\n(frustum volume)"]
    PROJ_DR["Splat into voxel grid"]
    EXPECT2["Voxel at depth d must receive\nmass from depth bin containing d"]
    DR --> PROJ_DR --> EXPECT2
  end

  subgraph EQUIVAR["Equivariance probe"]
    EQ["Rotate camera rig by 90° around Z\nkeep same image features"]
    RESULT["Lifted BEV volume must be\nrotated 90° accordingly\n(spatial consistency check)"]
    EQ --> RESULT
  end
```

---

## 7. Training stage configs

```mermaid
flowchart LR
  subgraph PRETRAIN_S["pretrain_lift.yaml"]
    SP["Synthetic dataset\n500K generated scenes\ndepth supervision on LSS frustum"]
  end
  subgraph MIDTRAIN_S["mid_train_nuscenes.yaml\nmid_train_waymo.yaml"]
    SM1["nuScenes v1.0-trainval\n6 cameras, 700 scenes\nday/night/rain/city"]
    SM2["Waymo Open\n5 cameras, 1000 segments\nsuburban + highway"]
  end
  subgraph POSTTRAIN_S["post_train_occ3d.yaml\npost_train_map_seg.yaml"]
    SPT1["nuScenes + Occ3D labels\n17-class 3D occupancy"]
    SPT2["nuScenes map segmentation\n6-class BEV semantic map"]
  end
  PRETRAIN_S -->|"inherits:"| MIDTRAIN_S -->|"inherits:"| POSTTRAIN_S
```

---

## 8. Triplane (TPV) lifting for every technique

`lifting.triplane` gives every lifter a triplane counterpart without editing the existing lifters: a separate
`TRIPLANE_LIFTERS` registry whose classes wrap the unchanged BEV lifters (composition) and add only the parts needed
to keep height (`VoxelToTriplane` for voxel-based lifters, `PlaneQueryEncoder` / `PlaneProjection` for query-based
ones, `BEVToTPV` for the generic adapter). Per-technique explanations, detailed diagrams, animations and results are in
[`triplane.md`](triplane.md).

```mermaid
flowchart TB
  subgraph EXIST["lifting.lifters · LIFTERS registry (existing, unchanged)"]
    direction LR
    L1[simple_bev]:::reused
    L2[depth_warp]:::reused
    L3[lift_splat]:::reused
    L4[tiim]:::reused
    L5[bevformer]:::reused
    L6[tpvformer]:::reused
    L7[alpamayo]:::reused
    L8[mlp_view]:::reused
  end
  subgraph NEW["lifting.triplane · TRIPLANE_LIFTERS registry (new)"]
    direction LR
    T1["SimpleBEVTriplaneLifter"]:::learned
    T2["DepthWarpTriplaneLifter"]:::learned
    T3["LiftSplatTriplaneLifter"]:::learned
    T4["PolarRayTriplaneLifter"]:::learned
    T5["BEVFormerTriplaneLifter"]:::learned
    T6["TPVFormerTriplaneLifter"]:::learned
    T7["AlpamayoTriplaneAdapter"]:::learned
    T8["MLPViewTriplaneLifter"]:::learned
    T9["BEVLifterTriplaneAdapter"]:::learned
  end
  L1 -. "core (composition)" .-> T1
  L2 -. core .-> T2
  L3 -. core .-> T3
  L4 -. core .-> T4
  L6 -. core .-> T6
  L7 -. core .-> T7
  EXIST -. "any entry" .-> T9
  BASE["BaseTriplaneLifter(BaseLifter)<br/>lift_planes() → TPVPlanes<br/>forward() → BEV from planes"]:::geo
  NEW --> BASE
  BASE --> OCC["build_triplane_model → TriplaneOccupancyModel<br/>3D occupancy + point queries"]:::out
  BASE --> BEVM["build_triplane_bev_model → BEVSegmentationModel<br/>BEV segmentation, existing trainer / verifier"]:::out

  classDef data fill:#fff7e6,stroke:#d48806,color:#222
  classDef geo fill:#eaf2fb,stroke:#1f77b4,color:#222
  classDef learned fill:#f1e6f8,stroke:#8e44ad,color:#222
  classDef reused fill:#eaf7ea,stroke:#2e8b57,color:#222
  classDef out fill:#fdecea,stroke:#c0392b,color:#222
```

```mermaid
classDiagram
  direction LR
  class BaseLifter {
    <<existing ABC>>
    +grid: GridSpec
    +uses_geometry: bool
    +forward(features, cameras) Tensor BEV
  }
  class BaseTriplaneLifter {
    <<new ABC>>
    +bev_head: TriplaneBEVHead
    +lift_planes(features, cameras)* TPVPlanes
    +forward(features, cameras) Tensor BEV
    +resolve_kwargs(kwargs, context) dict
  }
  class VoxelToTriplane {
    +forward(voxels) TPVPlanes
  }
  class PlaneQueryEncoder {
    +forward(values, shapes, ref, visible) Tensor
  }
  class PlaneProjection {
    +forward(planes) TPVPlanes
  }
  class TriplaneBEVHead {
    +forward(planes) Tensor
  }
  class TriplaneOccupancyModel {
    +forward(images, cameras, points) dict
  }
  BaseLifter <|-- BaseTriplaneLifter
  BaseTriplaneLifter <|-- SimpleBEVTriplaneLifter
  BaseTriplaneLifter <|-- DepthWarpTriplaneLifter
  BaseTriplaneLifter <|-- LiftSplatTriplaneLifter
  BaseTriplaneLifter <|-- PolarRayTriplaneLifter
  BaseTriplaneLifter <|-- BEVFormerTriplaneLifter
  BaseTriplaneLifter <|-- TPVFormerTriplaneLifter
  BaseTriplaneLifter <|-- AlpamayoTriplaneAdapter
  BaseTriplaneLifter <|-- MLPViewTriplaneLifter
  BaseTriplaneLifter <|-- BEVLifterTriplaneAdapter
  BaseTriplaneLifter *-- TriplaneBEVHead
  SimpleBEVTriplaneLifter *-- VoxelToTriplane
  DepthWarpTriplaneLifter *-- VoxelToTriplane
  LiftSplatTriplaneLifter *-- VoxelToTriplane
  PolarRayTriplaneLifter *-- VoxelToTriplane
  BEVFormerTriplaneLifter *-- PlaneQueryEncoder
  BEVFormerTriplaneLifter *-- PlaneProjection
  TPVFormerTriplaneLifter *-- PlaneProjection
  AlpamayoTriplaneAdapter *-- PlaneProjection
  TriplaneOccupancyModel o-- BaseTriplaneLifter
```

