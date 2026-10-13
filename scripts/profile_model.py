#!/usr/bin/env python3
"""Profile Halos-lift BEV lifter components: depth head, feature backbone, lifter, deformable attention.

Profiles architecture-equivalent stubs so no Halos-lift install is required.
The stubs match the real layer structure used by LiftSplatLifter and the
attention modules.

Usage:
    python scripts/profile_model.py                          # full BEV pipeline stub, CPU
    python scripts/profile_model.py --component depth        # DepthHead only
    python scripts/profile_model.py --component backbone     # image feature backbone
    python scripts/profile_model.py --component bev          # full lift+splat pipeline
    python scripts/profile_model.py --device cuda --torch-profiler --trace out/trace.json
    python scripts/profile_model.py --sort flops --top 30 --json out/profile.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "HaleBlocks" / "src"))

import torch
import torch.nn as nn

from hale_core.profiling import ModelProfiler


# ---------------------------------------------------------------------------
# Stub modules matching Halos-lift architecture
# ---------------------------------------------------------------------------

class StubDepthHead(nn.Module):
    """DepthHead: conv backbone that predicts per-pixel depth distribution + context."""

    def __init__(self, in_channels: int = 64, num_depth_bins: int = 64, context_channels: int = 64):
        super().__init__()
        self.context_net = nn.Sequential(
            nn.Conv2d(in_channels, context_channels, 3, padding=1),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(),
            nn.Conv2d(context_channels, context_channels, 1),
        )
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_depth_bins, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx   = self.context_net(x)
        depth = self.depth_net(x)
        return depth.softmax(dim=1), ctx


class StubImageBackbone(nn.Module):
    """Lightweight ResNet-style image feature extractor."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64, levels: int = 3):
        super().__init__()
        ch = [in_channels, 32, 64, out_channels]
        self.levels = nn.ModuleList()
        for i in range(levels):
            self.levels.append(nn.Sequential(
                nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU(),
                nn.Conv2d(ch[i + 1], ch[i + 1], 3, padding=1),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU(),
            ))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for lvl in self.levels:
            x = lvl(x)
            feats.append(x)
        return feats


class StubHeightCompressor(nn.Module):
    """Collapse voxel height dimension → BEV feature map."""

    def __init__(self, context_ch: int = 64, z_bins: int = 4, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(context_ch * z_bins, out_ch * 2, 3, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StubDeformableAttention(nn.Module):
    """Simplified spatial cross-attention that mimics the deformable attention interface."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_levels: int = 3):
        super().__init__()
        self.query_proj  = nn.Linear(d_model, d_model)
        self.key_proj    = nn.Linear(d_model, d_model)
        self.value_proj  = nn.Linear(d_model, d_model)
        self.out_proj    = nn.Linear(d_model, d_model)
        self.attn        = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm        = nn.LayerNorm(d_model)
        self.ffn         = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(key)
        out, _ = self.attn(q, k, v, need_weights=False)
        out = self.norm(query + self.out_proj(out))
        return out + self.ffn(out)


class StubBEVPipeline(nn.Module):
    """Full BEV pipeline: image backbone → depth head → BEV compressor → deformable cross-attention."""

    def __init__(
        self,
        img_channels: int = 3,
        feat_channels: int = 64,
        bev_channels: int = 64,
        bev_h: int = 50,
        bev_w: int = 50,
        n_cams: int = 6,
        n_attn_layers: int = 2,
        n_heads: int = 4,
        num_depth_bins: int = 32,
    ):
        super().__init__()
        self.backbone     = StubImageBackbone(img_channels, feat_channels)
        self.depth_head   = StubDepthHead(feat_channels, num_depth_bins, feat_channels)
        self.compressor   = StubHeightCompressor(feat_channels, z_bins=4, out_ch=bev_channels)
        self.bev_embed    = nn.Embedding(bev_h * bev_w, bev_channels)
        self.attn_layers  = nn.ModuleList([
            StubDeformableAttention(bev_channels, n_heads) for _ in range(n_attn_layers)
        ])
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, N_cams, C, H, W)
        b, n, c, h, w = images.shape
        imgs_flat = images.reshape(b * n, c, h, w)

        feats = self.backbone(imgs_flat)
        feat  = feats[-1]                     # (B*N, feat_ch, H', W')
        _, ctx = self.depth_head(feat)        # ctx: (B*N, feat_ch, H', W')

        # Fake height compress: reshape context to voxel grid
        ctx_bev = ctx.reshape(b * n, -1, h // 8, w // 8)
        bev     = self.compressor(ctx_bev)    # (B*N, bev_ch, H'', W'')

        # BEV cross-attention query
        bev_flat = bev.reshape(b, n * bev.shape[-2] * bev.shape[-1], -1)
        ids   = torch.arange(self.bev_h * self.bev_w, device=images.device)
        query = self.bev_embed(ids).unsqueeze(0).expand(b, -1, -1)

        for attn in self.attn_layers:
            query = attn(query, bev_flat)

        return query.reshape(b, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_depth(args):
    model = StubDepthHead(in_channels=64, num_depth_bins=args.depth_bins)
    dummy = torch.randn(args.batch_size * args.n_cams, 64, args.feat_h, args.feat_w)
    return model, dummy


def build_backbone(args):
    model = StubImageBackbone(out_channels=64)
    dummy = torch.randn(args.batch_size * args.n_cams, 3, 224, 224)
    return model, dummy


def build_bev(args):
    model = StubBEVPipeline(
        feat_channels=args.feat_channels,
        bev_channels=args.bev_channels,
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        n_cams=args.n_cams,
        n_attn_layers=args.n_attn_layers,
        n_heads=args.n_heads,
        num_depth_bins=args.depth_bins,
    )
    images = torch.randn(args.batch_size, args.n_cams, 3, 224, 224)
    return model, images


_BUILDERS = {"depth": build_depth, "backbone": build_backbone, "bev": build_bev}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Halos-lift profiler")
    p.add_argument("--component",     default="bev",  choices=list(_BUILDERS))
    p.add_argument("--device",        default="cpu")
    p.add_argument("--batch-size",    type=int, default=1)
    p.add_argument("--n-cams",        type=int, default=6,   help="number of camera views")
    p.add_argument("--feat-h",        type=int, default=28,  help="feature map height")
    p.add_argument("--feat-w",        type=int, default=28,  help="feature map width")
    p.add_argument("--feat-channels", type=int, default=64)
    p.add_argument("--bev-channels",  type=int, default=64)
    p.add_argument("--bev-h",         type=int, default=50)
    p.add_argument("--bev-w",         type=int, default=50)
    p.add_argument("--depth-bins",    type=int, default=32)
    p.add_argument("--n-attn-layers", type=int, default=2)
    p.add_argument("--n-heads",       type=int, default=4)
    p.add_argument("--top",           type=int, default=40)
    p.add_argument("--sort",          default="latency", choices=["latency", "flops", "matmuls"])
    p.add_argument("--warmup",        type=int, default=1)
    p.add_argument("--torch-profiler", action="store_true")
    p.add_argument("--trace",         default=None)
    p.add_argument("--json",          default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"\n[profile] Halos-lift  component={args.component}  device={device}  "
          f"batch={args.batch_size}  n_cams={args.n_cams}  "
          f"bev={args.bev_h}x{args.bev_w}  depth_bins={args.depth_bins}")

    model, dummy = _BUILDERS[args.component](args)
    model = model.to(device).eval()
    dummy = dummy.to(device) if isinstance(dummy, torch.Tensor) else tuple(t.to(device) for t in dummy)

    def _fwd():
        return model(*dummy) if isinstance(dummy, tuple) else model(dummy)

    with torch.no_grad():
        for _ in range(args.warmup):
            _fwd()

    profiler = ModelProfiler(
        model,
        use_cuda_events=(device.type == "cuda"),
        use_torch_profiler=args.torch_profiler,
        torch_profiler_trace=args.trace,
    )

    with torch.no_grad(), profiler.profile():
        _fwd()

    report = profiler.report()
    print("\n" + report.summary_table(top_k=args.top, sort_by=args.sort))

    if args.torch_profiler:
        print("\n── torch.profiler key averages ──")
        print(report.torch_key_averages())

    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[profile] JSON → {args.json}")

    if args.trace:
        print(f"[profile] Chrome trace → {args.trace}")


if __name__ == "__main__":
    main()
