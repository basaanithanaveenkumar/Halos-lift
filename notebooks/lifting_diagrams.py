"""Explanatory diagrams for the lifting notebooks (matplotlib only, no data needed).

Every function draws one figure with plt.show(). They are schematic: the geometry is simplified so
the idea is easy to see. All figures use a white background so they read well in light and dark UIs.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

C = dict(
    cam="#333333", ray="#9a9a9a", grid="#d4d4d4", pull="#1f77b4", push="#e8710a", obj="#d62728",
    img="#2ca02c", learn="#8e44ad", geo="#1f77b4", note="#555555", hole="#f4c7c3",
)
FACE = "white"


def _fig(w, h, **kw):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=FACE, **kw)
    return fig, ax


def _arrow(ax, p0, p1, color="k", lw=1.6, style="-|>", ls="-", rad=0.0, ms=12, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                 mutation_scale=ms, connectionstyle=f"arc3,rad={rad}", zorder=z))


def _box(ax, x, y, w, h, text, fc="#eaf2fb", ec="#1f77b4", fs=9, lw=1.6, bold=False, tc="k"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


# ----------------------------------------------------------------------------------------------
# Generic pipeline
# ----------------------------------------------------------------------------------------------
def pipeline(steps, title="", learned=(), width=None):
    """Horizontal flow chart. ``steps``: list of (name, detail). ``learned``: indices drawn purple
    (trainable); the rest are blue (fixed geometry / data)."""
    n = len(steps)
    bw, gap = 2.3, 0.7
    total = n * bw + (n - 1) * gap
    fig, ax = _fig(max(8, width or total * 0.78), 2.5)
    for i, (name, detail) in enumerate(steps):
        x = i * (bw + gap)
        fc, ec = ("#f1e6f8", C["learn"]) if i in learned else ("#eaf2fb", C["geo"])
        _box(ax, x, 0.35, bw, 1.3, f"{name}\n{detail}", fc=fc, ec=ec, fs=8.6)
        if i < n - 1:
            _arrow(ax, (x + bw + 0.04, 1.0), (x + bw + gap - 0.04, 1.0), lw=1.8)
    ax.set_xlim(-0.2, total + 0.2); ax.set_ylim(0, 2.3); ax.axis("off")
    ax.text(0, 2.05, title, fontsize=11, fontweight="bold")
    ax.add_patch(Rectangle((total - 4.6, -0.02), 0.3, 0.17, fc="#f1e6f8", ec=C["learn"], lw=1.2))
    ax.text(total - 4.2, 0.065, "learned", fontsize=8, va="center")
    ax.add_patch(Rectangle((total - 2.9, -0.02), 0.3, 0.17, fc="#eaf2fb", ec=C["geo"], lw=1.2))
    ax.text(total - 2.5, 0.065, "geometry / fixed", fontsize=8, va="center")
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------------------------
# Pull vs push (Simple-BEV / Depth-Warp vs Lift-Splat)
# ----------------------------------------------------------------------------------------------
def _topdown_scene(ax, cell=1.0, nx=7, ny=5, y0=2.0, x0=-3.5, plane_y=1.0, plane_half=1.0, npix=9):
    for i in range(nx + 1):
        ax.plot([x0 + i * cell] * 2, [y0, y0 + ny * cell], color=C["grid"], lw=1, zorder=1)
    for j in range(ny + 1):
        ax.plot([x0, x0 + nx * cell], [y0 + j * cell] * 2, color=C["grid"], lw=1, zorder=1)
    ax.plot([-plane_half, plane_half], [plane_y] * 2, color=C["img"], lw=3, zorder=2)
    for p in np.linspace(-plane_half, plane_half, npix + 1):
        ax.plot([p, p], [plane_y - 0.06, plane_y + 0.06], color=C["img"], lw=1.2, zorder=2)
    ax.plot([0], [0], marker="^", color=C["cam"], ms=13, zorder=6)
    ax.text(0.22, -0.05, "camera", fontsize=9, color=C["cam"], va="center")
    ax.text(plane_half + 0.1, plane_y, "image plane", fontsize=8.5, color=C["img"], va="center")
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-0.7, y0 + ny * cell + 0.5); ax.set_aspect("equal"); ax.axis("off")


def fig_pull_vs_push():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=FACE)
    plane_y, half = 1.0, 1.0
    # ---- PULL ----
    ax = axes[0]; _topdown_scene(ax)
    ax.add_patch(Rectangle((0.5, 4.0), 1, 1, fc="#f8d7d6", ec=C["obj"], lw=2, zorder=1.5))
    ax.text(1.45, 4.88, "car", ha="right", va="top", color=C["obj"], fontsize=9)
    for (vx, vy), col in zip([(-1.5, 3.5), (1.0, 4.5), (2.5, 6.5)], ["#1f77b4", "#c0392b", "#2c3e50"]):
        px = vx * plane_y / vy
        ax.plot([0, vx], [0, vy], ls=":", color="#888", lw=1.1, zorder=2)
        ax.plot([vx], [vy], "s", color=col, ms=9, zorder=5)
        ax.plot([px], [plane_y], "o", color=col, ms=8, zorder=6)
        _arrow(ax, (px + 0.05, plane_y + 0.08), (vx - 0.12, vy - 0.18), color=col, lw=2.0, rad=-0.22, z=4)
    ax.set_title("PULL  (Simple-BEV, Depth-Warp)\nloop over voxels: project the centre, read the pixel", fontsize=11,
                 color=C["pull"], fontweight="bold")
    ax.text(0, -0.55, "one voxel  →  exactly one pixel   (well defined, easy)", ha="center", fontsize=9.5)
    # ---- PUSH ----
    ax = axes[1]; _topdown_scene(ax)
    depths = np.arange(2.2, 7.0, 0.8)
    for k, px in enumerate(np.linspace(-0.8, 0.8, 5)):
        col = plt.cm.tab10(k)
        ax.plot([0, px * 7.2 / plane_y], [0, 7.2], color=C["ray"], lw=0.9, zorder=2)
        ax.plot([px], [plane_y], "o", color=col, ms=7, zorder=6)
        pts = np.array([[px * d / plane_y, d] for d in depths])
        ax.scatter(pts[:, 0], pts[:, 1], s=28, color=col, zorder=5)
    ax.set_title("PUSH  (Lift-Splat)\nloop over pixels: try every depth, drop the point in a voxel",
                 fontsize=11, color=C["push"], fontweight="bold")
    ax.text(0, -0.55, "one pixel  →  many candidate voxels   (depth unknown, hard)", ha="center", fontsize=9.5)
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------------------------
# TPV: three shadows
# ----------------------------------------------------------------------------------------------
def fig_tpv_shadows():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12.5, 5.6), facecolor=FACE)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    P = np.array([0.65, 0.7, 0.55])
    # unit cube wireframe
    for a in (0, 1):
        for b in (0, 1):
            ax.plot([0, 1], [a, a], [b, b], color="#bbbbbb", lw=0.8)
            ax.plot([a, a], [0, 1], [b, b], color="#bbbbbb", lw=0.8)
            ax.plot([a, a], [b, b], [0, 1], color="#bbbbbb", lw=0.8)
    def plane(verts, color, alpha=0.28):
        ax.add_collection3d(Poly3DCollection([verts], facecolors=color, edgecolors=color, alpha=alpha))
    plane([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], "#2ca02c")      # xy at z=0 (floor)
    plane([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)], "#1f77b4")      # xz at y=1 (back wall)
    plane([(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], "#e8710a")      # yz at x=0 (side wall)
    proj = {"xy": (P[0], P[1], 0), "xz": (P[0], 1, P[2]), "yz": (0, P[1], P[2])}
    cols = {"xy": "#2ca02c", "xz": "#1f77b4", "yz": "#e8710a"}
    ax.scatter(*P, color=C["obj"], s=90, zorder=10, depthshade=False)
    ax.text(P[0], P[1], P[2] + 0.08, "  3D point (x, y, z)", color=C["obj"], fontsize=9)
    for k, q in proj.items():
        ax.plot([P[0], q[0]], [P[1], q[1]], [P[2], q[2]], ls="--", color=cols[k], lw=1.5)
        ax.scatter(*q, color=cols[k], s=55, depthshade=False)
    ax.text(0.5, 0.5, 0.0, "xy plane\n(top / BEV)", color="#1d7a1d", fontsize=9, ha="center")
    ax.text(0.5, 1.0, 0.82, "xz plane\n(side)", color="#14527d", fontsize=9, ha="center")
    ax.text(0.0, 0.5, 0.82, "yz plane\n(front)", color="#a5530a", fontsize=9, ha="center")
    ax.set_xlabel("x (forward)"); ax.set_ylabel("y (left)"); ax.set_zlabel("z (up)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1); ax.view_init(elev=22, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("A 3D point casts a 'shadow' on three planes", fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(1, 2, 2); ax2.axis("off"); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.set_title("…so its feature is the SUM of three 2D lookups", fontsize=11, fontweight="bold")
    _box(ax2, 0.3, 7.2, 3.0, 1.7, "xy plane\n(X × Y)\nf_xy[x, y]", fc="#e5f4e5", ec="#2ca02c", fs=10)
    _box(ax2, 3.5, 7.2, 3.0, 1.7, "xz plane\n(X × Z)\nf_xz[x, z]", fc="#e4eef9", ec="#1f77b4", fs=10)
    _box(ax2, 6.7, 7.2, 3.0, 1.7, "yz plane\n(Y × Z)\nf_yz[y, z]", fc="#fdecd9", ec="#e8710a", fs=10)
    for x in (1.8, 5.0, 8.2):
        _arrow(ax2, (x, 7.15), (5.0, 5.75), color="#555")
    _box(ax2, 2.2, 4.6, 5.6, 1.1, "f(x, y, z) = f_xy + f_xz + f_yz", fc="#fff5d6", ec="#b8860b", fs=11, bold=True)
    _arrow(ax2, (5.0, 4.55), (5.0, 3.55), color="#555")
    _box(ax2, 2.2, 2.4, 5.6, 1.1, "MLP  →  class (empty / car / person)", fc="#f1e6f8", ec=C["learn"], fs=10.5)
    ax2.text(5.0, 1.3, "Memory: X·Y + X·Z + Y·Z   instead of   X·Y·Z", ha="center", fontsize=10.5,
             color=C["note"])
    ax2.text(5.0, 0.55, "(200×200×16: 46,400 cells instead of 640,000)", ha="center", fontsize=9, color=C["note"])
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------------------------
# Depth-Warp: the frustum and the voxel pulling from it
# ----------------------------------------------------------------------------------------------
def fig_frustum_pull():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=FACE, gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    fov = np.deg2rad(50)
    dmax, nb = 7.0, 7
    edges = np.linspace(1.0, dmax, nb + 1)
    npix = 6
    u_edges = np.linspace(-np.tan(fov), np.tan(fov), npix + 1)
    # frustum cells (u slice × depth slice)
    rng = np.random.default_rng(1)
    for i in range(npix):
        for j in range(nb):
            d0, d1 = edges[j], edges[j + 1]
            poly = [(u_edges[i] * d0, d0), (u_edges[i + 1] * d0, d0), (u_edges[i + 1] * d1, d1), (u_edges[i] * d1, d1)]
            shade = 0.06
            ax.add_patch(Polygon(poly, fc=(0.12, 0.47, 0.71, shade), ec="#9db9d6", lw=0.6, zorder=1))
    # highlight the depth distribution of one pixel column (i=3) with a peaked p(d)
    i = 3
    p = np.exp(-0.5 * ((np.arange(nb) - 3) / 0.9) ** 2); p /= p.sum()
    for j in range(nb):
        d0, d1 = edges[j], edges[j + 1]
        poly = [(u_edges[i] * d0, d0), (u_edges[i + 1] * d0, d0), (u_edges[i + 1] * d1, d1), (u_edges[i] * d1, d1)]
        ax.add_patch(Polygon(poly, fc=(0.9, 0.44, 0.04, min(0.95, 0.12 + 2.2 * p[j])), ec="#e8710a", lw=0.8, zorder=2))
    # voxel grid
    for gx in np.arange(-4, 5):
        ax.plot([gx, gx], [0.0, 7.5], color=C["grid"], lw=0.7, zorder=0)
    for gy in np.arange(0, 8):
        ax.plot([-4, 4], [gy, gy], color=C["grid"], lw=0.7, zorder=0)
    ax.plot([0], [0], marker="^", color=C["cam"], ms=13, zorder=6)
    # one voxel pulling
    vx, vy = 0.5, 4.5
    ax.add_patch(Rectangle((vx - 0.5, vy - 0.5), 1, 1, fc="none", ec=C["pull"], lw=2.5, zorder=5))
    ax.plot([vx], [vy], "s", color=C["pull"], ms=7, zorder=6)
    ax.annotate("voxel centre\n(x, y, z)", (vx + 0.1, vy), xytext=(2.6, 2.0), fontsize=9.5, color=C["pull"],
                bbox=dict(boxstyle="round", fc="white", ec=C["pull"]), arrowprops=dict(arrowstyle="->", color=C["pull"]))
    ax.plot([0, vx], [0, vy], ls="--", color=C["pull"], lw=1.2, zorder=4)
    ax.text(-3.9, 6.9, "project → (u, v, depth)\nthen trilinear read of the\nfrustum cell at that spot", fontsize=9.3,
            color=C["pull"], va="top")
    ax.text(-1.9, 0.9, "frustum = (u, v, depth)\ngrid in front of the camera", fontsize=9, color="#4a6f94", ha="center")
    ax.set_xlim(-4.3, 4.3); ax.set_ylim(-0.6, 7.7); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Depth-Warp: voxels PULL from the camera frustum (top view)", fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.barh(np.arange(nb), p, color=[(0.9, 0.44, 0.04, 0.9)] * nb, edgecolor="#b35a0a")
    ax.set_yticks(np.arange(nb)); ax.set_yticklabels([f"{(edges[j] + edges[j + 1]) / 2:.1f} m" for j in range(nb)], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("p(depth) of ONE pixel (softmax over bins)")
    ax.set_title("The depth head's answer for one pixel", fontsize=11, fontweight="bold")
    ax.text(0.99, 0.02, "frustum feature at bin d  =\n  p(d) × context vector", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round", fc="#fff5d6", ec="#b8860b"))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------------------------
# Lift-Splat: points pushed into voxels, with holes
# ----------------------------------------------------------------------------------------------
def fig_splat_holes():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=FACE)
    fov = np.deg2rad(50)
    npix, depths = 9, np.arange(1.5, 9.5, 0.8)
    us = np.linspace(-np.tan(fov), np.tan(fov), npix)
    pts = np.array([[u * d, d] for u in us for d in depths])
    xs, ys = np.arange(-5, 6), np.arange(0, 10)
    counts = np.zeros((len(xs) - 1, len(ys) - 1), int)
    for x, y in pts:
        i, j = int(np.floor(x + 5)), int(np.floor(y))
        if 0 <= i < counts.shape[0] and 0 <= j < counts.shape[1]:
            counts[i, j] += 1
    ax = axes[0]
    for gx in xs: ax.plot([gx, gx], [0, 9], color=C["grid"], lw=0.8)
    for gy in ys: ax.plot([-5, 5], [gy, gy], color=C["grid"], lw=0.8)
    for u in us:
        ax.plot([0, u * 9.2], [0, 9.2], color=C["ray"], lw=0.7)
    ax.scatter(pts[:, 0], pts[:, 1], s=16, color=C["push"], zorder=5)
    ax.plot([0], [0], marker="^", color=C["cam"], ms=13, zorder=6)
    ax.set_xlim(-5.2, 5.2); ax.set_ylim(-0.6, 9.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("LIFT: one 3D point per (pixel, depth bin)", fontsize=11, fontweight="bold", color=C["push"])
    ax.text(0, -0.5, "near the camera the points are dense, far away they spread out", ha="center", fontsize=9.3)

    ax = axes[1]
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            c = counts[i, j]
            fc = C["hole"] if c == 0 else plt.cm.Oranges(min(0.15 + 0.12 * c, 0.95))
            ax.add_patch(Rectangle((xs[i], ys[j]), 1, 1, fc=fc, ec="white", lw=1))
            if c: ax.text(xs[i] + 0.5, ys[j] + 0.5, str(c), ha="center", va="center", fontsize=7.5)
    ax.plot([0], [0], marker="^", color=C["cam"], ms=13, zorder=6)
    ax.set_xlim(-5.2, 5.2); ax.set_ylim(-0.6, 9.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("SPLAT: sum the points that land in each voxel", fontsize=11, fontweight="bold", color=C["push"])
    ax.add_patch(Rectangle((-5, -0.55), 0.5, 0.4, fc=C["hole"], ec="#c0392b"))
    ax.text(-4.35, -0.35, "pink = HOLE (no point landed)", fontsize=9, va="center")
    ax.text(5.1, -0.35, "numbers = points per voxel", fontsize=9, va="center", ha="right")
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------------------------
# TIIM: column -> ray
# ----------------------------------------------------------------------------------------------
def fig_tiim_column_to_ray():
    fig = plt.figure(figsize=(14, 4.9), facecolor=FACE)
    # panel 1: image
    ax1 = fig.add_axes([0.02, 0.1, 0.27, 0.78])
    H, W = 40, 64
    img = np.zeros((H, W, 3))
    img[:18] = [0.62, 0.78, 0.95]
    for r in range(18, H):
        g = 0.45 + 0.35 * (r - 18) / (H - 18)
        img[r] = [g, g, g]
    img[22:27, 36:42] = [0.84, 0.15, 0.12]
    ax1.imshow(img, extent=[0, W, H, 0]); ax1.axvline(39, color="#ffd400", lw=3)
    ax1.set_title("1. camera image: pick ONE column", fontsize=10.5, fontweight="bold")
    ax1.annotate("", xy=(54, 38), xytext=(54, 19), arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
    ax1.text(52.5, 28.5, "closer", color="white", rotation=90, ha="right", va="center", fontsize=9)
    ax1.text(2, 16.5, "far / sky", fontsize=8.5, color="#1a3d63"); ax1.set_xticks([]); ax1.set_yticks([])
    # panel 2: polar
    ax2 = fig.add_axes([0.385, 0.1, 0.235, 0.78])
    D, Wp = 10, 12
    rng = np.random.default_rng(3)
    polar = rng.uniform(0.15, 0.35, (D, Wp)); polar[:, 7] = np.linspace(0.95, 0.35, D)
    ax2.imshow(polar, cmap="Purples", vmin=0, vmax=1.1, aspect="auto", extent=[0, Wp, D, 0])
    ax2.add_patch(Rectangle((7, 0), 1, D, fill=False, ec="#ffb300", lw=3))
    ax2.set_title("2. polar map (depth × column)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("image column  u"); ax2.set_yticks([0.5, D - 0.5]); ax2.set_yticklabels(["near", "far"])
    # panel 3: BEV fan
    ax3 = fig.add_axes([0.70, 0.04, 0.29, 0.86])
    cols = np.linspace(-0.9, 0.9, Wp + 1)
    for k in range(Wp):
        for j in range(D):
            r0, r1 = 1 + j * 0.7, 1 + (j + 1) * 0.7
            a0, a1 = np.arctan(cols[k]), np.arctan(cols[k + 1])
            poly = [(r0 * np.sin(a0), r0 * np.cos(a0)), (r1 * np.sin(a0), r1 * np.cos(a0)),
                    (r1 * np.sin(a1), r1 * np.cos(a1)), (r0 * np.sin(a1), r0 * np.cos(a1))]
            hl = k == 7
            ax3.add_patch(Polygon(poly, fc=plt.cm.Purples(0.25 + 0.5 * (1 - j / D)) if hl else "#f3f3f3",
                                  ec="#ffb300" if hl else "white", lw=1.4 if hl else 0.6))
    ax3.plot([0], [0], marker="^", color=C["cam"], ms=12)
    ax3.set_xlim(-6, 6); ax3.set_ylim(-0.5, 8.5); ax3.set_aspect("equal"); ax3.axis("off")
    ax3.set_title("3. the same column = one RAY on the map", fontsize=10.5, fontweight="bold")
    fig.add_artist(FancyArrowPatch((0.30, 0.5), (0.375, 0.5), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=18, lw=2, color=C["learn"]))
    fig.text(0.3375, 0.56, "Transformer", ha="center", fontsize=8.5, color=C["learn"], fontweight="bold")
    fig.text(0.3375, 0.44, "H pixels\n→ D depths", ha="center", fontsize=8, color=C["learn"], va="top")
    fig.add_artist(FancyArrowPatch((0.625, 0.5), (0.69, 0.5), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=18, lw=2, color=C["geo"]))
    fig.text(0.657, 0.56, "resample", ha="center", fontsize=8.5, color=C["geo"], fontweight="bold")
    fig.text(0.657, 0.44, "(geometry)", ha="center", fontsize=8, color=C["geo"], va="top")
    plt.show()


# ----------------------------------------------------------------------------------------------
# BEVFormer: queries -> pillar -> projection -> deformable sampling
# ----------------------------------------------------------------------------------------------
def fig_bevformer_mechanism():
    fig = plt.figure(figsize=(14.5, 5.0), facecolor=FACE)
    # panel 1: BEV queries with self-attention
    ax1 = fig.add_axes([0.02, 0.08, 0.24, 0.8])
    n = 7
    for i in range(n):
        for j in range(n):
            ax1.add_patch(Rectangle((i, j), 1, 1, fc="#f6f6f6", ec="white", lw=1.5))
    ax1.add_patch(Rectangle((3, 3), 1, 1, fc="#f1e6f8", ec=C["learn"], lw=3))
    rng = np.random.default_rng(5)
    for dx, dy in [(-2, 1), (1, 2), (2, -1), (-1, -2)]:
        ax1.add_patch(Rectangle((3 + dx, 3 + dy), 1, 1, fc="#e0cdf0", ec="white", lw=1.5))
        _arrow(ax1, (3.5, 3.5), (3.5 + dx, 3.5 + dy), color=C["learn"], lw=1.4, rad=0.15)
    ax1.set_xlim(0, n); ax1.set_ylim(0, n); ax1.set_aspect("equal"); ax1.axis("off")
    ax1.set_title("1. one learned QUERY per BEV cell\n+ self-attention to nearby cells", fontsize=10, fontweight="bold")

    # panel 2: side view
    ax2 = fig.add_axes([0.29, 0.08, 0.40, 0.8])
    cam = np.array([0.0, 1.6]); plane_x = 2.0
    ax2.axhline(0, color="#777", lw=1.2); ax2.text(10.5, -0.35, "ground", fontsize=8.5, ha="right", color="#777")
    ax2.plot(*cam, marker="^", color=C["cam"], ms=12); ax2.text(0.1, 1.95, "camera", fontsize=9, color=C["cam"])
    ax2.plot([plane_x, plane_x], [-0.1, 3.1], color=C["img"], lw=3); ax2.text(plane_x, 3.25, "image", ha="center", fontsize=9, color=C["img"])
    px = 8.0
    ax2.add_patch(Rectangle((px - 0.35, 0), 0.7, 3.4, fc="#f1e6f8", ec=C["learn"], lw=1.5, alpha=0.6))
    ax2.text(px, 3.55, "BEV cell = a vertical PILLAR", ha="center", fontsize=9.3, color=C["learn"], fontweight="bold")
    zs = [-0.5, 0.5, 1.5, 2.5]
    for z in zs:
        ax2.plot([px], [z + 0.5], "o", color=C["obj"], ms=7, zorder=6)
        t = (plane_x - cam[0]) / (px - cam[0])
        yimg = cam[1] + (z + 0.5 - cam[1]) * t
        ax2.plot([cam[0], px], [cam[1], z + 0.5], ls="--", color="#999", lw=0.9, zorder=2)
        ax2.plot([plane_x], [yimg], "x", color=C["obj"], ms=8, mew=2, zorder=7)
    ax2.text(px, -0.75, "4 anchors  (A = 4)", ha="center", fontsize=9, color=C["obj"])
    ax2.text(5.2, 2.8, "project every anchor\ninto the image", fontsize=9.2, color=C["note"], ha="center")
    ax2.set_xlim(-0.5, 10.6); ax2.set_ylim(-1.1, 4.0); ax2.set_aspect("equal"); ax2.axis("off")
    ax2.set_title("2. 3D anchors above the cell → projected pixels (geometry)", fontsize=10, fontweight="bold")

    # panel 3: zoomed image with deformable samples
    ax3 = fig.add_axes([0.73, 0.08, 0.26, 0.8])
    ax3.add_patch(Rectangle((0, 0), 10, 6, fc="#cfe1f3", ec="#555"))
    ax3.add_patch(Rectangle((0, 0), 10, 2.4, fc="#9a9a9a", ec="none"))
    ax3.add_patch(Rectangle((5.6, 2.4), 2.2, 1.3, fc=C["obj"], ec="none"))
    anchors = np.array([[4.9, 1.2], [4.9, 2.4], [4.9, 3.6], [4.9, 4.8]])
    ax3.scatter(anchors[:, 0], anchors[:, 1], marker="x", s=80, c=C["obj"], linewidths=2.5, zorder=6)
    rng = np.random.default_rng(2)
    for h, col in enumerate(["#1f77b4", "#2ca02c", "#e8710a", "#8e44ad"]):
        a = anchors[h]
        offs = rng.normal(0, 0.55, (2, 2)) + [[0.8, 0.1]]
        for o in offs:
            ax3.annotate("", xy=a + o, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2), zorder=5)
            ax3.scatter(*(a + o), s=60 + 160 * rng.uniform(0.3, 1), color=col, alpha=0.8, zorder=6)
    ax3.set_xlim(-0.2, 10.2); ax3.set_ylim(-0.2, 6.2); ax3.set_aspect("equal"); ax3.axis("off")
    ax3.set_title("3. learned OFFSETS + weights\n(deformable attention reads here)", fontsize=10, fontweight="bold")
    ax3.text(5, -0.65, "✕ geometry     ● learned sample (size = attention weight)", ha="center", fontsize=8.5)
    fig.add_artist(FancyArrowPatch((0.265, 0.48), (0.30, 0.48), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, lw=2))
    fig.add_artist(FancyArrowPatch((0.695, 0.48), (0.735, 0.48), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, lw=2))
    plt.show()


# ----------------------------------------------------------------------------------------------
# TPV layer
# ----------------------------------------------------------------------------------------------
def fig_tpv_layer():
    fig, ax = _fig(14, 5.4)
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.6); ax.axis("off")
    _box(ax, 0.1, 3.7, 2.5, 1.2, "learned queries\n(xy | xz | yz tokens)", fc="#f1e6f8", ec=C["learn"], fs=9.2)
    _box(ax, 0.1, 0.5, 2.5, 1.3, "N camera images\n→ image encoder\n→ feature pyramid", fc="#e5f4e5", ec=C["img"], fs=9.2)
    xs = [3.5, 6.2, 8.9]
    names = [("Cross-view\nhybrid attention", "the 3 planes exchange info\n(deformable self-attention,\nplanes = 3 'levels')"),
             ("Image\ncross-attention", "per plane: project 3D anchors\ninto the cameras, read pixels\n(deformable, masked by view)"),
             ("Feed-forward\nnetwork", "per-token MLP")]
    for x, (a_, b_) in zip(xs, names):
        _box(ax, x, 3.5, 2.2, 1.6, a_, fc="#f1e6f8", ec=C["learn"], fs=10, bold=True)
        ax.text(x + (0.45 if "Image" in a_ else 1.1), 3.25, b_, ha="left" if "Image" in a_ else "center", va="top", fontsize=8.2, color=C["note"])
    for x in xs[:-1]:
        _arrow(ax, (x + 2.23, 4.3), (x + 2.67, 4.3))
    _arrow(ax, (2.65, 4.3), (3.47, 4.3))
    ax.plot([2.65, 6.45], [1.15, 1.15], color=C["img"], lw=2)
    _arrow(ax, (6.45, 1.15), (6.45, 3.45), color=C["img"], lw=2)
    ax.text(4.9, 0.75, "image features (keys / values)", fontsize=8.8, color=C["img"], ha="center")
    ax.add_patch(FancyBboxPatch((3.35, 3.35), 8.0, 1.9, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none", ec="#999", lw=1.2, ls="--"))
    ax.text(11.3, 5.4, "× num_layers", fontsize=9.5, color="#555", ha="right")
    _arrow(ax, (11.4, 4.3), (11.95, 4.3))
    _box(ax, 12.0, 3.7, 1.8, 1.2, "TPV planes\nxy · xz · yz", fc="#fff5d6", ec="#b8860b", fs=9.3, bold=True)
    _arrow(ax, (12.9, 3.65), (12.9, 2.75))
    _box(ax, 11.85, 1.6, 2.1, 1.1, "Σ of 3 lookups\n→ MLP → classes", fc="#f1e6f8", ec=C["learn"], fs=8.8)
    ax.set_title("Inside TPVFormer: one encoder layer (repeated), then the aggregator", fontsize=11, fontweight="bold", loc="left")
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------------------------
# Comparison notebook
# ----------------------------------------------------------------------------------------------
def fig_lifter_family():
    fig, ax = _fig(14, 6.6)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
    _box(ax, 4.7, 5.5, 4.6, 0.9, "Goal: 2D image features  →  BEV / 3D features", fc="#fff5d6", ec="#b8860b", fs=11, bold=True)
    _box(ax, 0.2, 3.7, 3.0, 1.1, "No geometry\nmlp_view  (control)", fc="#f2f2f2", ec="#777", fs=9.5)
    _box(ax, 3.5, 3.7, 3.0, 1.1, "Geometry only\nsimple_bev", fc="#eaf2fb", ec=C["geo"], fs=9.5, bold=True)
    _box(ax, 6.8, 3.7, 3.4, 1.1, "Explicit depth\n(depth distribution)", fc="#fdecd9", ec=C["push"], fs=9.5, bold=True)
    _box(ax, 10.5, 3.7, 3.3, 1.1, "Attention does the\nwork (queries)", fc="#f1e6f8", ec=C["learn"], fs=9.5, bold=True)
    for x in (1.7, 5.0, 8.5, 12.1):
        _arrow(ax, (7.0, 5.45), (x, 4.85), color="#666", lw=1.3)
    _box(ax, 3.4, 1.9, 2.7, 1.0, "tiim\ncolumn → ray\n(row ↔ depth)", fc="#eaf2fb", ec=C["geo"], fs=8.8)
    _arrow(ax, (5.0, 3.65), (5.0, 2.95), color="#666", lw=1.2)
    _box(ax, 6.5, 1.9, 1.9, 1.0, "depth_warp\nPULL", fc="#fdecd9", ec=C["pull"], fs=9)
    _box(ax, 8.6, 1.9, 1.9, 1.0, "lift_splat\nPUSH", fc="#fdecd9", ec=C["push"], fs=9)
    _arrow(ax, (8.5, 3.65), (7.4, 2.95), color="#666", lw=1.2); _arrow(ax, (8.5, 3.65), (9.5, 2.95), color="#666", lw=1.2)
    _box(ax, 10.7, 1.9, 1.5, 1.0, "bevformer\nBEV plane", fc="#f1e6f8", ec=C["learn"], fs=9)
    _box(ax, 12.3, 1.9, 1.5, 1.0, "tpvformer\n3 planes", fc="#f1e6f8", ec=C["learn"], fs=9)
    _arrow(ax, (12.1, 3.65), (11.45, 2.95), color="#666", lw=1.2); _arrow(ax, (12.1, 3.65), (13.05, 2.95), color="#666", lw=1.2)
    ax.text(0.2, 0.9, "Reading the tree: moving right = more learned machinery, more parameters, more compute, more data needed.\n"
                      "Every geometric lifter must beat the control (mlp_view) AND collapse when the camera poses are rotated.",
            fontsize=9.3, color=C["note"], va="center")
    ax.set_title("The lifter family", fontsize=12, fontweight="bold", loc="left")
    plt.tight_layout(); plt.show()


def fig_decision_flow():
    fig, ax = _fig(14.5, 5.0)
    ax.set_xlim(0, 14.5); ax.set_ylim(0, 5.0); ax.axis("off")
    qs = ["Is a simple\nbaseline enough?", "Need sharper objects /\ndepth reasoning?",
          "Many cameras, long range,\nlots of data?", "Need height / 3D occupancy /\nany-point queries?"]
    ans = [("simple_bev", C["geo"]), ("depth_warp\nor lift_splat", C["push"]), ("bevformer", C["learn"]), ("tpvformer", C["learn"])]
    x0, w, gap = 0.2, 3.1, 0.55
    for i, (qt, (at, ac)) in enumerate(zip(qs, ans)):
        x = x0 + i * (w + gap)
        _box(ax, x, 3.5, w, 1.1, ("START: " if i == 0 else "") + qt, fc="#fff5d6", ec="#b8860b", fs=9.4)
        _box(ax, x + 0.45, 1.85, w - 0.9, 0.85, at, fc="#eaf2fb", ec=ac, fs=10, bold=True)
        _arrow(ax, (x + w / 2, 3.45), (x + w / 2, 2.75), color="#555"); ax.text(x + w / 2 + 0.12, 3.08, "yes", fontsize=9)
        if i < 3:
            _arrow(ax, (x + w + 0.02, 4.05), (x + w + gap - 0.02, 4.05)); ax.text(x + w + gap / 2, 4.2, "no", fontsize=8.5, ha="center")
    _box(ax, 0.2, 0.25, 14.1, 0.95, "Always keep mlp_view as the control: if your lifter doesn't beat it, or doesn't collapse under rotated extrinsics, "
                                    "it is NOT using the camera geometry.", fc="#f2f2f2", ec="#777", fs=9.6)
    ax.set_title("Which lifter should I try? (a rule of thumb, not a law)", fontsize=12, fontweight="bold", loc="left")
    plt.tight_layout(); plt.show()


def fig_mlp_vs_geometry():
    fig = plt.figure(figsize=(14, 4.4), facecolor=FACE)
    ax = fig.add_axes([0.0, 0.05, 0.40, 0.85]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_title("MLP-View: a fixed, learned table", fontsize=11, fontweight="bold")
    for i in range(3):
        for j in range(4):
            ax.add_patch(Rectangle((0.3 + j * 0.6, 3.2 + i * 0.6), 0.6, 0.6, fc="#dfeccf", ec="white", lw=1.2))
    ax.text(1.5, 2.85, "image cells\n(per camera)", ha="center", va="top", fontsize=8.5)
    _box(ax, 3.4, 3.0, 2.5, 2.0, "Linear\nh·w → X·Y\n(never sees\nthe camera pose)", fc="#f2f2f2", ec="#777", fs=8.8)
    _arrow(ax, (2.9, 4.0), (3.35, 4.0))
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((7.0 + j * 0.55, 3.0 + i * 0.55), 0.55, 0.55, fc="#f3d9d6", ec="white", lw=1.2))
    ax.text(8.1, 2.65, "BEV cells", ha="center", va="top", fontsize=8.5)
    _arrow(ax, (5.95, 4.0), (6.95, 4.0))
    ax.text(5, 1.0, "same pixel cell → same BEV cells, always.", ha="center", fontsize=9.3, color=C["note"])

    th = np.deg2rad(np.linspace(-50, 50, 40))
    for k, (yaw, title) in enumerate([(0, "scene A: rig yaw 0°  (training)"), (70, "scene B: rig yaw 70°  (new pose)")]):
        ax = fig.add_axes([0.42 + k * 0.29, 0.05, 0.28, 0.85])
        r = np.deg2rad(yaw)
        ax.fill(np.r_[0, 6 * np.sin(th + r), 0], np.r_[0, 6 * np.cos(th + r), 0], color="#cfe1f3", alpha=0.8)
        pa = r + np.deg2rad(18)                                    # the ray of one fixed pixel
        ax.plot([0, 6.4 * np.sin(pa)], [0, 6.4 * np.cos(pa)], color=C["push"], lw=2.5)
        ax.plot([0], [0], marker="^", color=C["cam"], ms=12)
        car = np.array([4.2 * np.sin(np.deg2rad(18)), 4.2 * np.cos(np.deg2rad(18))])           # car is where the table says
        ax.add_patch(Rectangle(car - 0.4, 0.8, 0.8, fc=C["obj"], ec="none", alpha=0.9 if k == 0 else 0.35))
        ax.text(*(car + [0.6, 0.0]), "table says:\n'car here'", fontsize=8.3, color=C["obj"], va="center")
        if k == 1:
            ax.plot([4.2 * np.sin(pa)], [4.2 * np.cos(pa)], "x", color=C["push"], ms=12, mew=3)
            ax.text(4.2 * np.sin(pa) + 0.4, 4.2 * np.cos(pa) + 0.5, "but this pixel\nnow sees here", fontsize=8.3, color=C["push"])
        ax.set_xlim(-5, 8); ax.set_ylim(-1, 7); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold")
    plt.show()
