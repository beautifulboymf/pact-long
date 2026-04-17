"""Render Figure 2 — PACT training-step diagram (v4: hierarchy + legibility).

Every block shares the same 6-node node-graph spine so the figure reads as
one system, but the two PACT plug-ins (GPE, VWL) are visibly larger and
carry their own internal operation diagram + formula, while the scaffold
blocks (X, p(Z*), M, μ̂, D, pilot, σ̂², w_i) are compact carriers.

Run from pact-long/:
    python images/scripts/make_arch.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PDF = OUT_DIR / "arch.pdf"
OUT_PNG = OUT_DIR / "arch_preview.png"

# ---------------- palette ----------------
BG = "#FFFFFF"
BIAS = "#2A8A8A"
BIAS_LT = "#8FC3C3"
BIAS_FILL = "#E2EFEF"
BIAS_DK = "#1C5E5E"
VAR = "#8A4FAD"
VAR_LT = "#C29FD6"
VAR_FILL = "#EEE1F3"
VAR_DK = "#5E2F7A"
WARM = "#C47642"
WARM_DK = "#8E4D1E"
CUT_RED = "#C94040"
NEU_EDGE = "#7E7E7E"
NEU_SOFT = "#B5B5B5"
NEU_DEEP = "#7F7F7F"
NEU_FILL = "#F4F2EE"
DIVIDER = "#CCCCCC"
INK = "#1A1A1A"
INK_SOFT = "#4A4A4A"
INK_FAINT = "#8A8A8A"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

# ---------------- shared node-graph spine ----------------
NODE_POS = np.array(
    [
        (-2.4, 1.9),
        (0.0, 2.7),
        (2.4, 1.7),
        (-2.0, -1.5),
        (0.7, -2.3),
        (2.6, -0.6),
    ]
)
EDGES = [(0, 1), (1, 2), (2, 5), (0, 3), (3, 4), (4, 5), (1, 3), (1, 4)]

SIGMAS = np.array([0.35, 0.95, 0.55, 0.75, 0.25, 0.65])
RESID_SIGN = np.array([+1, -1, +1, -1, +1, -1])
RESID_MAG = np.array([0.60, 0.45, 1.00, 0.55, 0.85, 0.40])
POS_SHADE = np.array([0.30, 0.85, 0.55, 0.70, 0.40, 0.65])


def _hex2rgb(c):
    c = c.lstrip("#")
    return np.array([int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _mix(c1, c2, t):
    a = _hex2rgb(c1)
    b = _hex2rgb(c2)
    return tuple(a * t + b * (1.0 - t))


def mininet(
    ax,
    cx,
    cy,
    *,
    node_face=NEU_SOFT,
    node_edge=None,
    node_size=0.36,
    node_sizes=None,
    node_faces=None,
    node_edges=None,
    node_dashed=False,
    halo_color=None,
    halo_sizes=None,
    halo_alpha=0.45,
    twin_colors=None,
    edge_color=NEU_EDGE,
    edge_alpha=0.55,
    edge_lw=0.7,
    edge_style="-",
    scale=1.0,
    spread_nodes=False,   # if True, `scale` spreads node POSITIONS only;
                          # node radii stay absolute. Use for "bigger graph,
                          # same-size nodes".
    zorder=5,
):
    pts = NODE_POS * scale + np.array([cx, cy])
    for a, b in EDGES:
        ax.plot(
            [pts[a][0], pts[b][0]],
            [pts[a][1], pts[b][1]],
            color=edge_color,
            lw=edge_lw,
            alpha=edge_alpha,
            linestyle=edge_style,
            zorder=zorder,
            solid_capstyle="round",
        )
    if halo_color is not None:
        sizes = halo_sizes if halo_sizes is not None else [0.72 * scale] * 6
        for p, s in zip(pts, sizes):
            ax.add_patch(
                mpatches.Circle(
                    p, s,
                    facecolor=halo_color, alpha=halo_alpha,
                    edgecolor="none", zorder=zorder + 0.05,
                )
            )
    if twin_colors is not None:
        chip_w = 0.38 * scale
        chip_h = 0.34 * scale
        for p in pts:
            for i, col in enumerate(twin_colors):
                ax.add_patch(
                    mpatches.Rectangle(
                        (p[0] + 0.45 * scale,
                         p[1] + (0.5 - i) * 0.52 * scale - 0.17 * scale),
                        chip_w, chip_h,
                        facecolor=col, edgecolor="none", alpha=0.92,
                        zorder=zorder + 0.15,
                    )
                )
    size_mult = 1.0 if spread_nodes else scale
    sizes = node_sizes if node_sizes is not None else [node_size * size_mult] * 6
    if node_sizes is not None:
        sizes = [s * size_mult for s in node_sizes]
    faces = node_faces if node_faces is not None else [node_face] * 6
    edges_c = (
        node_edges
        if node_edges is not None
        else [node_edge if node_edge is not None else f for f in faces]
    )
    ls = (0, (2.2, 1.6)) if node_dashed else "-"
    for p, s, f, e in zip(pts, sizes, faces, edges_c):
        ax.add_patch(
            mpatches.Circle(
                p, s,
                facecolor=f, edgecolor=e, linewidth=0.55,
                linestyle=ls, zorder=zorder + 0.2,
            )
        )


# ---------------- container primitives ----------------
def block(ax, x, y, w, h, *, accent=None, accent_fill=None,
          dashed=False, zorder=3, lw=None):
    ec = accent if accent is not None else NEU_EDGE
    fc = accent_fill if accent_fill is not None else NEU_FILL
    if lw is None:
        lw = 1.6 if accent is not None else 0.7
    ls = (0, (4, 2)) if dashed else "-"
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            linewidth=lw, edgecolor=ec, facecolor=fc,
            linestyle=ls, zorder=zorder,
        )
    )


def arrow(ax, p0, p1, *, colour=INK_SOFT, lw=1.2, dashed=False,
          zorder=5, head_size=1.0, rad=0.0):
    ls = (0, (3.5, 2)) if dashed else "-"
    cs = f"arc3,rad={rad}" if rad != 0 else None
    props = dict(
        arrowstyle=f"-|>,head_length={0.55 * head_size},head_width={0.36 * head_size}",
        color=colour, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
    )
    if cs is not None:
        props["connectionstyle"] = cs
    ax.annotate("", xy=p1, xytext=p0, arrowprops=props, zorder=zorder)


def matrix_icon(ax, cx, cy, *, w=3.0, h=3.0, rows=3, cols=3,
                colour=NEU_DEEP, seed=1, zorder=5):
    """Small feature-matrix visualization (distinct from the node-graph spine)."""
    rng = np.random.default_rng(seed)
    cell_w = w / cols
    cell_h = h / rows
    x0 = cx - w / 2
    y0 = cy - h / 2
    for i in range(rows):
        for j in range(cols):
            a = 0.25 + 0.70 * rng.random()
            ax.add_patch(mpatches.Rectangle(
                (x0 + j * cell_w, y0 + i * cell_h),
                cell_w * 0.86, cell_h * 0.86,
                facecolor=colour, alpha=a, edgecolor="none",
                zorder=zorder))


def bar_icon(ax, cx, cy, *, w=5.0, h=5.0, heights, colour=VAR,
             edge=VAR_DK, zorder=5):
    """Vertical bar chart — variety primitive."""
    n = len(heights)
    bar_w = w / n * 0.70
    gap = w / n * 0.30
    x0 = cx - w / 2
    y_base = cy - h / 2
    for i, hh in enumerate(heights):
        bar_h = 0.3 + (h - 0.4) * hh
        ax.add_patch(mpatches.Rectangle(
            (x0 + i * (bar_w + gap), y_base),
            bar_w, bar_h,
            facecolor=colour, alpha=0.45 + 0.50 * hh,
            edgecolor=edge, linewidth=0.45, zorder=zorder))


def x_cut(ax, cx, cy, *, size=1.0, colour=CUT_RED, lw=1.7, zorder=9):
    ax.plot([cx - size, cx + size], [cy - size, cy + size],
            color=colour, lw=lw, solid_capstyle="round", zorder=zorder)
    ax.plot([cx - size, cx + size], [cy + size, cy - size],
            color=colour, lw=lw, solid_capstyle="round", zorder=zorder)


# ---------------- scene geometry ----------------
FORWARD_Y = 38
BACKWARD_Y = 14
DIVIDER_Y = 26

SW = 8.0   # scaffold width
SH = 10.0  # scaffold height
BPAD = 0.2  # FancyBboxPatch pad — visible border sits at ±BPAD outside (x,w)
PH = 17.0      # GPE plug-in height (forward)
PH_VWL = 19.0  # VWL plug-in height (trimmed — removes the wasted top band)

GW = 26.0  # GPE width
VW = 38.0  # VWL width — narrowed; now mirrors forward row's new shorter span

# forward-row block x-origins (content: 10 → 78)
# X and p(Z*) share the same x-column (stacked vertically).
X_X    = 10.0
X_GPE  = 22.0
X_M    = 50.0
X_MU   = 60.0
X_D    = 70.0
# forward-row D center ≡ 74.0 — used to align VWL's internal (D−τ̂) input

# stacked-input geometry: X above, p(Z*) below, each half-height
SH_STACK = 8.0                         # smaller scaffold height for stack
Y_X_STACK = 38.0 + SH_STACK / 2 + 0.5  # = 42.5 (X centre)
Y_P_STACK = 38.0 - SH_STACK / 2 - 0.5  # = 33.5 (p centre)

# backward-row block x-origins (content: 10 → 78 — matches forward)
X_PILOT = 10.0
X_SIG   = 20.0
X_W     = 30.0
X_VWL   = 40.0


def caption(ax, cx, cy_bottom, text, *, color=INK_FAINT, size=6.4):
    ax.text(cx, cy_bottom, text,
            ha="center", va="center",
            fontsize=size, color=color, style="italic",
            zorder=6)


# ---------------- scaffold renderers ----------------
def _scaffold_graph(ax, cx, cy, mode):
    if mode == "X":
        # Feature matrix with meaningful shape: 6 rows × 4 cols reads as
        # "N nodes × d features", tying X to the 6-node graph spine.
        matrix_icon(ax, cx, cy, w=3.2, h=4.5, rows=6, cols=4,
                    colour=NEU_DEEP, seed=3)
    elif mode == "p":
        # Positional embedding — teal-shaded mini-graph (node2vec embeds
        # the graph structure, so a graph visualization carries that origin).
        faces = [_mix(BIAS, "#FFFFFF", 0.18 + 0.62 * s) for s in POS_SHADE]
        mininet(ax, cx, cy, scale=0.75,
                node_faces=faces, node_edge=BIAS_DK, node_size=0.34,
                edge_color=BIAS_LT, edge_alpha=0.55, edge_lw=0.6)
    elif mode == "M":
        mininet(ax, cx, cy, scale=0.85,
                node_face=NEU_DEEP, node_edge=INK_SOFT, node_size=0.44,
                edge_color=NEU_EDGE, edge_alpha=0.90, edge_lw=1.05)
    elif mode == "mu":
        mininet(ax, cx, cy, scale=0.85,
                node_face=NEU_SOFT, node_edge=NEU_DEEP, node_size=0.28,
                twin_colors=[BIAS_LT, VAR_LT],
                edge_color=NEU_SOFT, edge_alpha=0.3, edge_lw=0.55)
    elif mode == "D":
        faces = [BIAS if s > 0 else WARM for s in RESID_SIGN]
        edges_c = [BIAS_DK if s > 0 else WARM_DK for s in RESID_SIGN]
        sizes = [0.28 + 0.24 * m for m in RESID_MAG]
        mininet(ax, cx, cy, scale=0.85,
                node_faces=faces, node_edges=edges_c, node_sizes=sizes,
                edge_color=NEU_SOFT, edge_alpha=0.28, edge_lw=0.55)
    elif mode == "pilot":
        mininet(ax, cx, cy, scale=0.85,
                node_face=NEU_FILL, node_edge=INK_FAINT,
                node_dashed=True, node_size=0.34,
                edge_color=INK_FAINT, edge_alpha=0.55, edge_lw=0.6,
                edge_style=(0, (2.5, 1.8)))
    elif mode == "sigma":
        sizes = [0.22 + 0.40 * s for s in SIGMAS]
        mininet(ax, cx, cy, scale=0.85,
                node_face=VAR_LT, node_edge=VAR_DK, node_sizes=sizes,
                edge_color=VAR_LT, edge_alpha=0.50, edge_lw=0.65)
    elif mode == "w":
        weights = 1.0 / (SIGMAS ** 2)
        weights = weights / weights.max()
        sizes = [0.18 + 0.42 * w for w in weights]
        mininet(ax, cx, cy, scale=0.85,
                node_face=VAR, node_edge=VAR_DK, node_sizes=sizes,
                edge_color=VAR_LT, edge_alpha=0.45, edge_lw=0.65)


def scaffold(ax, x, y, *, mode, label_math, sub_caption, dashed=False,
             height=None, caption_pos="above"):
    """Render a scaffold block (narrow, thin border, carrier only).

    `height`: override default SH (used for the stacked X / p(Z*) pair).
    `caption_pos`: "above" (default) or "below" — useful for the lower
    scaffold in a stacked column so its caption doesn't fall in the
    gap between two stacked blocks.
    """
    h = SH if height is None else height
    block(ax, x, y - h / 2, SW, h, dashed=dashed)
    cx = x + SW / 2
    _scaffold_graph(ax, cx, y + 0.8, mode)
    ax.text(cx, y - h / 2 + 1.3, label_math,
            ha="center", va="center",
            fontsize=9.5, color=INK, zorder=6)
    cap_y = y + h / 2 + 1.3 if caption_pos == "above" else y - h / 2 - 1.3
    caption(ax, cx, cap_y, sub_caption)


# ---------------- plug-in blocks ----------------
def plugin_gpe(ax, x, y):
    """Large GPE plug-in: X + p(Z*)  →  Cross-Attention  →  fused z."""
    block(ax, x, y - PH / 2, GW, PH,
          accent=BIAS, accent_fill=BIAS_FILL, zorder=4, lw=1.8)
    cx_mid = x + GW / 2

    # Top-left input: X as 6×4 per-node feature matrix (matches outer X).
    xcx, xcy = x + 4.3, y + 3.2
    matrix_icon(ax, xcx, xcy, w=2.5, h=3.3, rows=6, cols=4,
                colour=NEU_DEEP, seed=3)
    ax.text(xcx - 2.5, xcy + 0.0, "$X$",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color=INK, zorder=7)

    # Bottom-left input: p(Z*) mini-graph (teal; matches outer scaffold).
    pcx, pcy = x + 4.3, y - 3.0
    faces = [_mix(BIAS, "#FFFFFF", 0.18 + 0.62 * s) for s in POS_SHADE]
    mininet(ax, pcx, pcy, scale=0.50,
            node_faces=faces, node_edge=BIAS_DK, node_size=0.30,
            edge_color=BIAS_LT, edge_alpha=0.55, edge_lw=0.5)
    ax.text(pcx - 2.9, pcy + 0.0, "$p$",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color=BIAS_DK, zorder=7)

    # Middle: Cross-Attn operator box
    ox, oy = x + 11.8, y
    ow, oh = 4.6, 4.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (ox - ow / 2, oy - oh / 2), ow, oh,
        boxstyle="round,pad=0.1,rounding_size=0.35",
        linewidth=1.1, edgecolor=BIAS, facecolor="white", zorder=5))
    ax.text(ox, oy + 0.55, "Cross",
            ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=BIAS_DK, zorder=6)
    ax.text(ox, oy - 0.55, "Attn",
            ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=BIAS_DK, zorder=6)

    # Right: output fused z mini-graph — solid teal nodes (no halo, which
    # previously looked like a data-bearing channel but was uniform).
    zcx, zcy = x + 20.5, y
    mininet(ax, zcx, zcy, scale=0.75,
            node_face=BIAS, node_edge=BIAS_DK, node_size=0.38,
            edge_color=BIAS, edge_alpha=0.70, edge_lw=0.85)
    ax.text(zcx, zcy - 4.3, "$z$",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=BIAS_DK, zorder=7)

    # Internal arrows: X → Attn, p → Attn, Attn → z
    arrow(ax, (xcx + 1.8, xcy - 0.4), (ox - ow / 2, oy + 0.9),
          colour=NEU_EDGE, lw=0.75, head_size=0.55)
    arrow(ax, (pcx + 1.8, pcy + 0.4), (ox - ow / 2, oy - 0.9),
          colour=BIAS_LT, lw=0.75, head_size=0.55)
    arrow(ax, (ox + ow / 2, oy), (zcx - 2.6, zcy),
          colour=BIAS, lw=1.1, head_size=0.8)

    # Title INSIDE block, top-centre (original position)
    ax.text(cx_mid, y + PH / 2 - 1.25,
            "GPE  ·  PACT plug-in",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=BIAS, zorder=11)

    # Formula — inside the block, no white background (sits directly on
    # the block fill, per user request).
    ax.text(cx_mid, y - PH / 2 + 1.8,
            r"$z_{i} \;=\; \mathrm{CrossAttn}\,(\,X_{i},\; p(Z^{*})\,)$",
            ha="center", va="center",
            fontsize=7.8, color=BIAS_DK, zorder=11)


def plugin_vwl(ax, x, y):
    """Large VWL plug-in — symmetric, vertically stacked.

    Horizontal composition (bottom row): w_i | integrated | (D − τ̂)
      — all three are mini-graphs on the shared spine, so they read as
        one family; what varies is the visual ATTRIBUTE each encodes.

    w_i    — halo-only graph (transparent light-purple nodes, halo size
             ∝ w_i). Looks like the outer-ring silhouette of the
             integrated graph, establishing visual kinship.
    D − τ̂  — sign-coloured graph (teal/orange nodes, size ∝ |residual|).
    integrated — centre: node colour ← sign(Dᵢ−τ̂ᵢ), node size ← w_i·(Dᵢ−τ̂_i)²,
                 halo ∝ same product. Makes the combination of w and
                 (D − τ̂) visible on a single visual.

    Top-centre: 𝓛 disk (prominent output) above an inline Σᵢ (no box),
    so the reader sees "∑ over the weighted residuals = 𝓛".
    """
    block(ax, x, y - PH_VWL / 2, VW, PH_VWL,
          accent=VAR, accent_fill=VAR_FILL, zorder=4, lw=1.8)
    cx_mid = x + VW / 2

    weights = 1.0 / (SIGMAS ** 2)
    weights = weights / weights.max()
    contrib = weights * (RESID_MAG ** 2)
    contrib = contrib / contrib.max()

    faces = [BIAS if s > 0 else WARM for s in RESID_SIGN]
    edges_c = [BIAS_DK if s > 0 else WARM_DK for s in RESID_SIGN]

    icx = cx_mid

    # =========================================================================
    # Vertical layout (PH_VWL = 19, block y = 4.5 → 23.5):
    #   y ≈ 22     title "VWL · PACT plug-in" (top-RIGHT of 𝓛)
    #   y ≈ 20.5   𝓛 disk at TOP-LEFT
    #   y ≈ 13     three mini-graphs row — MIDDLE of block
    #   y ≈ 8      math labels ($w_i$, …) just under graphs
    #   y ≈ 6.3    formula pill fully INSIDE (lifted off the border)
    #
    # Σᵢ rides on the diagonal arrow from integrated graph to 𝓛.
    # Operator pills (×, (·)²) sit on the two horizontal input arrows
    # so the computation chain reads inline.
    # =========================================================================

    y_row   = 13.0
    label_y = 8.0
    lx, ly  = 46.0, 20.5   # 𝓛 at top-LEFT (block now starts at x=40)
    title_y = 22.0

    wcx = 44.0                       # left graph column
    rcx = 74.0                       # right graph column (= new forward D centre)

    # w_i — halo-only, transparent inner (outer-ring silhouette)
    mininet(ax, wcx, y_row, scale=1.30, spread_nodes=True,
            node_face=(1.0, 1.0, 1.0, 0.0),
            node_edge=VAR_LT, node_size=0.15,
            halo_color=VAR,
            halo_sizes=[0.34 + 0.62 * w for w in weights],
            halo_alpha=0.32,
            edge_color=VAR_LT, edge_alpha=0.40, edge_lw=0.65)
    ax.text(wcx, label_y, "$w_{i}$",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=VAR_DK, zorder=7)

    # integrated — colour from sign(Dᵢ−τ̂ᵢ), size from wᵢ(Dᵢ−τ̂_i)²
    c_sizes = [0.22 + 0.30 * c for c in contrib]
    halo_sizes_int = [0.36 + 0.62 * c for c in contrib]
    mininet(ax, icx, y_row, scale=1.40, spread_nodes=True,
            node_faces=faces, node_edges=edges_c, node_sizes=c_sizes,
            halo_color=VAR, halo_sizes=halo_sizes_int, halo_alpha=0.32,
            edge_color=VAR_LT, edge_alpha=0.55, edge_lw=0.75)
    # (intentionally no label here — the central graph is the elementwise
    # product `w_i · (D_i − τ̂_i)²`, which the formula pill at the bottom
    # states explicitly; an extra label here would fight the pill.)

    # (D − τ̂) — sign-coloured, no halo
    sizes_D = [0.22 + 0.22 * m for m in RESID_MAG]
    mininet(ax, rcx, y_row, scale=1.30, spread_nodes=True,
            node_faces=faces, node_edges=edges_c, node_sizes=sizes_D,
            edge_color=NEU_SOFT, edge_alpha=0.40, edge_lw=0.65)
    ax.text(rcx, label_y, r"$D{-}\hat\tau$",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=INK, zorder=7)

    # 𝓛 disk — TOP-LEFT of the block
    ax.add_patch(mpatches.Circle(
        (lx, ly), 2.0,
        facecolor=VAR, edgecolor=VAR_DK, linewidth=1.7, zorder=6))
    ax.text(lx, ly, r"$\mathcal{L}$",
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=7)

    # Title — top, shifted to the RIGHT so it doesn't overlay 𝓛
    ax.text(62.0, title_y,
            "VWL  ·  PACT plug-in",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=VAR, zorder=11)

    # ============ arrows ============
    # w_i  →  integrated
    arrow(ax, (wcx + 3.5, y_row), (icx - 4.8, y_row),
          colour=VAR, lw=1.3, head_size=0.95)
    # (D − τ̂)  →  integrated
    arrow(ax, (rcx - 3.5, y_row), (icx + 4.8, y_row),
          colour=VAR, lw=1.3, head_size=0.95)

    # Operator pills sitting on the two horizontal input arrows:
    #   w_i  ·  (D − τ̂)²   →   integrated
    # Background is the block fill (VAR_FILL) so they blend into the block
    # rather than introducing a white hole.
    mx_mul = (wcx + 3.5 + icx - 4.8) / 2
    ax.text(mx_mul, y_row, r"$\times$",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=VAR_DK, zorder=9,
            bbox=dict(boxstyle="circle,pad=0.18", fc=VAR_FILL,
                      ec=VAR_LT, lw=0.8, alpha=0.99))
    mx_sq = (rcx - 3.5 + icx + 4.8) / 2
    ax.text(mx_sq, y_row, r"$(\cdot)^{2}$",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color=VAR_DK, zorder=9,
            bbox=dict(boxstyle="round,pad=0.22", fc=VAR_FILL,
                      ec=VAR_LT, lw=0.8, alpha=0.99))

    # integrated  →  𝓛  (single diagonal arrow; Σᵢ label rides on it)
    int_top_y = y_row + 3.78   # top of integrated graph (scale 1.40)
    arr_src = (icx - 0.4, int_top_y)
    arr_dst = (lx + 1.5, ly - 1.5)
    arrow(ax, arr_src, arr_dst,
          colour=VAR, lw=1.4, head_size=1.0)
    # Σᵢ label — styled like the × / (·)² operator pills (small, VAR_FILL
    # background), and floats ABOVE the diagonal arrow (offset along the
    # arrow's perpendicular so the line still reads as a continuous path
    # from the integrated graph up to 𝓛).
    sig_mx = (arr_src[0] + arr_dst[0]) / 2
    sig_my = (arr_src[1] + arr_dst[1]) / 2
    dx = arr_dst[0] - arr_src[0]
    dy = arr_dst[1] - arr_src[1]
    L = (dx * dx + dy * dy) ** 0.5
    # perpendicular pointing UP (positive y)
    perp = (-dy / L, dx / L)
    if perp[1] < 0:
        perp = (-perp[0], -perp[1])
    off = 1.1
    sig_pos = (sig_mx + perp[0] * off, sig_my + perp[1] * off)
    ax.text(sig_pos[0], sig_pos[1], r"$\Sigma_{i}$",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color=VAR_DK, zorder=9)

    # Formula — inside the block, no white background (sits directly on
    # the block fill, per user request).
    ax.text(cx_mid, y - PH_VWL / 2 + 1.8,
            r"$\mathcal{L} \;=\; \sum_{i} w_{i}\,(D_{i}-\hat\tau_{i})^{2},"
            r"\quad w_{i}=1/\hat\sigma^{2}_{t}(X_{i})$",
            ha="center", va="center",
            fontsize=7.8, color=VAR_DK, zorder=11)


# ---------------- scene ----------------
def draw_sidebar(ax):
    ax.text(3.4, FORWARD_Y, "FORWARD",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=BIAS,
            rotation=90, rotation_mode="anchor")
    ax.text(3.4, BACKWARD_Y, "BACKWARD",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=VAR,
            rotation=90, rotation_mode="anchor")
    ax.plot([6.6, 6.6], [3, 53], color=DIVIDER, lw=0.5, zorder=1)


def draw_divider(ax):
    ax.plot([8.5, 86.5], [DIVIDER_Y, DIVIDER_Y],
            color=DIVIDER, lw=0.55, linestyle=(0, (5, 3)), zorder=1)


def draw_legend(ax):
    """Top-of-figure key — at first glance, reviewer sees which blocks are
    PACT (ours) vs. the unchanged graph ITE backbone.  X-learner is not
    mentioned: it is off-the-shelf machinery, not our contribution."""
    y = 51.0
    # Coloured swatches = PACT (ours)
    ax.add_patch(mpatches.FancyBboxPatch(
        (10.5, y - 1.3), 3.0, 2.6,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        linewidth=1.6, edgecolor=BIAS, facecolor=BIAS_FILL, zorder=10))
    ax.add_patch(mpatches.FancyBboxPatch(
        (14.5, y - 1.3), 3.0, 2.6,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        linewidth=1.6, edgecolor=VAR, facecolor=VAR_FILL, zorder=10))
    ax.text(19.0, y, "PACT (ours)",
            ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=INK, zorder=11)
    # Gray swatch = unchanged backbone
    ax.add_patch(mpatches.FancyBboxPatch(
        (34.0, y - 1.3), 3.0, 2.6,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        linewidth=0.7, edgecolor=NEU_EDGE, facecolor=NEU_FILL, zorder=10))
    ax.text(38.0, y, "graph ITE backbone (unchanged)",
            ha="left", va="center",
            fontsize=8.5, color=INK_SOFT, zorder=11)


def draw_forward(ax):
    y = FORWARD_Y

    # Stacked input column: X above, p(Z*) below, both at x = X_X.
    # Both are PARALLEL inputs to GPE (p(Z*) is computed offline from the
    # graph 𝒢 via node2vec, NOT from X).
    scaffold(ax, X_X, Y_X_STACK, mode="X",
             label_math="$X$", sub_caption="node features",
             height=SH_STACK, caption_pos="above")
    scaffold(ax, X_X, Y_P_STACK, mode="p",
             label_math=r"$p(Z^{*})$",
             sub_caption=r"via node2vec$(\mathcal{G})$",
             dashed=True, height=SH_STACK, caption_pos="below")

    plugin_gpe(ax, X_GPE, y)

    scaffold(ax, X_M, y, mode="M",
             label_math="$M$", sub_caption="graph ITE backbone")
    scaffold(ax, X_MU, y, mode="mu",
             label_math=r"$\hat\mu_{0},\hat\mu_{1}$",
             sub_caption="outcome heads")
    scaffold(ax, X_D, y, mode="D",
             label_math="$D$", sub_caption="pseudo-outcome")

    # teal forward arrows — endpoints sit on the true visible border of
    # each block (FancyBboxPatch extends ±BPAD outside (x, w)).
    #
    # X → GPE and p(Z*) → GPE are two short parallel arrows from the
    # stacked input column into GPE's left edge at the corresponding
    # internal heights (inner X above, inner p below).
    arrow(ax, (X_X + SW + BPAD, Y_X_STACK),
          (X_GPE - BPAD, Y_X_STACK - 0.5),
          colour=BIAS, lw=1.3, head_size=0.95)
    arrow(ax, (X_X + SW + BPAD, Y_P_STACK),
          (X_GPE - BPAD, Y_P_STACK + 0.5),
          colour=BIAS, lw=1.3, head_size=0.95)
    # GPE → M
    arrow(ax, (X_GPE + GW + BPAD, y), (X_M - BPAD, y),
          colour=BIAS, lw=1.3, head_size=0.95)
    # M → μ̂
    arrow(ax, (X_M + SW + BPAD, y), (X_MU - BPAD, y),
          colour=BIAS, lw=1.3, head_size=0.95)
    # μ̂ → D
    arrow(ax, (X_MU + SW + BPAD, y), (X_D - BPAD, y),
          colour=BIAS, lw=1.3, head_size=0.95)

    return {
        "M_box": (X_M, y - SH / 2, SW, SH),
        "D_box": (X_D, y - SH / 2, SW, SH),
    }


def draw_backward(ax, hooks):
    y = BACKWARD_Y

    scaffold(ax, X_PILOT, y, mode="pilot",
             label_math=r"pilot $\hat\tau$",
             sub_caption="detached target", dashed=True)
    scaffold(ax, X_SIG, y, mode="sigma",
             label_math=r"$\log\hat\sigma^{2}_{t}$",
             sub_caption="noise head")
    scaffold(ax, X_W, y, mode="w",
             label_math=r"$w_{i}$",
             sub_caption="inv-variance weight")

    plugin_vwl(ax, X_VWL, y)

    # purple flow arrows, pilot → σ̂² (stop-grad) → w_i → VWL.
    # Endpoints land on the true border of each block (FancyBboxPatch pad).
    arrow(ax, (X_PILOT + SW + BPAD, y), (X_SIG - BPAD, y),
          colour=VAR, lw=1.2, dashed=True, head_size=0.95)
    cut_mid = (X_PILOT + SW + X_SIG) / 2
    # Smaller red ✗ = stop-gradient (detach), with a small italic caption
    # so the reviewer sees what it means without hunting the text.
    x_cut(ax, cut_mid, y, size=0.6, colour=CUT_RED, lw=1.3, zorder=12)
    # "detach" label sits in the empty band ABOVE the arrow (between the
    # arrow line and the block captions) so it doesn't crowd the ✗ or the
    # horizontal arrow shaft.
    ax.text(cut_mid, y + 2.1, "detach",
            ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=CUT_RED,
            style="italic", zorder=12)

    arrow(ax, (X_SIG + SW + BPAD, y), (X_W - BPAD, y),
          colour=VAR, lw=1.2, head_size=0.95)
    arrow(ax, (X_W + SW + BPAD, y), (X_VWL - BPAD, y),
          colour=VAR, lw=1.2, head_size=0.95)

    # D (forward) → (D − τ̂) input inside VWL: straight vertical drop.
    # X_D + SW/2 = 82.0 is exactly the x of the (D − τ̂) mini-graph inside
    # plugin_vwl(), so this arrow is perfectly vertical and lands on the
    # pseudo-outcome input directly.
    xD, yD, wD, _ = hooks["D_box"]
    dcx = xD + wD / 2           # forward D centre (= 82.0)
    # D-τ̂ mini-graph is at y_row=13.0 with scale 1.30; top node centre sits
    # at y ≈ 16.5 with radius ~0.32, so its top edge is ~16.85.  Land the
    # arrow well above that so the head doesn't pierce the orange node.
    d_tau_top_y = 17.3
    ax.annotate(
        "",
        xy=(dcx, d_tau_top_y),
        xytext=(dcx, yD - 0.3),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.62,head_width=0.38",
            color=VAR, lw=1.35, shrinkA=0, shrinkB=0,
        ),
        zorder=6,
    )

    # Gradient arc: 𝓛 → M.  𝓛 lives at top-LEFT of VWL: (46, 20.5).
    # Forward M centre is at x = 54 (with new layout) — arc goes up-right.
    xM, yM, wM, _ = hooks["M_box"]
    lcx = 46.0
    lcy = 20.5                   # matches loss position inside plugin_vwl
    grad_src = (lcx + 0.5, lcy + 2.0)    # leave from top edge of 𝓛
    grad_dst = (xM + wM / 2, yM - 0.3)
    ax.annotate(
        "", xy=grad_dst, xytext=grad_src,
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.75,head_width=0.46",
            color=VAR, lw=1.45, linestyle=(0, (4, 2)),
            connectionstyle="arc3,rad=0.35",
            shrinkA=2, shrinkB=2,
        ),
        zorder=7,
    )
    # Place the ∇_M𝓛 label along the arc's visible apex — for rad=+0.35
    # the arc bulges down-right below the straight line, so the apex sits
    # roughly at (straight_mid + rad·|d|·perp_right).
    smx = (grad_src[0] + grad_dst[0]) / 2
    smy = (grad_src[1] + grad_dst[1]) / 2
    ddx = grad_dst[0] - grad_src[0]
    ddy = grad_dst[1] - grad_src[1]
    L_ = (ddx * ddx + ddy * ddy) ** 0.5
    perp_r = (ddy / L_, -ddx / L_)   # perpendicular to right of forward dir
    off = 0.35 * L_ * 0.55           # ~55 % of the full rad offset → sits on arc
    lbl_pos = (smx + perp_r[0] * off, smy + perp_r[1] * off)
    ax.text(lbl_pos[0], lbl_pos[1],
            r"$\nabla_{M}\mathcal{L}$",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=VAR, zorder=8,
            bbox=dict(boxstyle="round,pad=0.18", fc=BG,
                      ec=VAR_LT, lw=0.55, alpha=0.99))


def main():
    fig = plt.figure(figsize=(8.4, 5.1), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.set_facecolor(BG)

    draw_sidebar(ax)
    draw_divider(ax)
    draw_legend(ax)
    hooks = draw_forward(ax)
    draw_backward(ax, hooks)

    fig.savefig(OUT_PDF, format="pdf", facecolor=BG)
    fig.savefig(OUT_PNG, format="png", dpi=220, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
