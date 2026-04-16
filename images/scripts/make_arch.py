"""Render Figure 2 — PACT Universal Causal Processor (academic schematic).

Paper-native isometric blueprint:
  * pure white background; flat-shaded isometric modules
  * thin directed arrows for the two causal channels
  * minimalist chip rack (three representative chips) hovering over
    the M-Socket; no "gaming-hardware" glow or gradients
  * PEHE^2 = bias^2 + variance equation integrated as the pedestal
    legend; colour-coding (teal/purple) is held strictly consistent

Run from pact-long/:
    python images/scripts/make_arch.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PDF = OUT_DIR / "arch.pdf"
OUT_PNG = OUT_DIR / "arch_preview.png"

# ---------------- palette ----------------
BG = "#FFFFFF"

NEU_TOP = "#F3F3F3"
NEU_RIGHT = "#DADADA"
NEU_FRONT = "#BEBEBE"
NEU_EDGE = "#5E5E5E"

SOC_TOP = "#DDDDDD"
SOC_RIGHT = "#BABABA"
SOC_FRONT = "#9C9C9C"
SOC_EDGE = "#3A3A3A"

CHIP_TOP = "#EDEDED"
CHIP_RIGHT = "#D0D0D0"
CHIP_FRONT = "#AEAEAE"
CHIP_EDGE = "#3C3C3C"

BIAS = "#2A8A8A"
BIAS_RIGHT = "#206565"
BIAS_FRONT = "#143E3E"

VAR = "#8A4FAD"
VAR_RIGHT = "#693B82"
VAR_FRONT = "#412454"

Z_TOP = "#7E7E7E"
Z_RIGHT = "#5F5F5F"
Z_FRONT = "#3E3E3E"
Z_EDGE = "#232323"

INK = "#111316"
INK_SOFT = "#4A4E56"
INK_FAINT = "#7F838B"

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

# ---------------- isometric projection ----------------
COS30 = float(np.cos(np.radians(30)))
SIN30 = 0.5


def iso(x, y, z=0.0):
    return ((x - y) * COS30, -(x + y) * SIN30 + z)


def iso_pts(pts3d):
    return [iso(*p) for p in pts3d]


# ---------------- primitives ----------------
def iso_plane(ax, x, y, w, d, z=0.0, *, fc, ec=None, lw=0.6,
              zorder=1, ls="-"):
    pts = iso_pts([
        (x, y, z), (x + w, y, z),
        (x + w, y + d, z), (x, y + d, z),
    ])
    ec = ec if ec is not None else fc
    ax.add_patch(mpatches.Polygon(
        pts, closed=True, facecolor=fc, edgecolor=ec, lw=lw,
        linestyle=ls, zorder=zorder, joinstyle="round"))
    return pts


def iso_box(ax, x, y, w, d, h, *, top, right, front,
            ec=INK, lw=0.7, zorder=5):
    front_pts = iso_pts([
        (x, y + d, 0), (x + w, y + d, 0),
        (x + w, y + d, h), (x, y + d, h),
    ])
    right_pts = iso_pts([
        (x + w, y, 0), (x + w, y + d, 0),
        (x + w, y + d, h), (x + w, y, h),
    ])
    top_pts = iso_pts([
        (x, y, h), (x + w, y, h),
        (x + w, y + d, h), (x, y + d, h),
    ])
    for pts, fc, z_add in [(front_pts, front, 0),
                           (right_pts, right, 0.05),
                           (top_pts, top, 0.10)]:
        ax.add_patch(mpatches.Polygon(
            pts, closed=True, facecolor=fc, edgecolor=ec,
            lw=lw, zorder=zorder + z_add, joinstyle="round"))


def thin_arrow(ax, p_from_3d, p_to_3d, *, colour=INK, lw=0.9,
               ls="-", zorder=7, arrow_style="-|>,head_length=0.5,head_width=0.3"):
    s = iso(*p_from_3d)
    e = iso(*p_to_3d)
    ax.annotate(
        "", xy=e, xytext=s,
        arrowprops=dict(arrowstyle=arrow_style,
                        color=colour, lw=lw,
                        linestyle=ls,
                        shrinkA=0, shrinkB=0),
        zorder=zorder,
    )


# ==========================================================
# SCENE DIMENSIONS
# ==========================================================
# Substrate: 48 wide x 16 deep
SW, SD = 48.0, 16.0

# Module footprints (x, y, w, d, h) in 3D
GPE = (3.0,  3.5, 10.0, 9.0, 5.0)
MSO = (17.0, 3.5, 14.0, 9.0, 1.4)
VWL = (35.0, 3.5, 10.0, 9.0, 5.0)


def draw_substrate(ax):
    iso_plane(ax, 0, 0, SW, SD, 0,
              fc=NEU_TOP, ec=NEU_EDGE, lw=0.8, zorder=1)
    # thin slab under the plane
    ax.add_patch(mpatches.Polygon(
        iso_pts([(0, SD, -0.35), (SW, SD, -0.35),
                 (SW, SD, 0), (0, SD, 0)]),
        closed=True, facecolor=NEU_FRONT, edgecolor=NEU_EDGE,
        lw=0.5, zorder=0.95))
    ax.add_patch(mpatches.Polygon(
        iso_pts([(SW, 0, -0.35), (SW, SD, -0.35),
                 (SW, SD, 0), (SW, 0, 0)]),
        closed=True, facecolor=NEU_RIGHT, edgecolor=NEU_EDGE,
        lw=0.5, zorder=0.97))


def draw_zstar(ax):
    """Grounded causal anchor at the front-centre of the substrate."""
    zx, zy = SW / 2, 1.1
    iso_box(ax, zx - 0.8, zy - 0.8, 1.6, 1.6, 0.55,
            top=Z_TOP, right=Z_RIGHT, front=Z_FRONT,
            ec=Z_EDGE, lw=0.7, zorder=5)
    p_anchor = iso(zx, zy, 0.6)
    ax.text(p_anchor[0], p_anchor[1], r"$Z^{*}$",
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white", zorder=7)

    # Small side label to the immediate RIGHT of the anchor, on the same
    # vertical plane (in screen space) but well clear of the socket /
    # chip rack which sit BEHIND and ABOVE it.
    p_side = (p_anchor[0] + 3.8, p_anchor[1] - 2.8)
    ax.annotate(
        "", xy=p_anchor, xytext=p_side,
        arrowprops=dict(arrowstyle="-", color=INK_FAINT, lw=0.5,
                        shrinkA=2, shrinkB=2),
        zorder=10)
    ax.text(p_side[0], p_side[1],
            r"$Z^{*}$  latent position",
            ha="left", va="center",
            fontsize=6.9, fontweight="bold", color=INK, zorder=11,
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=INK_FAINT, lw=0.5))
    return (zx, zy, 0.6)


def draw_ports(ax):
    gx, gy, gw, gd, gh = GPE
    mx, my, mw, md, mph = MSO
    vx, vy, vw, vd, vh = VWL

    # GPE (teal)
    iso_box(ax, gx, gy, gw, gd, gh,
            top=BIAS, right=BIAS_RIGHT, front=BIAS_FRONT,
            ec=INK, lw=0.7, zorder=6)
    g_c = iso(gx + gw / 2, gy + gd / 2, gh)
    ax.text(g_c[0], g_c[1] + 0.6, "GPE",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=12)
    ax.text(g_c[0], g_c[1] - 0.9,
            r"$X \oplus p(Z^{*})$",
            ha="center", va="center",
            fontsize=7, color="white", zorder=12)

    # M-Socket pedestal
    iso_box(ax, mx, my, mw, md, mph,
            top=SOC_TOP, right=SOC_RIGHT, front=SOC_FRONT,
            ec=SOC_EDGE, lw=0.7, zorder=6)
    # dashed socket inlay
    sx0, sy0 = mx + 1.3, my + 1.3
    sw, sd = mw - 2.6, md - 2.6
    sock_pts = iso_pts([
        (sx0, sy0, mph + 0.02),
        (sx0 + sw, sy0, mph + 0.02),
        (sx0 + sw, sy0 + sd, mph + 0.02),
        (sx0, sy0 + sd, mph + 0.02),
    ])
    ax.add_patch(mpatches.Polygon(
        sock_pts, closed=True, facecolor="white",
        edgecolor=SOC_EDGE, lw=0.9,
        linestyle=(0, (5, 2.5)), zorder=6.8))
    # (The "any graph ITE backbone" inscription is now carried by the
    #  chip-rack caption below the substrate.)

    # VWL (purple)
    iso_box(ax, vx, vy, vw, vd, vh,
            top=VAR, right=VAR_RIGHT, front=VAR_FRONT,
            ec=INK, lw=0.7, zorder=6)
    v_c = iso(vx + vw / 2, vy + vd / 2, vh)
    ax.text(v_c[0], v_c[1] + 0.6, "VWL",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=12)
    ax.text(v_c[0], v_c[1] - 0.9,
            r"$\sum_i w_i (D_i - \hat\tau)^2$",
            ha="center", va="center",
            fontsize=6.6, color="white", zorder=12)


def draw_chip_rack(ax):
    """Abstract chip rack — three anonymous flat modules hovering over the
    M-Socket. Chip labels are replaced by a single compat caption under
    the substrate (see draw_backbone_caption). This keeps the rack
    visually clean and the labels unambiguously out of the scene."""
    mx, my, mw, md, mph = MSO
    chip_w = mw - 4.0
    chip_d = md - 3.0
    chip_h = 0.4
    gap = 1.5
    base_z = mph + 2.0

    for idx_from_bottom in range(3):
        z = base_z + idx_from_bottom * (chip_h + gap)
        ox = -0.35 if idx_from_bottom == 0 else 0.0
        oy = 0.25 if idx_from_bottom == 0 else 0.0
        cx = mx + 2.0 + ox
        cy = my + 1.5 + oy
        iso_box(ax, cx, cy, chip_w, chip_d, chip_h,
                top=CHIP_TOP, right=CHIP_RIGHT, front=CHIP_FRONT,
                ec=CHIP_EDGE, lw=0.5, zorder=9 + idx_from_bottom * 0.4)


def draw_causal_arrows(ax, z_anchor):
    """Thin directed arrows Z* -> GPE, Z* -> VWL with causal-path labels."""
    zx, zy, zh = z_anchor
    # Source: just above Z* core
    src = (zx, zy, zh + 0.2)

    # Destinations: front-face centres of GPE and VWL (so the arrows
    # terminate cleanly at the port surfaces).
    gx, gy, gw, gd, gh = GPE
    vx, vy, vw, vd, vh = VWL
    gpe_dst = (gx + gw / 2, gy + gd, gh * 0.55)
    vwl_dst = (vx + vw / 2, vy + vd, vh * 0.55)

    # GPE arrow (teal)
    thin_arrow(ax, src, gpe_dst, colour=BIAS, lw=0.95, zorder=7.5)
    # VWL arrow (purple)
    thin_arrow(ax, src, vwl_dst, colour=VAR, lw=0.95, zorder=7.5)

    # Labels — placed along each arrow midpoint in SCREEN coords.
    s_screen = iso(*src)
    g_screen = iso(*gpe_dst)
    v_screen = iso(*vwl_dst)

    # For the teal arrow: midpoint, then offset slightly to the
    # north-west so the label sits above the line.
    tm = (0.5 * (s_screen[0] + g_screen[0]) - 0.3,
          0.5 * (s_screen[1] + g_screen[1]) + 1.1)
    ax.text(tm[0], tm[1],
            r"$Z^{*}\!\to T,\,Y(t)$",
            ha="center", va="bottom",
            fontsize=6.4, color=BIAS, fontweight="bold", zorder=12)

    pm = (0.5 * (s_screen[0] + v_screen[0]) + 0.3,
          0.5 * (s_screen[1] + v_screen[1]) + 1.1)
    ax.text(pm[0], pm[1],
            r"$Z^{*}\!\to \sigma^{2}(Y)$",
            ha="center", va="bottom",
            fontsize=6.4, color=VAR, fontweight="bold", zorder=12)


def draw_title(ax):
    title_x = (iso(0, SD, 0)[0] + iso(SW, 0, 0)[0]) / 2
    ax.text(title_x, 12.5,
            "PACT: Universal Causal Processor",
            ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=INK, zorder=15)
    ax.text(title_x, 10.5,
            "a wrapper architecture for any graph ITE backbone",
            ha="center", va="center",
            fontsize=7.8, style="italic", color=INK_SOFT, zorder=15)


def draw_backbone_caption(ax):
    """One-line caption naming the seven compatible backbones, placed
    just below the substrate front-edge."""
    base_x = (iso(0, SD, 0)[0] + iso(SW, 0, 0)[0]) / 2
    base_y = iso(SW, SD, 0)[1] - 1.6
    ax.text(base_x, base_y,
            "chip rack  { M }:  BNN  ·  TARNet  ·  X-learner  ·  "
            "NetDeconf  ·  GIAL  ·  GNUM  ·  GDC",
            ha="center", va="center",
            fontsize=6.6, style="italic", color=INK_SOFT, zorder=12)


def draw_equation(ax):
    base_x = (iso(0, SD, 0)[0] + iso(SW, 0, 0)[0]) / 2
    base_y = iso(SW, SD, 0)[1] - 6.5

    # simple horizontal pedestal band (no heavy box)
    band_h = 4.6
    band = mpatches.FancyBboxPatch(
        (base_x - 23, base_y - band_h / 2), 46, band_h,
        boxstyle="round,pad=0.3,rounding_size=1.1",
        linewidth=0.7, edgecolor=NEU_EDGE, facecolor="#FAFAFA",
        zorder=11)
    ax.add_patch(band)

    # Piece-wise coloured equation
    ax.text(base_x - 13.0, base_y, r"$\mathrm{PEHE}^2 \;=\;$",
            ha="right", va="center",
            fontsize=11.5, color=INK, zorder=13)
    ax.text(base_x - 7.0, base_y, r"$\mathrm{bias}^{2}$",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=BIAS, zorder=13)
    ax.text(base_x - 7.0, base_y - 1.8, "(GPE)",
            ha="center", va="center",
            fontsize=6.8, style="italic", color=BIAS, zorder=13)
    ax.text(base_x - 1.0, base_y, "+",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=INK, zorder=13)
    ax.text(base_x + 6.0, base_y, "variance",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=VAR, zorder=13)
    ax.text(base_x + 6.0, base_y - 1.8, "(VWL)",
            ha="center", va="center",
            fontsize=6.8, style="italic", color=VAR, zorder=13)

    # short underline ticks (the "underbrace")
    ax.plot([base_x - 10.4, base_x - 3.6],
            [base_y - 1.0, base_y - 1.0],
            color=BIAS, lw=0.8, zorder=12)
    ax.plot([base_x + 2.2, base_x + 9.8],
            [base_y - 1.0, base_y - 1.0],
            color=VAR, lw=0.8, zorder=12)

    ax.text(base_x + 18.5, base_y,
            "Prop. 1",
            ha="center", va="center",
            fontsize=7.8, style="italic", color=INK_SOFT, zorder=13)


def main():
    fig = plt.figure(figsize=(7.4, 4.3), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.axis("off")

    xs_left = iso(0, SD, 0)[0] - 5
    xs_right = iso(SW, 0, 0)[0] + 5
    ys_bottom = iso(SW, SD, 0)[1] - 11
    ys_top = 15
    ax.set_xlim(xs_left, xs_right)
    ax.set_ylim(ys_bottom, ys_top)

    draw_title(ax)
    draw_substrate(ax)
    z_anchor = draw_zstar(ax)
    draw_ports(ax)
    draw_chip_rack(ax)
    draw_causal_arrows(ax, z_anchor)
    draw_backbone_caption(ax)
    draw_equation(ax)

    fig.savefig(OUT_PDF, format="pdf", facecolor=BG)
    fig.savefig(OUT_PNG, format="png", dpi=220, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
