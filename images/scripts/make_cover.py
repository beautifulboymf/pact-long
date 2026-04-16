"""Render Figure 1 (cover) — Dual-Root Dissection, single-column v9.

Changes over v8:
  * X cuts moved ONTO the causal arrows (Z* -> bias^2 and
    Z* -> variance) so they read as "intervention severs pathway"
    instead of a decorative mark on the plug-in leader.
  * Plug-in labels (GPE / VWL) are tagged directly at each cut, so the
    visual reads: "here is where this plug-in cuts."
  * Z* pill now contains a miniature graph (four nodes + edges) so it
    is not just a text label.
  * GPE card contains a micro-diagram of the input fusion
    X (+) p(Z*) -> cross-attn.
  * VWL card shows the weighted-loss formula with w_i highlighted.

Run from pact-long/:
    python images/scripts/make_cover.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PDF = OUT_DIR / "Uplift_cover.pdf"
OUT_PNG = OUT_DIR / "Uplift_cover_preview.png"

BIAS_ACCENT = "#2A8A8A"
VAR_ACCENT = "#8A4FAD"
CUT_RED = "#C94040"
LATENT_GREY = "#6A6A6A"
CLOUD_FILL = "#EFECE4"
MINI_EDGE = "#B5B0A6"
MINI_NODE = "#D8D3C6"
TEXT_DARK = "#1F1F1F"
TEXT_GREY = "#5A5A5A"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def _x_cut(ax, cx, cy, size=2.6, colour=CUT_RED, lw=2.3, zorder=9):
    d = size
    ax.plot([cx - d, cx + d], [cy - d, cy + d], color=colour,
            linewidth=lw, solid_capstyle="round", zorder=zorder)
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color=colour,
            linewidth=lw, solid_capstyle="round", zorder=zorder)


# ===================================================================
# ZONE A — Z* pill with a miniature graph inside
# ===================================================================
def draw_zone_a(ax):
    pill = mpatches.FancyBboxPatch(
        (18.0, 124.0), 64.0, 22.0,
        boxstyle="round,pad=0.25,rounding_size=2.4",
        linewidth=1.1, edgecolor=LATENT_GREY, facecolor=CLOUD_FILL,
        linestyle=(0, (4, 2)), zorder=3,
    )
    ax.add_patch(pill)

    # Z* heading on the left
    ax.text(27.0, 141.5, r"$Z^*$",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=LATENT_GREY, zorder=6)
    ax.text(27.0, 134.0, "latent",
            ha="center", va="center",
            fontsize=7.8, style="italic", color=TEXT_GREY, zorder=6)
    ax.text(27.0, 130.0, "position",
            ha="center", va="center",
            fontsize=7.8, style="italic", color=TEXT_GREY, zorder=6)

    # Miniature graph on the right side of the pill
    mini_nodes = {
        0: (43.0, 142.0),
        1: (52.0, 140.5),
        2: (49.0, 132.0),
        3: (60.0, 137.5),
        4: (68.0, 142.5),
        5: (65.0, 130.5),
        6: (74.0, 134.5),
    }
    mini_edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 6), (5, 6), (2, 5)]
    for u, v in mini_edges:
        ax.plot(
            [mini_nodes[u][0], mini_nodes[v][0]],
            [mini_nodes[u][1], mini_nodes[v][1]],
            color=MINI_EDGE, linewidth=0.6, alpha=0.85, zorder=4,
            solid_capstyle="round",
        )
    for i, (x, y) in mini_nodes.items():
        ax.scatter(x, y, s=60, c=MINI_NODE,
                   edgecolors=LATENT_GREY, linewidths=0.55,
                   zorder=5)


# ===================================================================
# ZONE B — causal arrows down, with X cuts labeled GPE / VWL,
#          then bias^2 / variance boxes, then decomposition caption
# ===================================================================
def draw_zone_b(ax):
    # Two causal arrows from Z* (pill bottom y=124) down to the error boxes
    # (top y=105). The X cut sits at mid-arrow (around y=115).
    ax.annotate("", xy=(20.0, 106.0), xytext=(38.0, 123.5),
                arrowprops=dict(arrowstyle="-|>", color=BIAS_ACCENT, lw=1.6,
                                shrinkA=0, shrinkB=2,
                                connectionstyle="arc3,rad=-0.1"),
                zorder=5)
    ax.annotate("", xy=(80.0, 106.0), xytext=(62.0, 123.5),
                arrowprops=dict(arrowstyle="-|>", color=VAR_ACCENT, lw=1.6,
                                shrinkA=0, shrinkB=2,
                                connectionstyle="arc3,rad=0.1"),
                zorder=5)

    # Small causal-pathway labels — placed just BELOW the Z* pill so
    # they live in the strip between the pill and the X cuts, clear of
    # the GPE/VWL labels that sit at y=114.
    ax.text(16.0, 121.0, r"$Z^{*}\!\to T,\; Z^{*}\!\to Y(t)$",
            fontsize=7.6, color=BIAS_ACCENT, fontweight="bold",
            ha="center", va="center", zorder=6)
    ax.text(86.0, 121.0, r"$Z^{*}\!\to \sigma^2(Y)$",
            fontsize=7.6, color=VAR_ACCENT, fontweight="bold",
            ha="center", va="center", zorder=6)

    # X cuts ON the causal arrows, with plug-in tag to the OUTER side
    # of each arrow (so the tags sit in clear whitespace, not on the
    # post-cut arrow segment).
    _x_cut(ax, 26.0, 114.0, size=2.6)
    ax.text(19.0, 114.0, "GPE",
            ha="right", va="center",
            fontsize=8.4, fontweight="bold", color=BIAS_ACCENT, zorder=9)

    _x_cut(ax, 74.0, 114.0, size=2.6)
    ax.text(81.0, 114.0, "VWL",
            ha="left", va="center",
            fontsize=8.4, fontweight="bold", color=VAR_ACCENT, zorder=9)

    # bias^2 box (teal)
    bias_box = mpatches.FancyBboxPatch(
        (4.0, 88.0), 38.0, 17.0,
        boxstyle="round,pad=0.25,rounding_size=2.2",
        linewidth=1.1, edgecolor=BIAS_ACCENT,
        facecolor=BIAS_ACCENT, alpha=0.14, zorder=3,
    )
    ax.add_patch(bias_box)
    ax.text(23.0, 99.5, r"$\mathrm{bias}^2$",
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=BIAS_ACCENT, zorder=5)
    ax.text(23.0, 92.5, "(confounding)",
            ha="center", va="center",
            fontsize=7.4, style="italic", color=BIAS_ACCENT, zorder=5)

    # variance box (purple)
    var_box = mpatches.FancyBboxPatch(
        (58.0, 88.0), 38.0, 17.0,
        boxstyle="round,pad=0.25,rounding_size=2.2",
        linewidth=1.1, edgecolor=VAR_ACCENT,
        facecolor=VAR_ACCENT, alpha=0.13, zorder=3,
    )
    ax.add_patch(var_box)
    ax.text(77.0, 99.5, "variance",
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=VAR_ACCENT, zorder=5)
    ax.text(77.0, 92.5, "(heteroscedasticity)",
            ha="center", va="center",
            fontsize=7.4, style="italic", color=VAR_ACCENT, zorder=5)

    # + sign
    ax.text(50.0, 96.5, "+",
            ha="center", va="center",
            fontsize=18, color=TEXT_DARK, fontweight="bold", zorder=5)


# ===================================================================
# ZONE C — two plug-in cards with micro-visualizations
# (All y-coordinates shifted up by 10 vs. v10 to close the gap between
#  the bias^2/variance boxes and the cards, per user feedback on
#  vertical whitespace.)
# ===================================================================
def draw_zone_c(ax):
    # Cards shifted up +6 so their top sits close (4 units) below the
    # bias^2/variance boxes.
    CARD_Y = 30
    CARD_H = 54

    # =============== GPE card ===============
    gpe = mpatches.FancyBboxPatch(
        (3.0, CARD_Y), 45.0, CARD_H,
        boxstyle="round,pad=0.3,rounding_size=2.6",
        linewidth=1.3, edgecolor=BIAS_ACCENT,
        facecolor=BIAS_ACCENT, alpha=0.08, zorder=3,
    )
    ax.add_patch(gpe)
    ax.text(25.5, 79.0, "GPE",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=BIAS_ACCENT, zorder=5)

    # Row 1: X  (+)  p(Z*)
    _mini_box(ax, 7.0, 66.0, 11.0, 7.5, "$X$", BIAS_ACCENT, text_size=10)
    ax.text(21.5, 69.75, "+", ha="center", va="center",
            fontsize=13, color=BIAS_ACCENT, fontweight="bold", zorder=5)
    _mini_box(ax, 25.0, 66.0, 19.0, 7.5, r"$p(Z^*)$",
              BIAS_ACCENT, dashed=True, text_size=10)
    ax.text(34.5, 62.2, "node2vec",
            ha="center", va="center",
            fontsize=6.2, style="italic", color=BIAS_ACCENT, zorder=5)

    # Compact down-arrow labelled in-line
    ax.annotate("", xy=(25.5, 57.0), xytext=(25.5, 61.5),
                arrowprops=dict(arrowstyle="-|>", color=BIAS_ACCENT, lw=1.2,
                                shrinkA=0, shrinkB=0),
                zorder=5)
    ax.text(29.0, 59.2, "cross-attn",
            ha="left", va="center",
            fontsize=6.4, style="italic", color=BIAS_ACCENT, zorder=5)

    # Row 2: fused input -> M
    _mini_box(ax, 7.0, 49.0, 22.0, 7.5, "fused $X$", BIAS_ACCENT, text_size=9)
    ax.annotate("", xy=(33.5, 52.75), xytext=(29.2, 52.75),
                arrowprops=dict(arrowstyle="-|>", color=BIAS_ACCENT, lw=1.2,
                                shrinkA=0, shrinkB=0),
                zorder=5)
    _mini_box(ax, 34.5, 49.0, 9.0, 7.5, "$M$", BIAS_ACCENT, text_size=10)

    # Takeaway lines
    ax.text(25.5, 42.5, r"$\Rightarrow\;$ reduces $\mathrm{bias}^2$",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=BIAS_ACCENT, zorder=5)
    ax.text(25.5, 37.5,
            r"cuts $Z^*\!\to T,\, Y(t)$",
            ha="center", va="center",
            fontsize=7.0, style="italic", color=BIAS_ACCENT, zorder=5)

    # =============== VWL card ===============
    vwl = mpatches.FancyBboxPatch(
        (52.0, CARD_Y), 45.0, CARD_H,
        boxstyle="round,pad=0.3,rounding_size=2.6",
        linewidth=1.3, edgecolor=VAR_ACCENT,
        facecolor=VAR_ACCENT, alpha=0.08, zorder=3,
    )
    ax.add_patch(vwl)
    ax.text(74.5, 80.0, "Variance-Weighted",
            ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=VAR_ACCENT, zorder=5)
    ax.text(74.5, 75.8, "Loss",
            ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=VAR_ACCENT, zorder=5)

    # Formula
    ax.text(74.5, 68.5,
            r"$\mathcal{L}\;=\;\sum_{i}\,w_i\,(D_i - \hat\tau)^2$",
            ha="center", va="center",
            fontsize=9.3, color=TEXT_DARK, zorder=5)

    # Bar chart
    bar_base_x = 58.0
    bar_base_y = 52.0
    bar_labels = [r"low $\sigma^2$", "mid", r"high $\sigma^2$"]
    bar_heights = [10.0, 5.5, 2.3]
    bar_alphas = [0.90, 0.58, 0.28]
    for i, (h, a, lbl) in enumerate(zip(bar_heights, bar_alphas, bar_labels)):
        x = bar_base_x + i * 9.0
        ax.add_patch(mpatches.Rectangle(
            (x, bar_base_y), 6.5, h,
            facecolor=VAR_ACCENT, alpha=a, edgecolor=VAR_ACCENT,
            linewidth=0.5, zorder=4,
        ))
        ax.text(x + 3.25, bar_base_y - 1.5, lbl,
                ha="center", va="top",
                fontsize=6.0, color=VAR_ACCENT, zorder=5)
    ax.annotate("", xy=(87.8, 52.0), xytext=(87.8, 63.5),
                arrowprops=dict(arrowstyle="<-", color=VAR_ACCENT, lw=0.8,
                                shrinkA=0, shrinkB=0),
                zorder=5)
    ax.text(89.0, 57.5, r"$w_i$",
            ha="left", va="center",
            fontsize=7.8, color=VAR_ACCENT, style="italic", zorder=5)

    # Takeaway lines
    ax.text(74.5, 42.5, r"$\Rightarrow\;$ reduces variance",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=VAR_ACCENT, zorder=5)
    ax.text(74.5, 37.5,
            r"cuts $Z^*\!\to \sigma^2(Y)$",
            ha="center", va="center",
            fontsize=7.0, style="italic", color=VAR_ACCENT, zorder=5)



def _mini_box(ax, x, y, w, h, label, colour, *, dashed=False, text_size=7.5):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=1.0",
        linewidth=0.8, edgecolor=colour,
        facecolor="white", alpha=0.92, zorder=4,
        linestyle=(0, (3, 2)) if dashed else "-",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=text_size, color=colour, zorder=5)


def main():
    fig = plt.figure(figsize=(3.45, 4.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(28, 155)
    ax.axis("off")

    ax.text(50.0, 151.0,
            "PACT: diagnosing & correcting the dual root",
            ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=TEXT_DARK)

    draw_zone_a(ax)
    draw_zone_b(ax)
    draw_zone_c(ax)

    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
