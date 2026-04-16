"""Render Figure 2 (architecture) for the PACT paper.

Matplotlib reimplementation of the training-pipeline figure, matching
the aesthetic of make_cover.py (same palette, fonts, box styling).
Horizontal flow showing how the two plug-ins attach to an arbitrary
base model without modifying its internals.

Run from pact-long/:
    python images/scripts/make_arch.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PDF = OUT_DIR / "arch.pdf"
OUT_PNG = OUT_DIR / "arch_preview.png"

# --- palette (shared with make_cover.py) ---
BIAS_ACCENT = "#2A8A8A"       # teal    — GPE / bias channel
VAR_ACCENT = "#8A4FAD"        # purple  — VWL / variance channel
DETACH_RED = "#C94040"        # detached-gradient arrow
NEUTRAL_EDGE = "#888888"
NEUTRAL_FILL = "#F3F1ED"
TEXT_DARK = "#1F1F1F"
TEXT_GREY = "#5A5A5A"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ---------------- helpers ----------------
def _rounded(ax, x, y, w, h, *, ec, fc, lw=1.0, dashed=False, zorder=3,
             alpha_fc=1.0):
    linestyle = (0, (3, 2)) if dashed else "-"
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.4",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha_fc,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(box)
    return (x + w / 2, y + h / 2)  # centre


def _arrow(ax, x0, y0, x1, y1, *, colour, lw=0.9, style="-"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=colour, lw=lw,
            linestyle=style, shrinkA=0, shrinkB=0,
        ),
        zorder=4,
    )


def _dashed_arrow(ax, x0, y0, x1, y1, *, colour):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=colour, lw=1.0,
            linestyle=(0, (3.5, 2.5)), shrinkA=0, shrinkB=0,
        ),
        zorder=4,
    )


def _badge(ax, x, y, text, colour):
    """Small 'PACT plug-in' capsule, centred at (x, y)."""
    w, h = 9.5, 2.4
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=1.0",
        linewidth=0.7, edgecolor=colour, facecolor="white",
        zorder=6,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=6.0, fontweight="bold", color=colour, zorder=7)


# ---------------- main drawing ----------------
def draw(ax):
    # === background bands behind the two plug-ins ===
    # GPE band
    gpe_band = mpatches.FancyBboxPatch(
        (13.5, 3.5), 17.5, 35.0,
        boxstyle="round,pad=0.3,rounding_size=1.8",
        linewidth=0.5, edgecolor=f"{BIAS_ACCENT}33",
        facecolor=BIAS_ACCENT, alpha=0.06, zorder=1,
    )
    ax.add_patch(gpe_band)
    # VWL band — wider, encompasses the pilot + sigma head below
    vwl_band = mpatches.FancyBboxPatch(
        (66.5, 1.0), 32.0, 37.5,
        boxstyle="round,pad=0.3,rounding_size=1.8",
        linewidth=0.5, edgecolor=f"{VAR_ACCENT}33",
        facecolor=VAR_ACCENT, alpha=0.06, zorder=1,
    )
    ax.add_patch(vwl_band)

    # ============ main pipeline blocks ============
    # y-band for pipeline: y = 22..34
    PY = 22.0
    PH = 12.0

    # 1) Input (X, G)
    input_c = _rounded(ax, 2.0, PY, 9.0, PH,
                       ec=NEUTRAL_EDGE, fc=NEUTRAL_FILL, lw=0.8)
    ax.text(input_c[0], input_c[1], r"$X,\;\mathcal{G}$",
            ha="center", va="center",
            fontsize=11, color=TEXT_DARK, zorder=5)

    # 2) GPE plug-in (teal, prominent)
    gpe_x, gpe_y, gpe_w, gpe_h = 15.5, 17.0, 13.5, 22.0
    _rounded(ax, gpe_x, gpe_y, gpe_w, gpe_h,
             ec=BIAS_ACCENT, fc=BIAS_ACCENT, lw=1.3, alpha_fc=0.14)
    gpe_cx = gpe_x + gpe_w / 2
    ax.text(gpe_cx, gpe_y + gpe_h - 4.0, "GPE",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=BIAS_ACCENT, zorder=5)
    ax.text(gpe_cx, gpe_y + gpe_h - 9.5,
            r"$X \,\oplus\, p(Z^*)$",
            ha="center", va="center",
            fontsize=9.0, color=BIAS_ACCENT, zorder=5)
    ax.text(gpe_cx, gpe_y + gpe_h - 14.0, "node2vec",
            ha="center", va="center",
            fontsize=7.0, style="italic", color=BIAS_ACCENT, zorder=5)
    ax.text(gpe_cx, gpe_y + gpe_h - 17.5, "+ cross-attn",
            ha="center", va="center",
            fontsize=7.0, style="italic", color=BIAS_ACCENT, zorder=5)
    _badge(ax, gpe_cx, gpe_y + gpe_h + 2.5, "PACT plug-in", BIAS_ACCENT)

    # 3) base model M (neutral anchor)
    M_x, M_y, M_w, M_h = 32.0, 17.0, 14.0, 22.0
    _rounded(ax, M_x, M_y, M_w, M_h,
             ec=NEUTRAL_EDGE, fc=NEUTRAL_FILL, lw=1.0)
    M_cx = M_x + M_w / 2
    ax.text(M_cx, M_y + M_h - 4.0, "base model",
            ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=TEXT_DARK, zorder=5)
    ax.text(M_cx, M_y + M_h - 8.5, r"$M$",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=TEXT_DARK, zorder=5)
    ax.text(M_cx, M_y + 5.5, "any graph ITE backbone:",
            ha="center", va="center",
            fontsize=6.4, style="italic", color=TEXT_GREY, zorder=5)
    ax.text(M_cx, M_y + 3.5, "BNN, TARNet, NetDeconf,",
            ha="center", va="center",
            fontsize=6.0, color=TEXT_GREY, zorder=5)
    ax.text(M_cx, M_y + 1.5, "GIAL, GNUM, GDC, X-learner",
            ha="center", va="center",
            fontsize=6.0, color=TEXT_GREY, zorder=5)

    # 4) outcome heads μ̂_0, μ̂_1
    mu_c = _rounded(ax, 49.0, PY, 10.0, PH,
                    ec=NEUTRAL_EDGE, fc=NEUTRAL_FILL, lw=0.8)
    ax.text(mu_c[0], mu_c[1], r"$\hat\mu_0,\;\hat\mu_1$",
            ha="center", va="center",
            fontsize=11, color=TEXT_DARK, zorder=5)

    # 5) pseudo-outcome D
    d_c = _rounded(ax, 61.0, PY, 13.0, PH,
                   ec=NEUTRAL_EDGE, fc=NEUTRAL_FILL, lw=0.8)
    ax.text(d_c[0], d_c[1], r"$D = Y - \hat\mu_{1-t}$",
            ha="center", va="center",
            fontsize=10, color=TEXT_DARK, zorder=5)

    # 6) VWL plug-in (purple, prominent)
    vwl_x, vwl_y, vwl_w, vwl_h = 77.0, 17.0, 20.0, 22.0
    _rounded(ax, vwl_x, vwl_y, vwl_w, vwl_h,
             ec=VAR_ACCENT, fc=VAR_ACCENT, lw=1.3, alpha_fc=0.12)
    vwl_cx = vwl_x + vwl_w / 2
    ax.text(vwl_cx, vwl_y + vwl_h - 3.8, "Variance-Weighted Loss",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=VAR_ACCENT, zorder=5)
    ax.text(vwl_cx, vwl_y + vwl_h - 9.5,
            r"$\mathcal{L} = \sum_i w_i\,(D_i - \hat\tau)^2$",
            ha="center", va="center",
            fontsize=9.5, color=TEXT_DARK, zorder=5)
    ax.text(vwl_cx, vwl_y + vwl_h - 14.5,
            r"$w_i = 1/\max\{\hat\sigma^2_t, \delta\}$",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=VAR_ACCENT, zorder=5)
    _badge(ax, vwl_cx, vwl_y + vwl_h + 2.5, "PACT plug-in", VAR_ACCENT)

    # ============ subordinate row: pilot + log-sigma^2 head ============
    # pilot τ̂ (detached): dashed neutral box
    pilot_x, pilot_y, pilot_w, pilot_h = 62.0, 6.0, 15.0, 6.5
    _rounded(ax, pilot_x, pilot_y, pilot_w, pilot_h,
             ec=NEUTRAL_EDGE, fc="white", lw=0.7, dashed=True)
    pilot_cx = pilot_x + pilot_w / 2
    ax.text(pilot_cx, pilot_y + pilot_h / 2,
            r"pilot $\hat\tau$  (detached)",
            ha="center", va="center",
            fontsize=7.5, style="italic", color=TEXT_GREY, zorder=5)

    # log-sigma^2 head (purple small block)
    sh_x, sh_y, sh_w, sh_h = 80.0, 6.0, 14.0, 6.5
    _rounded(ax, sh_x, sh_y, sh_w, sh_h,
             ec=VAR_ACCENT, fc=VAR_ACCENT, lw=0.9, alpha_fc=0.12)
    sh_cx = sh_x + sh_w / 2
    ax.text(sh_cx, sh_y + sh_h / 2,
            r"$\log\hat\sigma^2_t$ head",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=VAR_ACCENT, zorder=5)

    # ============ arrows ============
    # main pipeline (left to right)
    _arrow(ax, 11.0, PY + PH / 2, 15.5, PY + PH / 2, colour=BIAS_ACCENT, lw=1.2)
    _arrow(ax, 29.0, PY + PH / 2, 32.0, PY + PH / 2, colour=NEUTRAL_EDGE, lw=1.0)
    _arrow(ax, 46.0, PY + PH / 2, 49.0, PY + PH / 2, colour=NEUTRAL_EDGE, lw=1.0)
    _arrow(ax, 59.0, PY + PH / 2, 61.0, PY + PH / 2, colour=NEUTRAL_EDGE, lw=1.0)
    _arrow(ax, 74.0, PY + PH / 2, 77.0, PY + PH / 2, colour=VAR_ACCENT, lw=1.2)

    # pilot -> sigma head (dashed red, detached gradient)
    _dashed_arrow(ax, pilot_x + pilot_w, pilot_y + pilot_h / 2,
                  sh_x, sh_y + sh_h / 2, colour=DETACH_RED)

    # sigma head -> VWL (purple, upward)
    _arrow(ax, sh_cx, sh_y + sh_h, sh_cx, vwl_y,
           colour=VAR_ACCENT, lw=1.2)


def main():
    # ACM sigconf figure* is ~7 in wide. Use 7.1 × 3.2.
    fig = plt.figure(figsize=(7.1, 3.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)
    ax.axis("off")

    draw(ax)

    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
