"""Render the ablations figure — merges old Tables 3 (GDC ablation at rho=30)
and 4 (tabular Qini) into a single two-panel bar chart.

Left panel : PEHE (down) for GDC vs. +GPE vs. +Var vs. +GPE+Var on CoraFull,
             DBLP, PubMed at rho=30.
Right panel: Qini (up) for S vs. S+Var on six tabular uplift settings.

Run from pact-long/:
    python images/scripts/make_ablations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PDF = OUT_DIR / "ablations.pdf"
OUT_PNG = OUT_DIR / "ablations_preview.png"

# --- palette matches make_cover.py / make_arch.py ---
BIAS_ACCENT = "#2A8A8A"       # teal   — +GPE
VAR_ACCENT  = "#8A4FAD"       # purple — +Var
BOTH_ACCENT = "#4A6E7A"       # blend  — +GPE+Var
GREY        = "#7F7F7F"       # baseline
TEXT_DARK   = "#1F1F1F"
TEXT_GREY   = "#5A5A5A"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#3A3A3A",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    }
)


# =====================================================================
# Left panel — GDC ablation at rho=30 (PEHE down)
# =====================================================================
GDC_DATASETS = ["CoraFull", "DBLP", "PubMed"]
GDC_CONDS = ["GDC", "+GPE", "+Var", "+GPE+Var"]
GDC_COLORS = [GREY, BIAS_ACCENT, VAR_ACCENT, BOTH_ACCENT]
# rows: datasets, cols: conditions
GDC_MEAN = np.array(
    [
        [4.64, 5.14, 5.04, 3.99],   # CoraFull
        [4.02, 3.80, 3.87, 3.22],   # DBLP
        [4.47, 4.19, 3.19, 3.55],   # PubMed
    ]
)
GDC_STD = np.array(
    [
        [0.31, 0.95, 0.35, 0.58],
        [0.04, 0.14, 0.85, 0.16],
        [0.43, 0.60, 0.34, 0.88],
    ]
)
# per-row best is whichever +variant has lowest mean (used for bolding/star)
GDC_BEST_COL = GDC_MEAN[:, 1:].argmin(axis=1) + 1   # skip plain GDC column


# =====================================================================
# Right panel — tabular Qini (up)
# =====================================================================
TAB_LABELS = [
    "Hill-spend",
    "Hill-visit",
    "X5",
    "Criteo",
    "Lenta",
    "R-Hero$^\\dagger$",
]
TAB_S       = np.array([0.029, -0.001, 0.013, 0.167, 0.001, 0.008])
TAB_SVAR    = np.array([0.117, -0.005, 0.013, 0.151, 0.001, 0.009])
TAB_S_STD   = np.array([0.065,  0.012, 0.003, 0.037, 0.006, 0.004])
TAB_SVAR_STD = np.array([0.076, 0.012, 0.002, 0.038, 0.008, 0.003])


# =====================================================================
# Draw
# =====================================================================
def draw_left(ax):
    n_ds = len(GDC_DATASETS)
    n_cond = len(GDC_CONDS)
    x = np.arange(n_ds)
    bar_w = 0.20
    offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * bar_w

    for c, (cond, colour, off) in enumerate(zip(GDC_CONDS, GDC_COLORS, offsets)):
        means = GDC_MEAN[:, c]
        stds  = GDC_STD[:, c]
        bars = ax.bar(
            x + off, means, width=bar_w * 0.92,
            color=colour, edgecolor="none",
            label=cond, zorder=3,
        )
        ax.errorbar(
            x + off, means, yerr=stds,
            fmt="none", ecolor="#333333", elinewidth=0.55,
            capsize=1.4, capthick=0.55, zorder=4,
        )
        # star the best condition in each dataset row (skipping plain GDC)
        for row_i, best_c in enumerate(GDC_BEST_COL):
            if c == best_c:
                ax.text(row_i + offsets[c],
                        GDC_MEAN[row_i, c] + GDC_STD[row_i, c] + 0.18,
                        "$\\star$",
                        ha="center", va="bottom",
                        fontsize=6.0, color=BOTH_ACCENT, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(GDC_DATASETS, fontsize=5.8)
    ax.set_ylabel("PEHE\u2009($\\downarrow$)", fontsize=6.4)
    ax.set_title("GDC ablation, $\\rho{=}30$",
                 fontsize=6.8, fontweight="bold", color=TEXT_DARK, pad=2)
    ax.tick_params(axis="y", labelsize=5.6)
    ax.set_ylim(0, max(GDC_MEAN.max() + GDC_STD.max() + 0.8, 6.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#B5B5B5", zorder=1)
    ax.set_axisbelow(True)

    ax.legend(
        fontsize=5.2, loc="upper right", frameon=False,
        handlelength=0.8, handletextpad=0.25, columnspacing=0.5,
        ncol=2, borderaxespad=0.2,
    )


def draw_right(ax):
    n = len(TAB_LABELS)
    x = np.arange(n)
    bar_w = 0.36

    ax.bar(
        x - bar_w / 2, TAB_S, width=bar_w,
        color=GREY, edgecolor="none", label="S", zorder=3,
    )
    ax.bar(
        x + bar_w / 2, TAB_SVAR, width=bar_w,
        color=BIAS_ACCENT, edgecolor="none", label="S$+$Var", zorder=3,
    )
    ax.errorbar(
        x - bar_w / 2, TAB_S, yerr=TAB_S_STD,
        fmt="none", ecolor="#333333", elinewidth=0.55,
        capsize=1.4, capthick=0.55, zorder=4,
    )
    ax.errorbar(
        x + bar_w / 2, TAB_SVAR, yerr=TAB_SVAR_STD,
        fmt="none", ecolor="#333333", elinewidth=0.55,
        capsize=1.4, capthick=0.55, zorder=4,
    )

    # Highlight the Hillstrom-spend 4x jump with a small annotation
    ax.text(
        x[0] + bar_w / 2, TAB_SVAR[0] + TAB_SVAR_STD[0] + 0.02,
        "$4\\times$",
        ha="center", va="bottom",
        fontsize=6.0, color=BIAS_ACCENT, fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(TAB_LABELS, fontsize=5.2, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Qini\u2009($\\uparrow$)", fontsize=6.4)
    ax.set_title("Tabular Qini",
                 fontsize=6.8, fontweight="bold", color=TEXT_DARK, pad=2)
    ax.tick_params(axis="y", labelsize=5.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="#777777", linewidth=0.4, zorder=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#B5B5B5", zorder=1)
    ax.set_axisbelow(True)

    # y-range with breathing room for the 4x label
    all_means = np.concatenate([TAB_S, TAB_SVAR])
    top = max(all_means) + 0.09
    bot = min(all_means) - 0.04
    ax.set_ylim(bot, top)

    ax.legend(
        fontsize=5.2, loc="upper right", frameon=False,
        handlelength=0.8, handletextpad=0.25,
        borderaxespad=0.2,
    )


def main():
    # Single-column width (≈ 3.35 in), panels side-by-side in a flat aspect
    # so the figure only occupies ~half the page vertically.
    fig, (ax_L, ax_R) = plt.subplots(
        1, 2, figsize=(3.35, 2.0), facecolor="white",
        gridspec_kw=dict(wspace=0.45),
    )
    draw_left(ax_L)
    draw_right(ax_R)

    fig.savefig(OUT_PDF, format="pdf", facecolor="white")
    fig.savefig(OUT_PNG, format="png", dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
