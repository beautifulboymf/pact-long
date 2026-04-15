"""Render Figure 1 (cover illustration) for the PACT paper — v4.

Design: twin panels of the same graph, colour-coded by two different
consequences of the same latent structural position Z*.
  Left panel  : treatment propensity pi_i = sigmoid(alpha * z_i)
                (coral -> grey -> teal diverging colourmap)
  Right panel : outcome-noise variance sigma^2_i = monotone(|z_i|)
                (white -> purple sequential colourmap)
A banner above both panels names the single latent cause and cites
Proposition 1.

z_i is the Fiedler vector of the graph Laplacian — a principled
continuous 1-D "structural position" score that drops any notion of
discrete community. No community colouring anywhere.

Run from the repository root:
    python images/scripts/make_cover.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

SEED = 23
OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PDF = OUT_DIR / "Uplift_cover.pdf"
OUT_PNG = OUT_DIR / "Uplift_cover_preview.png"

# ---- palette (matches arch.tex) -----------------------------------
BIAS_ACCENT = "#2A8A8A"     # teal = bias-pathway accent
VAR_ACCENT = "#8A4FAD"      # purple = variance-pathway accent
EDGE_GREY = "#B3B3B3"
TEXT_DARK = "#222222"
TEXT_GREY = "#555555"
LATENT_GREY = "#333333"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    }
)


def make_graph(rng: np.random.Generator) -> tuple[nx.Graph, dict[int, tuple[float, float]], np.ndarray]:
    """A mild-modularity graph that is *not* a clean two-community SBM."""
    n = 48
    # Build a soft latent-position graph: each node is placed along a line in
    # [-1, 1] (our z*), edges formed with probability decaying in distance.
    latent = np.sort(rng.uniform(-1, 1, size=n))
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(latent[i] - latent[j])
            p = 0.85 * np.exp(-4.5 * d)
            if rng.random() < p:
                g.add_edge(i, j)
    # Guarantee connectivity by adding a spanning chain if needed.
    if not nx.is_connected(g):
        comps = list(nx.connected_components(g))
        for a, b in zip(comps, comps[1:]):
            g.add_edge(next(iter(a)), next(iter(b)))

    # Compute the Fiedler vector (second eigenvector of the normalised
    # Laplacian). We do this by hand with numpy to avoid a scipy.sparse
    # version mismatch with networkx's built-in path.
    A = nx.to_numpy_array(g, nodelist=range(n))
    D = np.diag(A.sum(axis=1))
    L = D - A
    # Symmetric eigendecomposition, ascending order.
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    # Normalise to [-1, 1] so downstream mappings are stable.
    fiedler = fiedler / max(np.max(np.abs(fiedler)), 1e-9)

    # Layout: Kamada–Kawai using the Fiedler vector as one anchoring coord
    # so the 1-D structure is visible, with a second perpendicular coord for
    # visual spread.
    init = {i: (fiedler[i] * 2.0, rng.normal(0, 0.35)) for i in range(n)}
    pos = nx.kamada_kawai_layout(g, pos=init, scale=1.8)

    # Re-anchor so the Fiedler axis remains horizontal.
    xs = np.array([pos[i][0] for i in range(n)])
    ys = np.array([pos[i][1] for i in range(n)])
    # Align x with the Fiedler vector (sign-flip if inverted).
    if np.corrcoef(xs, fiedler)[0, 1] < 0:
        fiedler = -fiedler
    return g, pos, fiedler


def _draw_graph_panel(ax, g, pos, node_values, cmap, vmin, vmax,
                      border_colour, subtitle, latent_expr):
    """Shared panel renderer."""
    # Edges
    for u, v in g.edges():
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=EDGE_GREY,
            linewidth=0.5,
            alpha=0.45,
            zorder=1,
            solid_capstyle="round",
        )
    # Nodes
    cm = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in g.nodes():
        x, y = pos[i]
        colour = cm(norm(node_values[i]))
        ax.scatter(
            x, y,
            s=170,
            c=[colour],
            edgecolors="#FFFFFF",
            linewidths=0.9,
            zorder=3,
        )
    # Panel border (soft)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")

    # Titles
    ax.set_title(subtitle, color=border_colour, fontsize=10.2,
                 fontweight="bold", pad=10)
    # Latent-DAG caption under the title
    ax.text(0.5, 0.985, latent_expr, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.8,
            color=TEXT_DARK)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(min(p[0] for p in pos.values()) - 0.3,
                max(p[0] for p in pos.values()) + 0.3)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.3,
                max(p[1] for p in pos.values()) + 0.3)
    return cm, norm


def _add_colourbar(fig, ax, cmap, norm, label):
    # Add a thin horizontal colourbar UNDER the panel.
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x0 + 0.05, bbox.y0 - 0.03,
                        bbox.width - 0.1, 0.015])
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(cmap), norm=norm,
        orientation="horizontal",
    )
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=7, length=2, pad=1.5, colors=TEXT_GREY)
    cb.set_label(label, fontsize=7.5, color=TEXT_GREY, labelpad=2)


def draw_top_banner(fig):
    """Clean banner above the two panels naming the shared latent cause."""
    ax = fig.add_axes([0.04, 0.865, 0.92, 0.13])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Central Z* pill — made wider and taller so the math does not overflow.
    pill_x, pill_y = 0.30, 0.26
    pill_w, pill_h = 0.40, 0.56
    pill = mpatches.FancyBboxPatch(
        (pill_x, pill_y), pill_w, pill_h,
        boxstyle="round,pad=0.015,rounding_size=0.05",
        linewidth=1.1, edgecolor=LATENT_GREY, facecolor="#F7F5F2",
        zorder=3,
    )
    ax.add_patch(pill)
    ax.text(pill_x + pill_w / 2, pill_y + 0.42,
            r"latent position $Z^*$",
            ha="center", va="center",
            fontsize=10.0, color=LATENT_GREY, fontweight="bold")
    ax.text(pill_x + pill_w / 2, pill_y + 0.16,
            r"$\Rightarrow\;\mathrm{PEHE}^2 = \mathrm{bias}^2 + \mathrm{variance}$"
            "   (Prop. 1)",
            ha="center", va="center", fontsize=8.8, color=TEXT_DARK)

    # Left arrow with bias label
    ax.annotate("", xy=(0.09, pill_y + pill_h / 2),
                xytext=(pill_x - 0.005, pill_y + pill_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=BIAS_ACCENT,
                                lw=1.2, shrinkA=2, shrinkB=2))
    ax.text(0.2, pill_y + pill_h / 2 + 0.22,
            "bias channel", ha="center", va="bottom",
            fontsize=8.8, color=BIAS_ACCENT, fontweight="bold")

    # Right arrow with variance label
    ax.annotate("", xy=(0.91, pill_y + pill_h / 2),
                xytext=(pill_x + pill_w + 0.005, pill_y + pill_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=VAR_ACCENT,
                                lw=1.2, shrinkA=2, shrinkB=2))
    ax.text(0.8, pill_y + pill_h / 2 + 0.22,
            "variance channel", ha="center", va="bottom",
            fontsize=8.8, color=VAR_ACCENT, fontweight="bold")

    # Subtitle strap
    ax.text(0.5, 0.05,
            "same graph, two colourings; both track the same latent position",
            ha="center", va="center", fontsize=8.2, color=TEXT_GREY,
            style="italic")


def main() -> None:
    rng = np.random.default_rng(SEED)
    g, pos, z = make_graph(rng)

    # ---- derive the two channels from z ----
    # Treatment propensity: diverging, centred at 0.5.
    pi = 1.0 / (1.0 + np.exp(-2.8 * z))
    # Outcome-noise variance: U-shaped in z (position-magnitude drives noise),
    # normalised so the lowest-variance node is near 0 and the highest near 1.
    sigma2 = 0.15 + 0.85 * np.abs(z)
    sigma2 = (sigma2 - sigma2.min()) / (sigma2.max() - sigma2.min())

    # Build the colour maps.
    # Diverging coral↔teal for propensity (untreated=coral, treated=teal).
    prop_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "prop_div", ["#E07A5F", "#EAE4D2", BIAS_ACCENT], N=256
    )
    # White -> purple sequential for variance.
    var_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "var_seq", ["#F2EEF5", "#C7A8D9", VAR_ACCENT, "#4A2566"], N=256
    )

    # ---- figure layout: banner (top) + two panels (bottom) ----
    fig = plt.figure(figsize=(7.6, 4.8))
    ax_left = fig.add_axes([0.055, 0.12, 0.42, 0.68])
    ax_right = fig.add_axes([0.535, 0.12, 0.42, 0.68])

    cm_l, norm_l = _draw_graph_panel(
        ax_left, g, pos, pi, prop_cmap, 0.0, 1.0,
        BIAS_ACCENT,
        "treatment propensity — bias source",
        r"$Z^* \rightarrow T,\; Z^* \rightarrow Y(t)$",
    )
    cm_r, norm_r = _draw_graph_panel(
        ax_right, g, pos, sigma2, var_cmap, 0.0, 1.0,
        VAR_ACCENT,
        "outcome-noise variance — variance source",
        r"$Z^* \rightarrow \sigma^2(Y)$",
    )

    # Thin colourbars under each panel.
    _add_colourbar(fig, ax_left, prop_cmap, norm_l,
                   r"$\pi_i = \Pr(T_i = 1 \mid Z^*_i)$")
    _add_colourbar(fig, ax_right, var_cmap, norm_r,
                   r"$\sigma^2_i = \mathrm{Var}(Y_i \mid Z^*_i)$")

    draw_top_banner(fig)

    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
