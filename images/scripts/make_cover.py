"""Render Figure 1 (cover illustration) for the PACT paper — v5.

Design changes over v4:
  * 2-D latent positions (not 1-D Fiedler). Horizontal axis drives the
    LEFT panel's propensity; vertical axis drives the RIGHT panel's
    variance. The two channels therefore dissociate on individual
    nodes while still sharing a single latent cause Z*.
  * Sparser graph: 30 nodes, radius-ball connectivity thinned to
    average degree ~3.2. No hairballs.
  * Representative "target" user pinned near (0, 0) in latent space —
    medium propensity and medium variance — with a visible halo.
  * Per-node sampling noise so propensity and variance values are not
    deterministic functions of position; the reader sees statistical
    co-variation rather than two identical colourmaps.

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

SEED = 41
OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PDF = OUT_DIR / "Uplift_cover.pdf"
OUT_PNG = OUT_DIR / "Uplift_cover_preview.png"

BIAS_ACCENT = "#2A8A8A"    # teal, bias channel
VAR_ACCENT = "#8A4FAD"     # purple, variance channel
TARGET_ORANGE = "#E08E44"
HALO_ORANGE = "#F2B97C"
EDGE_GREY = "#D0D0D0"
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


def sample_latent(n: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-mixture latent positions in 2-D, centred at origin."""
    centres = np.array([[-1.2, 0.3], [1.1, 0.2], [0.0, -1.0], [0.1, 1.1]])
    weights = [0.32, 0.32, 0.18, 0.18]
    assignments = rng.choice(len(centres), size=n, p=weights)
    points = centres[assignments] + 0.45 * rng.normal(size=(n, 2))
    # Centre the cloud on the origin (so the anchor point sits at (0,0)).
    points = points - points.mean(axis=0, keepdims=True)
    return points


def build_graph(positions: np.ndarray, rng: np.random.Generator,
                target_degree: float = 3.2) -> nx.Graph:
    """Radius-ball graph, thinned down to a target average degree."""
    n = positions.shape[0]
    dists = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    # First pass: connect everything within a radius that gives plenty of
    # candidate edges; we then thin.
    radius = 1.1
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < radius:
                # short edges get higher score; random jitter so ties break.
                score = dists[i, j] + 0.15 * rng.random()
                candidates.append((score, i, j))
    candidates.sort()

    g = nx.Graph()
    g.add_nodes_from(range(n))

    # Phase 1 — ensure connectivity via a minimum spanning tree of the
    # fully-connected distance graph.
    mst = nx.minimum_spanning_tree(
        nx.from_numpy_array(dists), algorithm="kruskal"
    )
    g.add_edges_from(mst.edges())

    # Phase 2 — add short-range candidate edges until we hit target degree.
    target_edges = int(round(target_degree * n / 2))
    for _score, i, j in candidates:
        if g.number_of_edges() >= target_edges:
            break
        if not g.has_edge(i, j):
            g.add_edge(i, j)

    return g


def main() -> None:
    rng = np.random.default_rng(SEED)

    n = 30
    lat = sample_latent(n, rng)

    # Pin one node to the origin — the "target user" anchor.
    target = int(np.argmin(np.linalg.norm(lat, axis=1)))
    lat[target] = [0.0, 0.0]

    g = build_graph(lat, rng)

    # Layout: use latent positions directly. Tiny jitter for labels but keep
    # the correlation between page-position and latent-position.
    pos = {i: tuple(lat[i]) for i in range(n)}

    # --- derive the two channels -------------------------------------
    # Left panel: treatment propensity tracks the HORIZONTAL axis.
    pi_clean = 1.0 / (1.0 + np.exp(-2.3 * lat[:, 0]))
    pi = np.clip(pi_clean + rng.normal(0, 0.07, size=n), 0.01, 0.99)

    # Right panel: outcome noise variance tracks the VERTICAL axis magnitude.
    sigma2_clean = 0.25 + 0.75 * np.clip(np.abs(lat[:, 1]) / 1.5, 0, 1)
    sigma2 = np.clip(sigma2_clean + rng.normal(0, 0.08, size=n), 0.02, 1.0)
    sigma2 = (sigma2 - sigma2.min()) / (sigma2.max() - sigma2.min())

    # --- colourmaps --------------------------------------------------
    prop_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "prop_div", ["#E07A5F", "#EAE4D2", BIAS_ACCENT], N=256
    )
    var_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "var_seq", ["#F7F2FB", "#D8C0E4", VAR_ACCENT, "#4A2566"], N=256
    )

    # --- figure layout -----------------------------------------------
    fig = plt.figure(figsize=(7.6, 4.8))
    ax_left = fig.add_axes([0.05, 0.12, 0.42, 0.64])
    ax_right = fig.add_axes([0.535, 0.12, 0.42, 0.64])

    _draw_panel(
        ax_left, g, pos, pi, prop_cmap, 0.0, 1.0,
        title="treatment propensity",
        title_color=BIAS_ACCENT,
        subtitle=r"$Z^* \to T,\;\; Z^* \to Y(t)$ — bias root",
        target=target,
    )
    _draw_panel(
        ax_right, g, pos, sigma2, var_cmap, 0.0, 1.0,
        title="outcome-noise variance",
        title_color=VAR_ACCENT,
        subtitle=r"$Z^* \to \sigma^2(Y)$ — variance root",
        target=target,
    )

    _add_colourbar(fig, ax_left, prop_cmap, 0.0, 1.0,
                   r"$\pi_i = \Pr(T_i = 1 \mid Z^*_i)$")
    _add_colourbar(fig, ax_right, var_cmap, 0.0, 1.0,
                   r"$\sigma^2_i = \mathrm{Var}(Y_i \mid Z^*_i)$")

    _draw_banner(fig)

    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


def _draw_panel(ax, g, pos, values, cmap, vmin, vmax,
                *, title, title_color, subtitle, target):
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

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)

    # Non-target nodes
    for v in g.nodes():
        if v == target:
            continue
        x, y = pos[v]
        ax.scatter(
            x, y,
            s=185,
            c=[cm(norm(values[v]))],
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )

    # Target node halo + dot — same orange in both panels so reader
    # tracks the same node across the two channels.
    tx, ty = pos[target]
    ax.add_patch(mpatches.Circle(
        (tx, ty), 0.24, facecolor=HALO_ORANGE, edgecolor="none",
        alpha=0.55, zorder=3.4,
    ))
    ax.scatter(
        tx, ty,
        s=260,
        c=[cm(norm(values[target]))],
        edgecolors=TARGET_ORANGE,
        linewidths=1.8,
        zorder=4,
    )

    # Title + subtitle inside the panel
    ax.text(0.02, 0.97, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=11.0, fontweight="bold",
            color=title_color)
    ax.text(0.02, 0.90, subtitle, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.0, color=TEXT_GREY)

    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Axis limits tight on the content.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.35
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def _add_colourbar(fig, ax, cmap, vmin, vmax, label):
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x0 + 0.06, bbox.y0 - 0.045,
                        bbox.width - 0.12, 0.016])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(cmap), norm=norm,
        orientation="horizontal",
    )
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=7.5, length=2, pad=1.5, colors=TEXT_GREY)
    cb.set_label(label, fontsize=7.8, color=TEXT_GREY, labelpad=2)


def _draw_banner(fig):
    ax = fig.add_axes([0.05, 0.81, 0.92, 0.17])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Central pill — large enough to comfortably hold two lines.
    pill_x, pill_y = 0.28, 0.15
    pill_w, pill_h = 0.44, 0.72
    pill = mpatches.FancyBboxPatch(
        (pill_x, pill_y), pill_w, pill_h,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=1.1, edgecolor=LATENT_GREY, facecolor="#F7F5F2",
        zorder=3,
    )
    ax.add_patch(pill)
    ax.text(0.5, pill_y + 0.65 * pill_h,
            r"latent position $Z^*$", ha="center", va="center",
            fontsize=11.0, color=LATENT_GREY, fontweight="bold")
    ax.text(0.5, pill_y + 0.32 * pill_h,
            r"$\Rightarrow\;\mathrm{PEHE}^2 = \mathrm{bias}^2 + \mathrm{variance}$",
            ha="center", va="center", fontsize=9.3, color=TEXT_DARK)
    ax.text(0.5, pill_y + 0.09 * pill_h,
            "(Proposition 1)",
            ha="center", va="center", fontsize=8.5, color=TEXT_GREY,
            style="italic")

    # Left channel arrow/label.
    mid_y = pill_y + 0.5 * pill_h
    ax.annotate("", xy=(0.045, mid_y),
                xytext=(pill_x - 0.003, mid_y),
                arrowprops=dict(arrowstyle="-|>", color=BIAS_ACCENT,
                                lw=1.3, shrinkA=2, shrinkB=2))
    ax.text((pill_x - 0.005 + 0.045) / 2, mid_y + 0.14,
            "bias channel", ha="center", va="bottom",
            fontsize=9.6, color=BIAS_ACCENT, fontweight="bold")

    # Right channel arrow/label.
    ax.annotate("", xy=(0.955, mid_y),
                xytext=(pill_x + pill_w + 0.003, mid_y),
                arrowprops=dict(arrowstyle="-|>", color=VAR_ACCENT,
                                lw=1.3, shrinkA=2, shrinkB=2))
    ax.text((pill_x + pill_w + 0.003 + 0.955) / 2, mid_y + 0.14,
            "variance channel", ha="center", va="bottom",
            fontsize=9.6, color=VAR_ACCENT, fontweight="bold")


if __name__ == "__main__":
    main()
