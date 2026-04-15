"""Render Figure 1 (cover illustration) for the PACT paper — v3.

Key upgrades over v2:
  * Larger stochastic-block-model (~28 nodes per community) rendered with a
    repulsive Kamada–Kawai layout so individual nodes and edges are visible
    instead of a blob.
  * Weak inter-community ties (multiple bridge edges) — looks like a real
    social graph, not a staged two-island picture.
  * sigma^2(Z_i) mapped to node *size* over a 4x range, plus outline width,
    so heteroscedasticity reads at thumbnail size.
  * Dual-root story made explicit: one horizontal baseline sits under the
    graph with two compact panels — "bias root" (community position ->
    T, Y(t)) and "variance root" (position -> sigma^2) — connected by a
    single shared arrow to the latent Z*.
  * Legend merged into the same baseline, no floating chips.

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

SEED = 11
OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PDF = OUT_DIR / "Uplift_cover.pdf"
OUT_PNG = OUT_DIR / "Uplift_cover_preview.png"

# Palette — cool blue + warm green + orange target + teal/purple accents
# matching the paper's pactaccent teal and a complementary variance accent.
COMM_BLUE = "#4C72B0"
COMM_GREEN = "#55A868"
TARGET_ORANGE = "#E08E44"
HALO_ORANGE = "#F2B97C"
NOISE_EDGE = "#7E7E7E"
EDGE_GREY = "#B0B0B0"
TEXT_DARK = "#2B2B2B"
TEXT_GREY = "#555555"
BIAS_ACCENT = "#2A8A8A"      # teal, matches pactaccent in arch.tex
VAR_ACCENT = "#8A4FAD"       # purple, complementary role
LATENT_GREY = "#3A3A3A"

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


def build_graph(rng: np.random.Generator):
    n1, n2 = 28, 28
    p_in, p_out = 0.18, 0.012
    g = nx.stochastic_block_model(
        [n1, n2], [[p_in, p_out], [p_out, p_in]], seed=SEED
    )

    # Enforce at least 5 inter-community bridge edges.
    inter = [(u, v) for u, v in g.edges if (u < n1) != (v < n1)]
    while len(inter) < 5:
        u = rng.integers(0, n1)
        v = rng.integers(n1, n1 + n2)
        if not g.has_edge(u, v):
            g.add_edge(u, v)
            inter.append((u, v))

    # Compute positions with a bipartite-ish initial guess so the two
    # communities lay out on the left and right halves, then refine with
    # Kamada–Kawai for smooth spacing.
    init = {}
    for v in range(n1):
        init[v] = (-1.6 + 0.2 * rng.normal(), 0.25 * rng.normal())
    for v in range(n1, n1 + n2):
        init[v] = (1.6 + 0.2 * rng.normal(), 0.25 * rng.normal())
    pos = nx.kamada_kawai_layout(g, pos=init, scale=1.9)

    # Anchor communities left/right in case kamada tilted them.
    for v in range(n1):
        x, y = pos[v]
        pos[v] = (-abs(x) - 0.4, y * 1.1)
    for v in range(n1, n1 + n2):
        x, y = pos[v]
        pos[v] = (abs(x) + 0.4, y * 1.1)

    # Heteroscedastic noise: highest near community boundaries (bridge-like
    # nodes) and lowest deep inside a community. Use betweenness centrality
    # as the proxy — high-betweenness nodes straddle the two camps.
    bc = nx.betweenness_centrality(g)
    bc_arr = np.array([bc[v] for v in g.nodes()])
    if bc_arr.max() > 0:
        bc_arr = bc_arr / bc_arr.max()
    # Map to [0.15, 1.0]; add a pinch of jitter so nodes inside a community
    # still vary.
    noise = 0.15 + 0.85 * bc_arr
    noise += rng.normal(0, 0.05, size=noise.shape)
    noise = np.clip(noise, 0.10, 1.10)

    # Target: the highest-betweenness node, pinned to the centre so it is
    # visually between the two communities.
    target = int(np.argmax(bc_arr))
    pos[target] = (0.0, 0.1)
    return g, pos, noise, target, n1


def draw_graph(ax, g, pos, noise, target, n1):
    # --- 1) soft community halos ----------------------------------
    for colour, members in [
        (COMM_BLUE, range(n1)),
        (COMM_GREEN, range(n1, 2 * n1)),
    ]:
        pts = np.array([pos[m] for m in members])
        centre = pts.mean(axis=0)
        span = np.linalg.norm(pts - centre, axis=1).max() + 0.3
        halo = mpatches.Circle(
            centre,
            span,
            facecolor=colour,
            edgecolor="none",
            alpha=0.11,
            zorder=0,
        )
        ax.add_patch(halo)

    # --- 2) edges --------------------------------------------------
    for u, v in g.edges():
        inter_edge = (u < n1) != (v < n1)
        is_target_edge = target in (u, v)
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=EDGE_GREY,
            linewidth=1.1 if is_target_edge else (0.7 if inter_edge else 0.45),
            alpha=0.85 if is_target_edge else (0.55 if inter_edge else 0.35),
            zorder=1,
            solid_capstyle="round",
        )

    # --- 3) nodes --------------------------------------------------
    #      size = 90 + 260 * noise    (≈ 4x dynamic range)
    #      edge_width linearly scales with noise too
    for v in g.nodes():
        if v == target:
            continue
        colour = COMM_BLUE if v < n1 else COMM_GREEN
        s = 90 + 260 * noise[v]
        lw = 0.6 + 1.4 * noise[v]
        ax.scatter(
            pos[v][0],
            pos[v][1],
            s=s,
            c=colour,
            edgecolors=NOISE_EDGE,
            linewidths=lw,
            zorder=3,
        )

    # --- 4) target node (orange halo + dot) ------------------------
    tx, ty = pos[target]
    halo = mpatches.Circle(
        (tx, ty), 0.28, facecolor=HALO_ORANGE, edgecolor="none",
        alpha=0.55, zorder=3.4,
    )
    ax.add_patch(halo)
    ax.scatter(
        tx, ty,
        s=320,
        c=TARGET_ORANGE,
        edgecolors="white",
        linewidths=1.6,
        zorder=4,
    )
    ax.annotate(
        "target user\n(estimate ITE)",
        xy=(tx, ty + 0.16),
        xytext=(tx, ty + 1.05),
        fontsize=9.0,
        color=TARGET_ORANGE,
        ha="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=TARGET_ORANGE, lw=1.0, shrinkA=0, shrinkB=6),
    )


def draw_dual_root_baseline(ax):
    """The explanatory baseline below the graph: Z* feeding two error roots."""
    # Coordinates on the axes (same axes as the graph to keep everything in
    # one canvas).
    y = -2.05
    # Latent node
    z_xy = (0.0, y)
    latent = mpatches.Circle(z_xy, 0.22, facecolor="white",
                              edgecolor=LATENT_GREY, linewidth=1.0, zorder=5)
    ax.add_patch(latent)
    ax.text(0.0, y, r"$Z^*$", ha="center", va="center",
            fontsize=10, color=LATENT_GREY, zorder=6)
    ax.text(0.0, y - 0.38, "network\nposition",
            ha="center", va="top", fontsize=7.5,
            color=TEXT_GREY, style="italic")

    # Left chip — bias root
    left_cx = -2.55
    chip_w, chip_h = 2.4, 0.9
    left_rect = mpatches.FancyBboxPatch(
        (left_cx - chip_w / 2, y - chip_h / 2), chip_w, chip_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.9, edgecolor=BIAS_ACCENT, facecolor=BIAS_ACCENT, alpha=0.12,
        zorder=4,
    )
    ax.add_patch(left_rect)
    ax.text(left_cx, y + 0.23, "bias root", fontsize=9.2,
            ha="center", fontweight="bold", color=BIAS_ACCENT)
    ax.text(left_cx, y + 0.02,
            r"$Z^* \to T,\;\; Z^* \to Y(t)$",
            fontsize=8.4, ha="center", color=TEXT_DARK)
    ax.text(left_cx, y - 0.22,
            "community position shifts treatment",
            fontsize=7.4, ha="center", color=TEXT_GREY, style="italic")

    # Right chip — variance root
    right_cx = 2.55
    right_rect = mpatches.FancyBboxPatch(
        (right_cx - chip_w / 2, y - chip_h / 2), chip_w, chip_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.9, edgecolor=VAR_ACCENT, facecolor=VAR_ACCENT, alpha=0.10,
        zorder=4,
    )
    ax.add_patch(right_rect)
    ax.text(right_cx, y + 0.23, "variance root", fontsize=9.2,
            ha="center", fontweight="bold", color=VAR_ACCENT)
    ax.text(right_cx, y + 0.02, r"$Z^* \to \sigma^2(Y)$",
            fontsize=8.4, ha="center", color=TEXT_DARK)
    ax.text(right_cx, y - 0.22, "position shifts outcome noise",
            fontsize=7.4, ha="center", color=TEXT_GREY, style="italic")

    # Arrows from Z* to each chip
    ax.annotate("", xy=(left_cx + chip_w / 2, y), xytext=(z_xy[0] - 0.22, y),
                arrowprops=dict(arrowstyle="-|>", color=BIAS_ACCENT, lw=1.2,
                                shrinkA=0, shrinkB=2))
    ax.annotate("", xy=(right_cx - chip_w / 2, y), xytext=(z_xy[0] + 0.22, y),
                arrowprops=dict(arrowstyle="-|>", color=VAR_ACCENT, lw=1.2,
                                shrinkA=0, shrinkB=2))

    # Arrows from the graph (top of canvas) to Z*, rendering the idea that
    # the *visual* phenomena above are instances of the two roots below.
    ax.annotate("", xy=(-0.15, y + 0.4), xytext=(-1.7, -1.2),
                arrowprops=dict(arrowstyle="-|>", color=BIAS_ACCENT,
                                lw=0.8, linestyle="dashed",
                                shrinkA=0, shrinkB=4, alpha=0.75))
    ax.annotate("", xy=(0.15, y + 0.4), xytext=(1.7, -1.2),
                arrowprops=dict(arrowstyle="-|>", color=VAR_ACCENT,
                                lw=0.8, linestyle="dashed",
                                shrinkA=0, shrinkB=4, alpha=0.75))


def draw_legend(ax, y):
    chip_specs = [
        (COMM_BLUE, "community 1", None),
        (COMM_GREEN, "community 2", None),
        (TARGET_ORANGE, "target", None),
        (None, r"node size $\propto \sigma^2(Z_i)$", "sigma"),
    ]
    chip_x = -3.2
    for colour, label, marker in chip_specs:
        if marker == "sigma":
            # draw two nested circles to show the size gradient
            ax.scatter(chip_x - 0.12, y, s=80, c="#CFCFCF",
                       edgecolors=NOISE_EDGE, linewidths=0.8, zorder=5)
            ax.scatter(chip_x + 0.15, y, s=240, c="#CFCFCF",
                       edgecolors=NOISE_EDGE, linewidths=1.4, zorder=5)
            ax.text(chip_x + 0.42, y, label, fontsize=8.0, va="center",
                    color=TEXT_GREY)
        else:
            ax.scatter(chip_x, y, s=120, c=colour, edgecolors="white",
                       linewidths=1.0, zorder=5)
            ax.text(chip_x + 0.2, y, label, fontsize=8.0, va="center",
                    color=TEXT_GREY)
        chip_x += 2.15


def main() -> None:
    rng = np.random.default_rng(SEED)
    g, pos, noise, target, n1 = build_graph(rng)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_graph(ax, g, pos, noise, target, n1)
    draw_dual_root_baseline(ax)
    draw_legend(ax, y=-3.2)

    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-3.6, 2.4)
    ax.axis("off")

    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
