"""Render Figure 1 (cover illustration) for the PACT paper.

Produces `images/Uplift_cover.pdf` — a small social graph with two
visible communities, a highlighted target node, and node radii scaled
by heteroscedastic outcome-noise variance. Wire the visual to the
dual-root thesis: same positional signal (community membership) drives
both confounding bias and outcome-noise variance.

Run from the repository root:
    python images/scripts/make_cover.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

SEED = 7
OUT_DIR = Path(__file__).resolve().parents[1]  # pact-long/images
OUT_PDF = OUT_DIR / "Uplift_cover.pdf"
OUT_PNG = OUT_DIR / "Uplift_cover_preview.png"

COMM_BLUE = "#4C72B0"
COMM_GREEN = "#55A868"
TARGET_ORANGE = "#E08E44"
HALO_ORANGE = "#F2B97C"
EDGE_GREY = "#9E9E9E"
TEXT_GREY = "#333333"
NOISE_SHADE = "#D6D6D6"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "axes.linewidth": 0.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def build_graph(rng: np.random.Generator) -> tuple[nx.Graph, dict[int, tuple[float, float]], np.ndarray, int]:
    """Two-community stochastic block model + target node."""
    n1, n2 = 12, 12
    p_in, p_out = 0.45, 0.04
    g = nx.stochastic_block_model(
        [n1, n2], [[p_in, p_out], [p_out, p_in]], seed=SEED
    )
    pos = nx.spring_layout(g, seed=SEED, k=0.9, iterations=200)

    # Put the two communities on left/right halves, spread wider.
    for node in range(n1):
        x, y = pos[node]
        pos[node] = (x * 0.9 - 2.0, y * 0.9)
    for node in range(n1, n1 + n2):
        x, y = pos[node]
        pos[node] = (x * 0.9 + 2.0, y * 0.9)

    # Heteroscedastic noise: higher near community boundaries and peripheries.
    coords = np.array([pos[v] for v in g.nodes()])
    centre_left = coords[:n1].mean(axis=0)
    centre_right = coords[n1:].mean(axis=0)
    d_left = np.linalg.norm(coords - centre_left, axis=1)
    d_right = np.linalg.norm(coords - centre_right, axis=1)
    # Distance to the nearer community centre, normalised; plus a small jitter.
    proximity = np.minimum(d_left, d_right)
    noise = proximity / proximity.max()
    noise = 0.3 + 0.7 * noise  # map to [0.3, 1.0]
    noise += rng.normal(0, 0.05, size=noise.shape)
    noise = np.clip(noise, 0.25, 1.1)

    # Choose the target: a bridge-like node, then move it to the midpoint
    # between the two communities so it is visually in neither camp.
    bc = nx.betweenness_centrality(g)
    target = max(bc, key=bc.get)
    pos[target] = (0.0, 0.15)
    return g, pos, noise, target


def draw(ax, g, pos, noise, target):
    n1 = 12

    # 1) Soft background halos for the two communities (wider + more saturated).
    for idx, (colour, members) in enumerate(
        [(COMM_BLUE, range(n1)), (COMM_GREEN, range(n1, 2 * n1))]
    ):
        pts = np.array([pos[m] for m in members])
        centre = pts.mean(axis=0)
        radius = 1.55
        halo = mpatches.Circle(
            centre,
            radius,
            facecolor=colour,
            edgecolor=colour,
            alpha=0.14,
            linewidth=0.0,
            zorder=0,
        )
        ax.add_patch(halo)

    # 2) Edges.
    for u, v in g.edges():
        xs = [pos[u][0], pos[v][0]]
        ys = [pos[u][1], pos[v][1]]
        is_target_edge = target in (u, v)
        ax.plot(
            xs,
            ys,
            color=EDGE_GREY,
            linewidth=1.3 if is_target_edge else 0.7,
            alpha=0.85 if is_target_edge else 0.32,
            zorder=1,
        )

    # 3) Noise-shading rings (grey halos whose radius scales with sigma^2(Z_i)).
    #    Muted so they visually sit *behind* the node colour instead of on top.
    for v in g.nodes():
        if v == target:
            continue
        x, y = pos[v]
        r = 0.12 + 0.18 * noise[v]
        ring = mpatches.Circle(
            (x, y),
            r,
            facecolor=NOISE_SHADE,
            edgecolor="none",
            alpha=0.28,
            zorder=1.5,
        )
        ax.add_patch(ring)

    # 4) Nodes.
    for v in g.nodes():
        x, y = pos[v]
        colour = COMM_BLUE if v < n1 else COMM_GREEN
        if v == target:
            continue
        ax.scatter(
            x,
            y,
            s=220,
            c=colour,
            edgecolors="white",
            linewidth=1.2,
            zorder=3,
        )

    # 5) Target node with orange halo on top.
    tx, ty = pos[target]
    halo = mpatches.Circle(
        (tx, ty),
        0.36,
        facecolor=HALO_ORANGE,
        edgecolor="none",
        alpha=0.55,
        zorder=3.5,
    )
    ax.add_patch(halo)
    ax.scatter(
        tx,
        ty,
        s=320,
        c=TARGET_ORANGE,
        edgecolors="white",
        linewidth=1.6,
        zorder=4,
    )

    # 6) Callouts — aligned with the phenomenon each points at.
    ax.annotate(
        "community membership\n"
        r"confounds $T$ and $Y(t)$"
        "\n(bias source)",
        xy=(-2.0, 0.4),
        xytext=(-3.5, 1.9),
        fontsize=9.0,
        color=COMM_BLUE,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=COMM_BLUE, lw=0.9, shrinkA=0, shrinkB=6),
    )
    ax.annotate(
        r"position drives noise $\sigma^2(Z_i)$"
        "\n"
        r"(variance source; ring radius $\propto \sigma^2$)",
        xy=(2.1, -0.6),
        xytext=(0.4, -2.3),
        fontsize=9.0,
        color=COMM_GREEN,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=COMM_GREEN, lw=0.9, shrinkA=0, shrinkB=6),
    )
    ax.annotate(
        "target user\n(estimate ITE)",
        xy=(tx, ty + 0.22),
        xytext=(tx + 0.05, ty + 1.25),
        fontsize=9.0,
        color=TARGET_ORANGE,
        ha="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=TARGET_ORANGE, lw=1.0, shrinkA=0, shrinkB=6),
    )

    # 7) Legend chips — adequately spaced.
    chip_y = -3.0
    chip_specs = [
        (COMM_BLUE, "community 1"),
        (COMM_GREEN, "community 2"),
        (TARGET_ORANGE, "target"),
        (NOISE_SHADE, r"outcome noise $\sigma^2(Z_i)$"),
    ]
    chip_x = -3.6
    for colour, label in chip_specs:
        ax.scatter(chip_x, chip_y, s=110, c=colour, edgecolors="white", linewidth=1.0, zorder=5)
        ax.text(chip_x + 0.2, chip_y, label, fontsize=8.5, va="center", color=TEXT_GREY)
        chip_x += 2.1

    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-3.3, 2.3)
    ax.axis("off")


def main() -> None:
    rng = np.random.default_rng(SEED)
    g, pos, noise, target = build_graph(rng)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    draw(ax, g, pos, noise, target)
    fig.savefig(OUT_PDF, format="pdf", transparent=False)
    fig.savefig(OUT_PNG, format="png", dpi=220, transparent=False)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(OUT_DIR.parent)}")
    print(f"wrote {OUT_PNG.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
