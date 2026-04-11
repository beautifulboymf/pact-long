#!/usr/bin/env python3
"""Regenerate t-SNE figure from saved embeddings (fix for n_iter -> max_iter)."""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, "/root/autodl-tmp/Uplift")
from cavin.dgp import detect_communities

RESULTS_DIR = "/root/autodl-tmp/Uplift/runs/tsne"


def make_tsne_figure(rep_gpe, rep_nogpe, labels, save_path):
    """Generate a side-by-side t-SNE figure."""
    unique, counts = np.unique(labels, return_counts=True)
    top_k = 8
    top_communities = unique[np.argsort(-counts)[:top_k]]
    labels_vis = labels.copy()
    labels_vis[~np.isin(labels, top_communities)] = -1

    n = rep_gpe.shape[0]
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=10000, replace=False)
        rep_gpe_sub = rep_gpe[idx]
        rep_nogpe_sub = rep_nogpe[idx]
        labels_sub = labels_vis[idx]
    else:
        rep_gpe_sub = rep_gpe
        rep_nogpe_sub = rep_nogpe
        labels_sub = labels_vis

    print("Running t-SNE for GPE model...")
    tsne_gpe = TSNE(
        n_components=2, perplexity=30, random_state=42,
        max_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(rep_gpe_sub)

    print("Running t-SNE for No-GPE model...")
    tsne_nogpe = TSNE(
        n_components=2, perplexity=30, random_state=42,
        max_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(rep_nogpe_sub)

    unique_labels = np.unique(labels_sub)
    cmap = plt.cm.get_cmap("Set1", len(unique_labels))
    color_map = {}
    ci = 0
    for lbl in sorted(unique_labels):
        if lbl == -1:
            color_map[lbl] = (0.8, 0.8, 0.8, 0.3)
        else:
            color_map[lbl] = (*cmap(ci)[:3], 0.6)
            ci += 1
    colors = np.array([color_map[l] for l in labels_sub])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order_bg_first = np.argsort(-(labels_sub == -1).astype(int))

    for ax, emb, title in [
        (axes[0], tsne_nogpe, "Without GPE"),
        (axes[1], tsne_gpe, "With GPE"),
    ]:
        ax.scatter(
            emb[order_bg_first, 0], emb[order_bg_first, 1],
            c=colors[order_bg_first], s=4, edgecolors="none", rasterized=True,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = []
    for lbl in sorted(unique_labels):
        if lbl == -1:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=(0.8, 0.8, 0.8), markersize=6, label="Other")
            )
        else:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map[lbl][:3], markersize=6,
                       label=f"Community {lbl}")
            )
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=min(len(legend_elements), 5),
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle("t-SNE of Node Representations on DBLP", fontsize=15, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved t-SNE figure to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    data = np.load(os.path.join(RESULTS_DIR, "embeddings.npz"))
    rep_gpe = data["rep_gpe"]
    rep_nogpe = data["rep_nogpe"]
    labels = data["labels"]
    print(f"Loaded embeddings: GPE={rep_gpe.shape}, NoGPE={rep_nogpe.shape}, labels={labels.shape}")

    save_path = os.path.join(RESULTS_DIR, "tsne_gpe.png")
    make_tsne_figure(rep_gpe, rep_nogpe, labels, save_path)
