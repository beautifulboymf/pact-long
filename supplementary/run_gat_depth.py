#!/usr/bin/env python3
"""GAT depth sensitivity experiment for ICML reviewer response.

Measures PEHE as a function of GAT depth L = {1, 2, 3, 5, 7} on DBLP
with TARNet at rho=10. Uses cached DGP data for reproducibility.

Runs 3 seeds per depth and reports mean +/- std.

Usage (on GPU server):
    conda activate cl
    cd /root/autodl-tmp/Uplift
    python -m supplementary.run_gat_depth --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, "/root/autodl-tmp/Uplift")

from cavin.main import _load_cached_sample, _load_yaml
from cavin.model import CAVINConfig, TARNetGraph
from cavin.train import TrainConfig, train_graph
from cavin.losses import LossWeights


# ---------- Configuration ----------

# gnn_hidden tuples for each depth L.
# These follow standard practice: start wider, taper to 128.
DEPTH_CONFIGS = {
    1: [128],
    2: [256, 128],
    3: [256, 128, 128],
    5: [256, 128, 128, 128, 128],      # default from paper
    7: [256, 256, 128, 128, 128, 128, 128],
}

SEEDS = [0, 1, 2]
CACHE_DIR = "/root/autodl-tmp/Uplift/runs/data_cache"
RESULTS_DIR = "/root/autodl-tmp/Uplift/runs/gat_depth"


def run_one(depth: int, seed: int, device: str) -> dict:
    """Train TARNet+GPE with `depth` GAT layers on DBLP (rho=10)."""
    cache_path = os.path.join(CACHE_DIR, f"dblp_rho10_seed{seed}.pt")
    assert os.path.exists(cache_path), f"Missing cache: {cache_path}"
    sample = _load_cached_sample(cache_path, device)

    gnn_hidden = tuple(DEPTH_CONFIGS[depth])
    model_cfg = CAVINConfig(
        in_dim=sample.X.size(1),
        pos_dim=128,
        fusion_embed_dim=256,
        fusion_heads=4,
        use_gpe=True,
        backbone="gat",
        gnn_hidden=gnn_hidden,
        gnn_heads=4,
        mlp_hidden=(128, 128, 128, 128),
        dropout=0.0,
        use_variance=False,  # TARNet does not use variance weighting
    )
    train_cfg = TrainConfig(
        epochs=200,
        lr=0.001,
        weight_decay=1e-4,
        train_frac=0.6,
        val_frac=0.2,
        learner="tarnet",
        loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
        use_uncertainty_weighting=False,
        delta=0.01,
        select_metric="pehe",  # select by PEHE (lower is better) for this experiment
        normalize_y=True,
        log_every=10,
        seed=seed,
    )

    model = TARNetGraph(model_cfg)
    result = train_graph(sample, model_cfg, train_cfg, device=device, model=model)
    best = result["best"]
    return {
        "depth": depth,
        "seed": seed,
        "epoch": best["epoch"],
        "test_pehe": best["test"]["pehe"],
        "test_qini": best["test"]["qini"],
        "val_pehe": best["val"]["pehe"],
        "val_qini": best["val"]["qini"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEPTH_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    for depth in args.depths:
        assert depth in DEPTH_CONFIGS, f"Unsupported depth {depth}"
        for seed in args.seeds:
            tag = f"L{depth}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"  GAT depth={depth}, seed={seed}, gnn_hidden={DEPTH_CONFIGS[depth]}")
            print(f"{'='*60}")
            t0 = time.time()
            res = run_one(depth, seed, args.device)
            elapsed = time.time() - t0
            res["elapsed_s"] = elapsed
            all_results.append(res)
            print(f"  -> test_pehe={res['test_pehe']:.4f}  test_qini={res['test_qini']:.4f}  ({elapsed:.0f}s)")

            # Save individual result
            out_path = os.path.join(RESULTS_DIR, f"{tag}.json")
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n\n" + "="*70)
    print("  GAT Depth Sensitivity — TARNet+GPE on DBLP (rho=10)")
    print("="*70)
    print(f"{'Depth':>6} | {'Test PEHE':>20} | {'Test Qini':>20}")
    print("-"*70)
    for depth in sorted(DEPTH_CONFIGS.keys()):
        depth_results = [r for r in all_results if r["depth"] == depth]
        if not depth_results:
            continue
        pehe_vals = [r["test_pehe"] for r in depth_results]
        qini_vals = [r["test_qini"] for r in depth_results]
        pehe_mean, pehe_std = np.mean(pehe_vals), np.std(pehe_vals)
        qini_mean, qini_std = np.mean(qini_vals), np.std(qini_vals)
        print(f"  L={depth:<3} | {pehe_mean:8.4f} +/- {pehe_std:.4f} | {qini_mean:8.4f} +/- {qini_std:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
