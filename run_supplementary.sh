#!/bin/bash
# Supplementary experiments for ICML reviewer response.
# Run on remote GPU server (ssh autodl).
#
# Experiment 1: GAT Depth Sensitivity (L=1,2,3,5,7, TARNet on DBLP rho=10)
# Experiment 2: t-SNE Visualization (CAVIN with/without GPE on DBLP)
#
# Prerequisites:
#   - SSH alias 'autodl' configured in ~/.ssh/config
#   - Conda env 'cl' with PyTorch, PyG, scikit-learn, matplotlib
#   - Cached DGP data in /root/autodl-tmp/Uplift/runs/data_cache/
#   - Scripts uploaded to /root/autodl-tmp/Uplift/supplementary/
#
# Usage:
#   bash run_supplementary.sh           # runs both experiments
#   bash run_supplementary.sh depth     # runs GAT depth only
#   bash run_supplementary.sh tsne      # runs t-SNE only

set -euo pipefail

REMOTE_CMD="source /root/miniconda3/etc/profile.d/conda.sh && conda activate cl && cd /root/autodl-tmp/Uplift"

run_depth() {
    echo "=== Running GAT Depth Sensitivity Experiment ==="
    ssh autodl "${REMOTE_CMD} && python -m supplementary.run_gat_depth --device cuda"
}

run_tsne() {
    echo "=== Running t-SNE Visualization Experiment ==="
    ssh autodl "${REMOTE_CMD} && python -m supplementary.run_tsne --device cuda"
    echo "=== Downloading t-SNE figure ==="
    ssh autodl 'cat /root/autodl-tmp/Uplift/runs/tsne/tsne_gpe.png' \
        > "$(dirname "$0")/images/tsne_gpe.png"
    echo "Saved to images/tsne_gpe.png"
}

case "${1:-all}" in
    depth) run_depth ;;
    tsne)  run_tsne ;;
    all)   run_depth && run_tsne ;;
    *)     echo "Usage: $0 [depth|tsne|all]"; exit 1 ;;
esac
