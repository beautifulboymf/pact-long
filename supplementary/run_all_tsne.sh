#!/usr/bin/env bash
# Run t-SNE visualizations for all 5 graph datasets on the autodl server.
# Usage: bash supplementary/run_all_tsne.sh
set -euo pipefail

echo "=== Uploading script to server ==="
scp /Users/fanruochen/Desktop/CC/overleaf-tmp/supplementary/run_all_tsne.py \
    autodl:/root/autodl-tmp/Uplift/supplementary/run_all_tsne.py

echo "=== Running t-SNE generation on server ==="
ssh autodl 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate cl && \
    cd /root/autodl-tmp/Uplift && \
    python supplementary/run_all_tsne.py --device cuda 2>&1' | tee /tmp/tsne_all_log.txt

echo "=== Downloading figures ==="
DEST="/Users/fanruochen/Desktop/CC/overleaf-tmp/images"
SRC="autodl:/root/autodl-tmp/Uplift/runs/tsne_all"

for ds in cora_full dblp pubmed blogcatalog flickr; do
    echo "  Downloading tsne_${ds}.png ..."
    scp "${SRC}/tsne_${ds}.png" "${DEST}/tsne_${ds}.png" 2>/dev/null || \
        echo "  WARNING: tsne_${ds}.png not found on server"
done

echo "=== Done ==="
echo "Figures saved to: ${DEST}/"
ls -la "${DEST}"/tsne_*.png 2>/dev/null || echo "No figures found."
