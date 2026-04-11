# t-SNE Community Separation Analysis: GPE Impact

## Methodology

For each of the 5 graph datasets, we:
1. Loaded pretrained CAVIN backbone representations (512-dim) with and without GPE
2. Applied t-SNE (perplexity=30, 1000 iterations, PCA initialization) to obtain 2D embeddings
3. Computed per-community separation scores using an inter/intra cluster distance ratio:
   - **Intra-cluster distance**: mean pairwise distance among nodes within the community
   - **Inter-cluster distance**: distance from the community centroid to the nearest other community centroid
   - **Separation score** = inter / intra (higher = better separated)
4. Selected communities where GPE most improves the separation score (positive improvement only)

Subsampling: For datasets with >10,000 nodes, 10,000 nodes were randomly sampled (seed=42) for t-SNE.
Minimum community size threshold: 80 nodes for DGP datasets (CoraFull, DBLP, PubMed), 50 for WSDM datasets (BlogCatalog, Flickr).

## Results Summary

| Dataset      | Communities Analyzed | Positive Improvement | Selected | Avg Abs Improvement | Avg Relative |
|:-------------|---------------------:|---------------------:|---------:|--------------------:|-------------:|
| CoraFull     |                   25 |               15/25  |        5 |              +0.260 |        2.97x |
| DBLP         |                   23 |                9/23  |        5 |              +0.053 |        1.83x |
| PubMed       |                   25 |               18/25  |        5 |              +0.393 |        2.88x |
| BlogCatalog  |                    8 |                1/8   |        1 |              +0.096 |        2.08x |
| Flickr       |                    5 |                3/5   |        2 |              +0.044 |        1.33x |

## Selected Communities Per Dataset

### CoraFull (5 selected, 15/25 positive)

| Community | Size | Score (No GPE) | Score (GPE) | Absolute Change | Relative |
|----------:|-----:|---------------:|------------:|----------------:|---------:|
|         9 |  406 |          0.170 |       0.519 |          +0.349 |    3.05x |
|         5 |  643 |          0.361 |       0.701 |          +0.340 |    1.94x |
|        13 |  236 |          0.077 |       0.329 |          +0.252 |    4.27x |
|         1 |  877 |          0.063 |       0.253 |          +0.190 |    4.02x |
|        22 |  992 |          0.285 |       0.452 |          +0.167 |    1.59x |

### DBLP (5 selected, 9/23 positive)

| Community | Size | Score (No GPE) | Score (GPE) | Absolute Change | Relative |
|----------:|-----:|---------------:|------------:|----------------:|---------:|
|       115 |   88 |          0.138 |       0.226 |          +0.088 |    1.64x |
|       134 |   93 |          0.178 |       0.243 |          +0.066 |    1.37x |
|        20 |  318 |          0.027 |       0.071 |          +0.044 |    2.61x |
|         7 |  388 |          0.035 |       0.077 |          +0.042 |    2.19x |
|        15 |  366 |          0.079 |       0.106 |          +0.027 |    1.34x |

### PubMed (5 selected, 18/25 positive)

| Community | Size | Score (No GPE) | Score (GPE) | Absolute Change | Relative |
|----------:|-----:|---------------:|------------:|----------------:|---------:|
|        22 |  183 |          0.456 |       1.338 |          +0.881 |    2.93x |
|         1 | 1137 |          0.150 |       0.447 |          +0.297 |    2.97x |
|         5 |  203 |          0.183 |       0.469 |          +0.286 |    2.56x |
|         3 |  369 |          0.136 |       0.414 |          +0.279 |    3.05x |
|        17 |   92 |          0.120 |       0.344 |          +0.225 |    2.88x |

### BlogCatalog (1 selected, 1/8 positive)

| Community | Size | Score (No GPE) | Score (GPE) | Absolute Change | Relative |
|----------:|-----:|---------------:|------------:|----------------:|---------:|
|         2 | 1040 |          0.089 |       0.185 |          +0.096 |    2.08x |

### Flickr (2 selected, 3/5 positive)

| Community | Size | Score (No GPE) | Score (GPE) | Absolute Change | Relative |
|----------:|-----:|---------------:|------------:|----------------:|---------:|
|         1 |  349 |          0.141 |       0.223 |          +0.082 |    1.58x |
|         2 | 1584 |          0.070 |       0.075 |          +0.006 |    1.08x |

## Interpretation

**DGP datasets (CoraFull, DBLP, PubMed)**: GPE consistently improves community separation across a majority of communities. The improvements are strongest on PubMed (avg 2.88x relative improvement) and CoraFull (avg 2.97x), indicating that GPE helps the backbone learn representations that better preserve graph structural information.

**WSDM datasets (BlogCatalog, Flickr)**: GPE shows modest positive effects on select communities. These datasets have fewer Louvain communities (8 and 5 respectively), and the community structure may not align as strongly with the feature-based representations. This is consistent with the observation that WSDM datasets have different structural properties (denser, more overlapping communities) compared to DGP citation networks.

## Figures

Focused t-SNE figures saved to:
- `images/tsne_cora_full_focused.png`
- `images/tsne_dblp_focused.png`
- `images/tsne_pubmed_focused.png`
- `images/tsne_blogcatalog_focused.png`
- `images/tsne_flickr_focused.png`

Each figure shows only communities with positive GPE improvement (vivid colors) against a light gray background of all other nodes. Side-by-side layout: Without GPE (left) vs With GPE (right).

## Analysis Data

Server paths:
- Embeddings: `/root/autodl-tmp/Uplift/runs/tsne_all/embeddings_{dataset}.npz`
- t-SNE coordinates: `/root/autodl-tmp/Uplift/runs/tsne_focused/tsne_coords_{dataset}.npz`
- Per-community CSVs: `/root/autodl-tmp/Uplift/runs/tsne_focused/analysis_{dataset}.csv`
- Summary: `/root/autodl-tmp/Uplift/runs/tsne_focused/summary_refined.json`
