"""Analyze correlation between node degree and ITE estimation noise.
Validates Path 3 of the Dual-Root narrative: network position causes heteroscedastic noise.

Run on server: python supplementary/degree_noise_analysis.py
"""
import sys, os, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, '/root/autodl-tmp/Uplift')

def analyze_dataset(ds_name, cfg_path, cache_path, dgp_cfg, seed=0):
    """Compute degree vs residual and degree vs true noise for one dataset."""
    from cavin.model import CAVINConfig, CAVIN
    from cavin.train import make_splits
    from cavin.dgp import semi_synthetic, normalize_adjacency

    # Load cached data
    d = torch.load(cache_path, map_location='cpu', weights_only=False)
    X = d['X']
    edge_index = d['edge_index']
    T = d['T']
    Y = d['Y']
    Y0 = d['Y0']
    Y1 = d['Y1']
    true_tau = d['true_tau']

    A_norm_indices = d['A_norm_indices']
    A_norm_values = d['A_norm_values']
    A_norm_shape = list(d['A_norm_shape'])
    A_norm = torch.sparse_coo_tensor(A_norm_indices, A_norm_values, A_norm_shape).coalesce()

    n = X.size(0)

    # Compute node degree from edge_index
    src = edge_index[0].numpy()
    degree = np.bincount(src, minlength=n).astype(float)

    # Recompute true noise sigma from DGP
    # Load raw graph data to run DGP
    from cavin.data import load_graph_dataset
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    ds_cfg = cfg['dataset']

    X_raw, A_raw = load_graph_dataset(ds_cfg['name'], ds_cfg['root'])
    rng = np.random.default_rng(seed)
    out = semi_synthetic(X_raw, A_raw, dgp_cfg, rng=rng)
    sigma = out['sigma']  # per-node noise std

    # Train a TARNet (no GPE) to get predictions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_cfg = CAVINConfig(
        in_dim=X.size(1), pos_dim=16, fusion_embed_dim=32, fusion_heads=4,
        use_gpe=False, backbone='gat',
        gnn_hidden=(256, 128, 128, 128, 128), gnn_heads=4,
        mlp_hidden=(128, 128, 128, 128), dropout=0.0, use_variance=False,
    )
    model = CAVIN(model_cfg).to(device)

    Xd = X.to(device)
    ed = edge_index.to(device)
    Td = T.to(device)
    Yd = Y.to(device)
    Ad = A_norm.to(device)

    # Normalize features
    Xm = Xd.mean(0, keepdim=True)
    Xs = Xd.std(0, keepdim=True) + 1e-6
    Xd = (Xd - Xm) / Xs

    train_idx, val_idx, test_idx = make_splits(n, 0.6, 0.2, seed=seed)
    train_t = torch.tensor(train_idx, device=device, dtype=torch.long)

    # Normalize Y
    ym = float(Yd[train_t].mean())
    ys = float(Yd[train_t].std() + 1e-6)
    Y_n = (Yd - ym) / ys

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = torch.nn.MSELoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(Xd, ed)
        y_pred = torch.where(Td > 0, out.mu1, out.mu0)
        loss = mse(y_pred[train_t], Y_n[train_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    # Get ITE predictions
    model.eval()
    with torch.no_grad():
        out = model(Xd, ed)
        tau_pred = (out.mu1 - out.mu0).cpu().numpy() * ys

    residuals = np.abs(tau_pred - true_tau.numpy())

    return degree, residuals, sigma

def make_figure(results, save_path):
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (ds_name, degree, residuals, sigma) in zip(axes, results):
        # Bin by degree
        max_deg = int(np.percentile(degree, 99))
        bins = np.linspace(0, max_deg, 30)
        bin_idx = np.digitize(degree, bins)

        bin_means_res = []
        bin_means_sig = []
        bin_centers = []
        for b in range(1, len(bins)):
            mask = bin_idx == b
            if mask.sum() > 10:
                bin_centers.append((bins[b-1] + bins[b]) / 2)
                bin_means_res.append(np.mean(residuals[mask]))
                bin_means_sig.append(np.mean(sigma[mask]))

        # Scatter: degree vs |residual|
        ax.scatter(degree, residuals, alpha=0.05, s=3, c='steelblue', rasterized=True)
        ax.plot(bin_centers, bin_means_res, 'r-', linewidth=2.5, label='Binned mean')

        # Spearman
        rho_res, p_res = spearmanr(degree, residuals)
        ax.set_xlabel('Node Degree', fontsize=12)
        ax.set_ylabel('|ITE Residual|', fontsize=12)
        ax.set_title(f'{ds_name}\nSpearman ρ={rho_res:.3f} (p={p_res:.1e})', fontsize=13)
        ax.set_xlim(0, max_deg)
        ax.legend(fontsize=10)

        # Also compute degree vs sigma correlation
        rho_sig, p_sig = spearmanr(degree, sigma)
        print(f"{ds_name}: degree vs |residual| Spearman ρ={rho_res:.3f} (p={p_res:.2e})")
        print(f"{ds_name}: degree vs true σ    Spearman ρ={rho_sig:.3f} (p={p_sig:.2e})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    results = []

    for ds_name, cfg_path, cache_name in [
        ('DBLP', 'cavin/configs/server/dblp.yaml', 'dblp_rho10_seed0.pt'),
        ('CoraFull', 'cavin/configs/server/cora_full.yaml', 'cora_full_rho10_seed0.pt'),
    ]:
        import yaml
        cfg = yaml.safe_load(open(cfg_path))
        dgp_cfg = cfg['dgp']
        cache_path = f'runs/data_cache/{cache_name}'

        print(f"\n=== Analyzing {ds_name} ===")
        degree, residuals, sigma = analyze_dataset(ds_name, cfg_path, cache_path, dgp_cfg)
        results.append((ds_name, degree, residuals, sigma))

    make_figure(results, 'images/degree_vs_noise.png')
