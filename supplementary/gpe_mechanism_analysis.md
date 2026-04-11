# GPE Mechanism Analysis: Why Does Graph Positional Encoding Improve ITE Estimation?

## 1. Problem Statement

GPE (Graph Positional Encoding) improves ITE estimation on 7/8 graph models, but we lack
a rigorous mechanistic explanation for *why* it works. The current paper claims "community
separation" based on t-SNE visualizations (Figure 5), but:

- t-SNE is a stochastic dimensionality reduction that does not preserve distances faithfully
- The visual difference between "with GPE" and "without GPE" is subjective
- DBLP shows only 9/23 communities with positive improvement (39%), which is close to random
- A reviewer can (and will) dismiss t-SNE plots as confirmation bias

We need **quantitative, falsifiable evidence** for the causal mechanism through which GPE
reduces PEHE. This document proposes five hypotheses, evaluates their testability, and designs
concrete experiments for the top three.

---

## 2. Literature: How Do Other Papers Explain Why PE Helps in GNNs?

### 2.1 Expressiveness / WL-Distinguishability (Theoretical)

**Graphormer (Ying et al., NeurIPS 2021)** argues that positional encodings (centrality,
spatial distance, edge features as attention bias) increase the expressive power of the
architecture *beyond* the 1-WL test. Without PE, message-passing GNNs cannot distinguish
structurally equivalent nodes in different global positions. Graphormer proves that with
sufficient positional information, the model becomes a universal approximator of functions
on graphs.

**SAN (Kreuzer et al., NeurIPS 2021)** uses Laplacian eigenvectors as spectral PE with
eigenvalue-reweighted attention and proves that the resulting learned PE is *universally
expressive* -- it can represent any function from graphs to labels, given sufficient capacity.

**Key insight for us:** These papers justify PE on *expressiveness* grounds -- the model can
now represent functions it literally could not before. This is a structural argument about
model capacity, not about debiasing or noise reduction.

### 2.2 Modular Position + Structure (Architectural)

**GraphGPS (Rampasek et al., NeurIPS 2022)** provides a *recipe*: combine PE, local
message-passing (MPNN), and global attention (Transformer). The key finding is that PE
increases MPNN expressiveness, and this benefit *compounds* with global attention. GraphGPS
does not explain *why* PE helps for any specific downstream task -- it shows that PE is
a universally beneficial ingredient in graph learning.

**PEARL (ICLR 2025)** shows that message-passing GNNs function as nonlinear mappings of
eigenvectors, so GNN architectures themselves can generate powerful PEs. This further
supports the "PE adds expressiveness" view.

### 2.3 Proxy Confounder Adjustment (Causal)

**Veitch, Wang, Blei (NeurIPS 2019)** -- "Using Embeddings to Correct for Unobserved
Confounding in Networks" -- provides the most directly relevant theoretical framework.
Their key theorem: network embeddings (including node2vec) can serve as *proxies* for
unobserved confounders when the network structure carries information about latent
attributes that affect both treatment and outcome. Adjusting for embeddings in propensity
estimation yields valid causal estimates under conditions on embedding quality. This paper
directly supports our H1 (Confounding Debiasing).

**Sridhar & Getoor (2019)** extend this to relational empirical risk minimization,
showing network embeddings improve ATE estimation by adjusting for network-mediated
confounding.

### 2.4 Over-Smoothing Prevention (Representation Quality)

**Chen et al. (AAAI 2020)** -- "Measuring and Relieving the Over-Smoothing Problem" --
introduce the MAD (Mean Average Distance) metric and MADGap. They show that as GNN depth
increases, MAD approaches zero and node representations become indistinguishable. PE
provides a unique per-node "anchor" that can resist this convergence.

However, no prior work has empirically demonstrated that PE *specifically* prevents
over-smoothing in deep GNNs used for causal inference.

### 2.5 Interference Representation (Network Causal Inference)

**CauGramer (Wu et al., ICLR 2025)** uses a full graph Transformer with cross-attention
to learn *interference representations* -- aggregated peer information that captures
spillover effects. This is architecturally close to our GPE but applied differently:
CauGramer replaces the GNN entirely, while GPE augments an existing GNN.

### 2.6 Summary: What Evidence Do They Provide?

| Paper | Mechanism Claimed | Evidence Type |
|:------|:-----------------|:-------------|
| Graphormer | Expressiveness beyond 1-WL | Formal proof + ablation |
| SAN | Universal approximation via learned PE | Theorem + synthetic experiments |
| GraphGPS | Modular recipe compounds benefits | Extensive ablation across 11 benchmarks |
| Veitch et al. | Proxy confounder adjustment | Theorem + propensity score analysis + simulation |
| Chen et al. | Over-smoothing measurement | MAD metric + depth experiments |
| CauGramer | Interference representation | PEHE comparison on BlogCatalog/Flickr |

**Observation:** The strongest papers provide BOTH theoretical justification AND targeted
quantitative experiments isolating the mechanism. t-SNE alone has never been accepted as
sufficient evidence at a top venue.

---

## 3. Hypothesis Assessment

### H1: Confounding Debiasing

**Claim:** Network position is a hidden confounder (position affects both treatment
assignment and outcome). GPE makes this confounder explicitly available, allowing the
model to adjust for it.

**DGP Analysis:** This hypothesis is *directly supported by the DGP structure*:
- Treatment probability: `prob = sigmoid(p1 - p0)` where
  `p1 = k1 * Z_c1 + k2 * AhZ_c1 + k4 * c_bias` (community bias enters treatment)
- Potential outcomes: `f1 = p1 + beta_1 + k3 * AT + k4 * c_bias` (community bias
  enters outcomes)
- The `c_bias` term is a per-community random projection that affects BOTH treatment
  assignment and outcomes -- this is literally a community-level confounder
- The `AhZ_c1` term is a multi-hop aggregate that depends on network topology --
  capturing this requires positional information beyond raw features

**Theoretical backing:** Veitch et al. (NeurIPS 2019) prove that network embeddings
can proxy for exactly this type of latent confounder.

**Test:** Measure propensity score AUC with and without GPE.
- Train a propensity classifier (logistic regression on learned representations, or
  use the model's own propensity head `e(x)`) on the same representations
- Compute ROC-AUC of `e(x)` predicting `T` on held-out nodes
- If GPE improves propensity AUC: the positional encoding captures treatment-relevant
  information that raw features + GNN alone miss = confounding debiasing

**Success:** Propensity AUC increases significantly (e.g., 0.65 -> 0.75) with GPE
across all datasets. The improvement should correlate with PEHE improvement.

**Failure:** Propensity AUC is unchanged or decreases with GPE. This would mean GPE
is NOT helping capture the treatment assignment mechanism, and the PEHE improvement
must come from a different channel.

**Failure interpretation:** If H1 fails, GPE may be improving *outcome prediction*
rather than *confounding adjustment*. This would redirect to H3 or H5.

**Testable with existing infrastructure?** YES. The propensity head `e(x)` already
exists in all models. We just need to extract and evaluate it.

**Estimated runtime:** 1-2 hours. Re-run existing models, extract propensity
predictions, compute AUC. No new training needed if saved model checkpoints exist.

### H2: Representation Balance

**Claim:** GPE helps balance the learned representations between T=1 and T=0 groups,
reducing selection bias in the representation space.

**DGP Analysis:** The DGP generates confounded treatment -- nodes with certain features
and positions are more likely to be treated. This creates a distributional shift in
representation space between treated and control groups. If GPE enriches the representation
with position information, the model *could* learn representations where T=1 and T=0
groups become more similar (balanced) -- similar to what NetDeconf explicitly optimizes
for with Wasserstein distance.

However, there is a tension: GPE adds *more* information that distinguishes nodes.
If treatment is correlated with position (which it is, via `c_bias`), adding position
information could *increase* the T=1/T=0 gap in representation space. The question is
whether the downstream model then adjusts for this.

**Test:** Compute distributional distance between T=1 and T=0 representations.
- Extract representations from the backbone (after GNN, before prediction heads)
- Compute Wasserstein distance (already implemented in `baselines.py`) and/or MMD
  between `rep[T==1]` and `rep[T==0]`
- Compare with/without GPE

**Success:** Wasserstein/MMD distance decreases with GPE. This would indicate GPE
helps produce balanced representations.

**Failure:** Distance increases or stays the same. This would NOT invalidate GPE's
benefit -- it would mean GPE works via a different mechanism than representation
balancing. In fact, if GPE *increases* representation distance but PEHE still
improves, it suggests the model is learning better-conditioned representations
where the treatment effect is more identifiable, even if groups are less balanced.

**Important nuance:** CFRNet/NetDeconf explicitly penalize representation imbalance.
Most of our base models do NOT have this penalty. So representation balance may not
be the relevant axis for non-balanced models (TARNet, X-learner, BNN).

**Testable with existing infrastructure?** YES. Wasserstein distance is already
implemented in `baselines.py`. We just need a forward pass + metric computation.

**Estimated runtime:** 1-2 hours. Same as H1 -- extract representations, compute metric.

### H3: Information Enrichment vs. Attention Architecture

**Claim:** GPE works simply because it adds positional features (information enrichment),
and the cross-attention mechanism is irrelevant. Alternatively, the attention mechanism
provides a crucial benefit beyond simple feature concatenation.

**DGP Analysis:** The DGP's treatment and outcome functions depend on multi-hop aggregates
(`AhZ_c1`, `AhZ_c0`) and community bias (`c_bias`). Node2vec embeddings capture both
of these. So the *information* in node2vec is clearly relevant. The question is whether
the *cross-attention fusion mechanism* matters, or whether concatenating node2vec to features
would be equally effective.

The cross-attention in GPE computes: Q = features, K/V = positions. Each node's features
attend to ALL other nodes' positions. This creates a global receptive field BEFORE any
GNN layer. Simple concatenation only gives each node its OWN position.

**Test:** Three-way comparison:
  (a) Vanilla (no positional info)
  (b) Concatenation: `[X ; node2vec]` as input to GNN (no attention)
  (c) Full GPE with cross-attention

If (b) ~= (c): The information content matters, not the fusion mechanism.
If (c) >> (b) >> (a): Both information and attention contribute.
If (c) >> (b) ~= (a): The attention mechanism is essential; raw concatenation fails.

**Success for "attention matters":** (c) significantly outperforms (b) on PEHE.
**Success for "information suffices":** (b) ~= (c), both better than (a).

**Failure interpretation:** If (b) ~= (a) << (c), then concatenation is useless and
the cross-attention's global receptive field is the key innovation. This would be a
strong result for the paper's architectural contribution.

**Testable with existing infrastructure?** PARTIALLY. The vanilla and full GPE are
already implemented (`use_gpe=True/False`). The concatenation variant requires a small
code change: skip the cross-attention, just concatenate `[X, pos]` and feed into backbone.
This is ~10 lines of code.

**Estimated runtime:** 4-6 hours. Need to run all datasets x noise levels x seeds for
the concatenation variant. But only need TARNet (simplest, most consistent GPE benefit)
and possibly GDC.

### H4: Over-smoothing Prevention

**Claim:** Deep GNNs suffer from over-smoothing (representations converge). GPE provides
a unique positional "anchor" per node that prevents this convergence.

**DGP Analysis:** We use 5 GAT layers. The depth sensitivity results (Table 5 in paper)
show a U-shaped curve with optimal PEHE at L=2-3, degrading at L=5 and L=7. This is
consistent with over-smoothing at deeper layers. If GPE prevents over-smoothing, we
should see the degradation at L=5/7 reduced with GPE.

However, the depth sensitivity was only measured WITH GPE. We lack the comparison
WITHOUT GPE to see if over-smoothing is worse without the positional anchor.

**Test:** Measure representation diversity across layers with and without GPE.
- For each GAT layer l, extract the intermediate representations h^(l)
- Compute MAD (Mean Average Distance): average pairwise L2 distance among all node
  representations at layer l
- Compare MAD curves (layer vs. MAD) with and without GPE
- If GPE maintains higher MAD at deeper layers: over-smoothing prevention confirmed

**Alternative metric:** Effective rank of the representation matrix at each layer.
rank_eff = exp(entropy of singular values). Higher effective rank = more diverse
representations = less over-smoothing.

**Success:** MAD or effective rank at layers 3-5 is significantly higher with GPE
than without. The gap should be larger at deeper layers.

**Failure:** MAD curves are similar with and without GPE, or the gap doesn't grow
with depth. This would mean over-smoothing prevention is not the mechanism.

**Failure interpretation:** If over-smoothing prevention is NOT the mechanism, then
GPE's benefit at L=5 comes from information enrichment (H3) or confounding debiasing
(H1) rather than from preserving representation diversity.

**Testable with existing infrastructure?** MOSTLY. Need to hook into intermediate
GAT layers to extract representations. The GAT class stores layers as `self.layers`;
we can add a forward hook or modify the forward pass to return intermediates.
~20 lines of code.

**Estimated runtime:** 2-3 hours. One forward pass per model per configuration
(no re-training needed), but need to compute pairwise distances which is O(N^2).

### H5: Treatment-Heterogeneity Capture

**Claim:** GPE helps separate-head models (TARNet, GDC) learn DIFFERENT treatment effects
for different network positions, while shared-head models (BNN) cannot leverage this
positional heterogeneity.

**DGP Analysis:** The DGP generates community-dependent treatment effects via `c_bias`:
different communities get different treatment effect magnitudes. This is exactly treatment
heterogeneity driven by network position. BNN uses a shared mu(x,t) head where t is
concatenated -- the treatment effect is `mu(x,1) - mu(x,0)`, which CAN vary by position
but the shared architecture makes it harder to learn separate response surfaces.

The observation that BNN is the ONLY model where GPE does NOT help (and GPE even slightly
hurts it) is strongly suggestive: BNN's shared-head architecture cannot exploit the
additional positional information for differential treatment effects.

**Test:** Compute the variance of predicted ITE across Louvain communities with and
without GPE.
- For each model, predict ITE for all nodes
- Group nodes by their Louvain community
- Compute the between-community variance of mean predicted ITE
- Higher variance = more heterogeneous (position-dependent) treatment effects

Additionally: correlate predicted per-community mean ITE with TRUE per-community mean
ITE. If GPE increases this correlation, it's helping the model capture genuine
heterogeneity, not just adding noise.

**Success:** (1) ITE variance across communities increases with GPE for separate-head
models but NOT for BNN. (2) Correlation between predicted and true community-level
ITE increases with GPE.

**Failure:** ITE variance is unchanged or correlation doesn't improve. This would
mean GPE is not helping with heterogeneity capture.

**Testable with existing infrastructure?** YES. We have `true_tau` and model
predictions. Community labels are in `node_class`. Pure post-processing.

**Estimated runtime:** 30 minutes. No re-training needed.

---

## 4. Hypothesis Ranking

Ranking by: (scientific rigor) x (ease of testing) x (relevance to RecSys reviewers)

### Rank 1: H1 -- Confounding Debiasing

**Scientific rigor: 9/10.** Directly grounded in Veitch et al. (NeurIPS 2019).
The DGP literally injects community-level confounding via `c_bias` that affects
both treatment and outcome. Propensity AUC is a well-understood metric with clear
interpretation. The causal inference community recognizes this as a legitimate
debiasing test.

**Ease of testing: 9/10.** The propensity head already exists. Extract predictions,
compute AUC. No code changes needed.

**Reviewer relevance: 10/10.** Reviewers will ask "why does adding position help
for causal estimation specifically?" Answering "because position is a confounder
and GPE makes it observable" is the most satisfying answer from a causal inference
perspective. It connects GPE to established causal theory. It also explains the
BNN exception: if BNN's shared head cannot properly adjust for this confounder
even when it's available, that's consistent.

### Rank 2: H3 -- Information Enrichment vs. Attention

**Scientific rigor: 8/10.** Clean ablation with three conditions. Standard ablation
methodology. The result is directly interpretable regardless of outcome.

**Ease of testing: 7/10.** Requires a small code change for the concatenation variant
and re-running experiments. Not zero-cost, but manageable.

**Reviewer relevance: 9/10.** This is EXACTLY what ICML reviewers asked about
("no alternative PE comparison" is listed as concern #5 in the reviews). The
concatenation variant partially addresses this by showing that the *fusion method*
matters independently of the PE method. If concatenation is nearly as good as
cross-attention, that actually simplifies the contribution story: "any positional
info helps." If cross-attention is critical, it justifies the architectural choice.

### Rank 3: H5 -- Treatment-Heterogeneity Capture

**Scientific rigor: 7/10.** The test (ITE variance across communities + correlation
with true community ITE) is sound but somewhat indirect. The variance metric alone
is hard to interpret -- more variance is only better if it correlates with true
heterogeneity.

**Ease of testing: 10/10.** Pure post-processing of existing predictions.

**Reviewer relevance: 8/10.** Explains the BNN anomaly, which is a natural question.
RecSys reviewers care about when-to-use guidance, and "GPE helps more when the
model architecture can exploit treatment heterogeneity" is actionable advice.

### Rank 4: H4 -- Over-smoothing Prevention

**Scientific rigor: 7/10.** MAD/effective rank are established metrics, but the
connection to PEHE improvement is indirect. Over-smoothing prevention helps all
tasks, not just causal estimation -- it does not explain why GPE specifically
helps ITE.

**Ease of testing: 6/10.** Requires hooking into intermediate layers and computing
O(N^2) pairwise distances. More engineering than the top 3.

**Reviewer relevance: 5/10.** Over-smoothing is a GNN concern, not a causal
inference concern. RecSys reviewers care more about the causal story.

### Rank 5: H2 -- Representation Balance

**Scientific rigor: 6/10.** The prediction is ambiguous: GPE could legitimately
INCREASE the representation gap (by making position -- which correlates with
treatment -- more explicit) while still improving PEHE. A result of "GPE increases
imbalance but PEHE improves" is hard to interpret without additional theory.

**Ease of testing: 8/10.** Wasserstein distance is already implemented.

**Reviewer relevance: 5/10.** Representation balance is relevant for NetDeconf-style
models that explicitly optimize for it, but not for the majority of our baselines
(TARNet, X-learner, GNUM). A mixed result would confuse rather than clarify.

---

## 5. Detailed Experiment Designs for Top 3 Hypotheses

### Experiment 1: H1 -- Propensity AUC Analysis (Confounding Debiasing)

#### 5.1.1 Overview

**Goal:** Quantify whether GPE captures treatment-assignment-relevant information
that raw features + GNN message-passing miss.

**Core metric:** ROC-AUC of the learned propensity score `e(x)` predicting treatment
assignment `T` on held-out test nodes.

#### 5.1.2 Protocol

```
For each dataset in {CoraFull, DBLP, PubMed}:
  For each rho in {5, 10, 30}:
    For each seed in {0, 1, 2, 3, 4}:  # 5 seeds for statistical power
      For each model in {TARNet, GDC, NetDeconf, X-learner, BNN}:
        1. Train model WITHOUT GPE (use_gpe=False) for 200 epochs
        2. Train model WITH GPE (use_gpe=True) for 200 epochs
        3. At best checkpoint (selected on val PEHE):
           a. Forward pass on full graph
           b. Extract propensity predictions e(x) for test nodes
           c. Compute ROC-AUC(e(x), T) on test nodes
           d. Compute calibration: Brier score of e(x)
        4. Additionally, train standalone propensity models:
           a. Logistic regression on raw features X
           b. Logistic regression on [X; node2vec]
           c. Logistic regression on GNN representations (no GPE)
           d. Logistic regression on GPE-fused representations
           This isolates representation quality from head capacity
```

#### 5.1.3 Analysis Plan

1. **Table: Propensity AUC with/without GPE** for each (model, dataset, rho).
   Format: same as main PEHE table but reporting AUC instead.

2. **Scatter plot: Delta-AUC vs. Delta-PEHE.** Each point is one (model, dataset,
   rho) configuration. If H1 is correct, these should be positively correlated
   (higher AUC improvement -> higher PEHE improvement).

3. **Per-community propensity analysis:** Compute AUC within each Louvain community.
   If GPE specifically helps in communities where position is most confounding
   (high `c_bias` variance), this provides granular support.

4. **BNN control:** BNN should show either (a) propensity AUC improves but PEHE
   does not (shared head cannot exploit better propensity), or (b) propensity AUC
   does not improve (BNN representations are too coarse for position to matter).
   Both interpretations are informative.

#### 5.1.4 Success Criteria

- AUC improves in >= 80% of (model x dataset x rho) configurations (excluding BNN)
- Correlation between Delta-AUC and Delta-PEHE has r >= 0.4 with p < 0.05
- BNN is a clear outlier in the scatter plot

#### 5.1.5 What to Report in Paper

If successful, add a new subsection "GPE as Confounder Proxy" with:
- 1 table (propensity AUC comparison)
- 1 figure (Delta-AUC vs Delta-PEHE scatter with regression line)
- 2-3 sentences connecting to Veitch et al. (2019) theory

This would be ~0.5 pages of paper content, replacing or supplementing the current
t-SNE visualization.

#### 5.1.6 Code Sketch

```python
# After loading best checkpoint and running forward pass:
from sklearn.metrics import roc_auc_score, brier_score_loss

model.eval()
with torch.no_grad():
    out = model(X, graph, pos)
    e_pred = out.e[test_idx].cpu().numpy()
    t_true = sample.T[test_idx].cpu().numpy()

auc = roc_auc_score(t_true, e_pred)
brier = brier_score_loss(t_true, e_pred)
```

---

### Experiment 2: H3 -- Information vs. Attention Ablation

#### 5.2.1 Overview

**Goal:** Isolate whether GPE's benefit comes from the information content of
node2vec embeddings or from the cross-attention fusion mechanism.

#### 5.2.2 Variants

| Variant | Description | Input to GNN | Code Change |
|:--------|:-----------|:-------------|:------------|
| (a) Vanilla | No positional info | X | `use_gpe=False` (existing) |
| (b) Concat | Concatenate node2vec | [X; pos] | New: skip attention, just cat |
| (c) Add | Add projected node2vec | X + W_p * pos | New: linear projection + add |
| (d) Full GPE | Cross-attention fusion | GPE(X, pos) | `use_gpe=True` (existing) |

Variant (b) is the critical control. If (b) ~= (d), the attention is irrelevant.
If (d) >> (b), the attention architecture provides genuine value.

Variant (c) adds a further control: does the projection itself matter, or is
it the cross-node attention? Add-mode only gives each node its own projected
position; cross-attention gives it access to all other nodes' positions.

#### 5.2.3 Protocol

```
For each dataset in {CoraFull, DBLP, PubMed}:
  For each rho in {5, 10, 30}:
    For each seed in {0, 1, 2, 3, 4}:
      For TARNet only (cleanest signal):
        Train variants (a), (b), (c), (d) with identical hyperparameters
        Record PEHE, Qini, ATE error at best-val checkpoint
      For GDC (to verify generality):
        Train variants (a) and (b) and (d) [skip (c) for time]
```

#### 5.2.4 Implementation

```python
# New ConcatFusion module (replaces GPEFusion for variant b):
class ConcatFusion(nn.Module):
    def __init__(self, feat_dim, pos_dim):
        super().__init__()
        self.out_dim = feat_dim + pos_dim

    def forward(self, h, p):
        return torch.cat([h, p], dim=-1)

# New AddFusion module (variant c):
class AddFusion(nn.Module):
    def __init__(self, feat_dim, pos_dim):
        super().__init__()
        self.proj = nn.Linear(pos_dim, feat_dim)
        self.out_dim = feat_dim

    def forward(self, h, p):
        return h + self.proj(p)
```

#### 5.2.5 Analysis Plan

1. **Table: PEHE across 4 variants** (same format as main table, but 4 rows
   instead of 2).

2. **Statistical test:** Paired t-test or Wilcoxon signed-rank between variants
   (b) and (d) across all (dataset, rho, seed) configurations. If p > 0.05, we
   cannot reject "concat is as good as attention."

3. **Parameter count analysis:** Report parameter counts for each variant to
   ensure the comparison is fair. If (d) has 2x parameters, the comparison
   needs qualification.

| Variant | Added Parameters | Notes |
|:--------|:----------------|:------|
| (a) Vanilla | 0 | Baseline |
| (b) Concat | 0 | pos_dim more input features to first GNN layer |
| (c) Add | pos_dim * feat_dim | One linear projection |
| (d) Full GPE | ~4 * embed_dim * (feat_dim + pos_dim) | Q/K/V/O projections + residual |

#### 5.2.6 Success Criteria

- If (d) > (b) by >= 5% PEHE on >= 7/9 cells: attention matters. Report this.
- If (b) ~= (d) (within 3% PEHE): simplify the story to "positional information
  is what matters; attention is a nice-to-have." This is actually publishable too.
- If (b) ~= (a): raw concatenation of node2vec does not help, only structured
  fusion works. This is the strongest result for the GPE architecture.

#### 5.2.7 What to Report in Paper

Replace or augment the "Why cross-attention?" paragraph in Section 6.2 with a table
and 3-4 sentences. This directly addresses ICML reviewer concern #5 (alternative PE
comparison) because it tests whether the *fusion method* matters.

---

### Experiment 3: H5 -- Treatment Heterogeneity Capture + BNN Explanation

#### 5.3.1 Overview

**Goal:** Show that GPE enables models with separate treatment heads to learn more
accurate treatment-heterogeneous effects across communities, while BNN's shared
architecture cannot exploit this.

#### 5.3.2 Metrics

1. **Between-community ITE variance (BCV):**
   For each community c, compute mean predicted ITE: `tau_c = mean(pred_ITE[nodes in c])`.
   Then BCV = Var({tau_c}) across all communities.

2. **Community ITE correlation (CIC):**
   Correlation between `{tau_c_predicted}` and `{tau_c_true}` across communities.
   This distinguishes signal from noise: high BCV but low CIC = GPE adds noise.

3. **Per-node ITE correlation (PIC):**
   For completeness: Pearson correlation between `pred_ITE_i` and `true_ITE_i`.

4. **Community-conditional PEHE:**
   PEHE computed separately within each community. If GPE helps more in communities
   with high `c_bias` (strong position-driven treatment effects), this is direct
   evidence.

#### 5.3.3 Protocol

```
For each dataset in {CoraFull, DBLP, PubMed}:
  For rho in {5, 10}:  # skip rho=30 where noise dominates
    For each seed in {0, 1, 2, 3, 4}:
      For each model in {TARNet, GDC, X-learner, BNN}:
        1. Load best checkpoint (with and without GPE)
        2. Predict ITE for all nodes
        3. Compute BCV, CIC, PIC with and without GPE
        4. Compute community-conditional PEHE
```

#### 5.3.4 Analysis Plan

1. **Bar chart: BCV with/without GPE** for each model. Expected pattern:
   - TARNet, GDC, X-learner: BCV increases with GPE (they learn more heterogeneous effects)
   - BNN: BCV unchanged (shared head cannot differentiate)

2. **Table: CIC with/without GPE.** The key test: does the increased BCV reflect
   *real* heterogeneity?
   - TARNet+GPE: CIC increases (captures true community-driven heterogeneity)
   - BNN+GPE: CIC unchanged (cannot exploit position for heterogeneity)

3. **Community-conditional PEHE heatmap:** (model x community) with color = PEHE.
   Show that GPE specifically helps in communities with high true treatment effect
   variance.

#### 5.3.5 Code Sketch

```python
import numpy as np
from scipy.stats import pearsonr

# pred_ite: (N,) predicted ITE; true_ite: (N,) ground truth
# node_class: (N,) community assignments

communities = np.unique(node_class)
pred_community_means = [pred_ite[node_class == c].mean() for c in communities]
true_community_means = [true_ite[node_class == c].mean() for c in communities]

bcv = np.var(pred_community_means)
cic, cic_p = pearsonr(pred_community_means, true_community_means)

# Community-conditional PEHE
for c in communities:
    mask = node_class == c
    pehe_c = np.sqrt(np.mean((pred_ite[mask] - true_ite[mask])**2))
```

#### 5.3.6 Success Criteria

- BCV increases with GPE for separate-head models (TARNet, GDC, X-learner) in >= 80% of configs
- BCV does NOT increase for BNN
- CIC increases with GPE for separate-head models (r improvement >= 0.1)
- Community-conditional PEHE improvement correlates with community's true treatment
  effect variance

#### 5.3.7 What to Report in Paper

This becomes the most interesting subsection: "GPE enables treatment heterogeneity
capture." Include:
- 1 bar chart (BCV across models, with/without GPE)
- 1 table (CIC values)
- A paragraph explaining the BNN exception in terms of architectural capacity

This replaces the current t-SNE visualization with a much more convincing analysis
and simultaneously explains the BNN anomaly that reviewers will ask about.

---

## 6. Experimental Priority and Timeline

| Priority | Experiment | New Code | Training Runs | Post-Processing Only | Estimated Time |
|:---------|:-----------|:---------|:-------------|:--------------------|:--------------|
| **1** | H5: Heterogeneity | None | 0 (use existing) | Yes | 30 min |
| **2** | H1: Propensity AUC | None | 0 (use existing) | Yes | 1-2 hrs |
| **3** | H3: Concat ablation | ~20 LOC | ~135 runs | No | 4-6 hrs |

**Recommended execution order:**
1. Run H5 first (30 min, pure post-processing, immediately informative)
2. Run H1 in parallel (extract propensity from same checkpoints)
3. If H1 and H5 both succeed, the mechanism story is strong without H3
4. Run H3 only if reviewer demands it or if H1/H5 give ambiguous results

**Total time for H1 + H5:** Under 2 hours with existing checkpoints.
**Total time including H3:** 6-8 hours.

---

## 7. Synthesis: The Most Likely Mechanism Story

Based on the DGP analysis, the most likely explanation is a *combination* of H1 and H5:

**GPE captures community-level confounding (H1) that enables separate-head models to
learn position-dependent treatment effects (H5).**

The causal chain is:
1. The DGP injects community bias (`c_bias`) that affects BOTH treatment assignment
   AND outcomes -- this is a classic confounder.
2. Raw features X + GNN message-passing can capture local neighborhood effects but
   NOT the global community identity that `c_bias` represents.
3. Node2vec embeddings encode community membership (this is what node2vec does by
   design with its random walk process).
4. GPE makes this community information available to the model as an explicit feature.
5. Separate-head models (TARNet, GDC, X-learner) can use this information to learn
   community-specific treatment effects: mu_1(x, community) and mu_0(x, community)
   produce different ITE estimates for different communities.
6. BNN's shared head `mu(x, t)` has the position information available but cannot
   efficiently use it for DIFFERENTIAL treatment effects because both treatment
   arms share the same function.

This story is:
- Grounded in established causal theory (Veitch et al. 2019)
- Consistent with the DGP structure (provably)
- Explains the BNN exception (architecturally)
- Testable with quantitative metrics (propensity AUC, community ITE correlation)
- Actionable for practitioners (use GPE when community structure drives treatment
  heterogeneity, and prefer separate-head architectures)

If the experiments confirm this story, Section 5.5 ("Representation Visualization")
should be rewritten as "Mechanism Analysis: GPE as Confounder Proxy" with quantitative
evidence replacing the t-SNE plots.

---

## 8. Risk Analysis: What If the Experiments Fail?

### Scenario A: H1 succeeds, H5 fails
GPE helps with confounding but not heterogeneity. The BNN exception must be explained
differently -- perhaps BNN's shared architecture is just worse at adjusting for any
confounder, even when it's available. This is still a publishable story.

### Scenario B: H1 fails, H5 succeeds
GPE does NOT help with confounding (propensity AUC unchanged) but DOES help with
heterogeneity. This is surprising but possible if the GNN already captures enough
treatment-assignment signal from local neighborhoods. The story becomes: "GPE helps
by providing a richer basis for heterogeneous treatment effect estimation" without
the confounder framing.

### Scenario C: Both H1 and H5 fail
GPE's benefit is not explained by confounding or heterogeneity. Revisit H3 and H4.
If the concatenation ablation (H3) shows that raw concatenation is as good as
cross-attention, the story is simply "more features = better predictions" -- a
weaker but honest result.

### Scenario D: DGP circularity concern
A reviewer may argue: "Of course GPE helps -- your DGP explicitly injects community
confounding that node2vec captures. This is circular." The defense is:
1. The same community structure exists in real social networks (our BlogCatalog/Flickr
   results, though weaker, show GPE helps there too)
2. The DGP is designed to be *realistic*, not to favor GPE. The community confounding
   reflects documented phenomena in social influence literature.
3. If GPE helps because it captures real confounders, that IS the point -- we want
   methods that capture the confounders that exist in real data.

---

## 9. Literature References for Citation

If the mechanism analysis enters the paper, cite these for theoretical grounding:

1. **Veitch, Wang, Blei (NeurIPS 2019)** -- "Using Embeddings to Correct for
   Unobserved Confounding in Networks." Direct theoretical support for H1.
   Key theorem: network embeddings serve as valid proxy confounders.

2. **Sridhar & Getoor (2019)** -- "Using Embeddings for Causal Estimation of Peer
   Influence in Social Networks." Extension of the above to peer influence settings.

3. **Shalit, Johansson, Sontag (ICML 2017)** -- CFRNet paper. For the representation
   balance framework (Wasserstein/MMD) used in H2, even if H2 is not the primary story.

4. **Chen et al. (AAAI 2020)** -- "Measuring and Relieving the Over-Smoothing Problem."
   For the MAD metric if H4 is tested.

5. **Rampasek et al. (NeurIPS 2022)** -- GraphGPS. For the general argument that PE
   increases MPNN expressiveness.

6. **Wu et al. (ICLR 2025)** -- CauGramer. For comparison with concurrent work using
   positional information in causal graph models.

---

## 10. Integration with Paper Revision Strategy

### Current paper weaknesses this addresses:

1. **"No mechanism explanation"** -- Section 5.5's t-SNE is weak evidence. This analysis
   provides quantitative alternatives.

2. **"Why does BNN not benefit?"** -- Currently hand-waved. H5 provides an architectural
   explanation grounded in treatment heterogeneity.

3. **ICML concern #5 (no alternative PE comparison)** -- H3's concat ablation partially
   addresses this without running Laplacian/RWPE experiments.

4. **Semi-synthetic circularity** -- The H1 analysis explicitly acknowledges and defends
   against the circularity concern, which is better than ignoring it.

### Suggested paper revision:

Replace Section 5.5 ("Representation Visualization") with a new Section 5.5
("Mechanism Analysis") structured as:

1. **Paragraph 1:** "Why does GPE help? We test three hypotheses..." (framing)
2. **Table + scatter plot:** Propensity AUC analysis (H1)
3. **Bar chart + table:** Treatment heterogeneity analysis (H5)
4. **If space permits:** Concat ablation table (H3)
5. **Paragraph:** "GPE captures community-level confounding..." (synthesis)
6. **Paragraph:** "This explains BNN's exception..." (architectural argument)

This transforms the weakest section of the paper into potentially the most
interesting one.
