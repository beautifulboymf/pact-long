# ICML 2025 Review Summary — Submission 8899 (CAVIN)

**Overall:** 4 reviewers, all scored 2 (Weak Reject). Rejected.

---

## Consensus Issues (raised by 3+ reviewers)

### 1. No real-world data validation (ALL 4 reviewers)
- Semi-synthetic datasets may be designed to favor CAVIN
- "The datasets are generated based on the paper's assumptions" — circular validation
- **Status in new paper:** ✅ ADDRESSED — we now have 6 real-world tabular datasets (Criteo, Hillstrom, RetailHero, Lenta, X5)

### 2. Insufficient novelty — combination of existing ideas (3/4 reviewers)
- GPE = node2vec + attention (both known), variance weighting = WLS (well-known)
- "Such combination is not a novel idea in this area" (Reviewer jpBE)
- **Status in new paper:** ✅ REFRAMED — we now position as "plug-and-play modules" tested on 8 models, not a new model

### 3. Theory is insufficient / unjustified assumptions (3/4 reviewers)
- Gauss-Markov proof only for linear case, but model is non-linear
- Estimated (not true) variance used — gap not analyzed
- Too many assumptions (7) without justification
- **Status in new paper:** ⚠️ PARTIALLY — we added the "Remark on Linearity" but should soften claims further

### 4. Missing baselines (3/4 reviewers)
- Missing graph transformer baselines (Graphormer, etc.) that inherently use positional encoding
- Missing stronger interference baselines (Forastiere 2021, Chen 2024 DR)
- Missing comparison to alternative positional encodings (Laplacian PE, RWPE)
- **Status in new paper:** ✅ MOSTLY — we added NetDeconf, GIAL, GNUM, GDC. Still missing graph transformers

### 5. Missing ablation / sensitivity analysis (3/4 reviewers)
- No isolated variance weighting ablation
- No hyperparameter sensitivity (GAT layers, LR, δ, etc.)
- No justification for 5 GAT layers / LR=0.05
- **Status in new paper:** ✅ ADDRESSED — we now have full ablation table + design choices discussion

---

## Reviewer-Specific Issues

### Reviewer jpBE (most detailed)
- Missing references: Ma 2021, Sun 2020, Zhao 2024, Huang 2023
- Typo in Eq. 22, Assumption 3.7 seems incorrect
- **Status:** ⚠️ Need to cite these 4 papers in new paper

### Reviewer ZqhE
- "Could you provide evidence for why real-world settings exhibit these patterns?"
- Community detection sensitivity — would results hold with other methods?
- **Status:** ✅ Our new plug-and-play framing + real-world results address this

### Reviewer ykvQ (most positive weak reject)
- Wants qualitative analysis (visualization of attention/communities)
- Wants more justification for Assumption 3.9 (conditional independence of residuals)
- **Status:** ⚠️ Visualization still missing. Could add t-SNE or attention heatmap

### Reviewer dSf9 (harshest)
- "Methodology novelty is minor"
- Missing graph transformer baselines
- Linear assumption is "NOT without loss of generality"
- Experiments on large graphs needed
- **Status:** ⚠️ Graph transformers still not compared. Large graph scalability not tested

---

## What we HAVE fixed in the new RecSys paper:
- ✅ Real-world tabular validation (6 datasets)
- ✅ 8 base models including 2025 SOTA (GDC)
- ✅ Plug-and-play framing (not monolithic model)
- ✅ Full ablation (GPE ± Var for each model)
- ✅ Bernoulli variance adaptation for binary outcomes (NEW)
- ✅ Design choices discussion (node2vec, δ, attention architecture)
- ✅ BlogCatalog/Flickr standard benchmarks

## What STILL needs fixing (paper-level, no new experiments):
- ⚠️ Soften theoretical claims ("under local linearity" not "without loss of generality")
- ⚠️ Cite Ma 2021, Sun 2020, Zhao 2024, Huang 2023
- ⚠️ Discuss graph transformer comparison as limitation
- ⚠️ Add sentence about Assumption 3.9 justification

## What would need NEW experiments:
- 🔬 Graph transformer baseline (Graphormer + PE)
- 🔬 Alternative PE comparison (Laplacian PE, RWPE vs node2vec)
- 🔬 Large graph scalability test
- 🔬 Sensitivity to community detection method
- 🔬 t-SNE visualization of embeddings
