# IJCNN Review Summary — Paper 4401 (CAVIN)

**Overall:** 5 reviewers. Scores: accept(4), weak accept(3), weak accept(3), weak reject(2), weak reject(2).
**Meta-review:** Weak Accept. "Needs more details about method and experimental analysis."

---

## Consensus Issues

### 1. Semi-synthetic only, no real-world validation (4/5 reviewers)
- "Semi-synthetic data generation may favor the proposed method" (R3)
- "Robustness and generalizability for practical industrial deployment remain to be verified" (R2)
- **Status in new paper:** ✅ ADDRESSED — 6 real-world tabular datasets

### 2. Combines existing components — novelty limited (3/5 reviewers)
- "Method mainly combines existing components" (R3)
- "Not clearly demonstrated sufficient novelty beyond this combination" (R3)
- **Status in new paper:** ✅ REFRAMED as plug-and-play paradigm

### 3. Missing sensitivity analysis (3/5 reviewers)
- "5 GAT layers not justified — may suffer over-smoothing" (R3)
- "Absence of sensitivity analyses for critical hyperparameters" (R2)
- **Status in new paper:** ⚠️ PARTIALLY — design choices discussed but no systematic sweep

### 4. Theoretical concerns about variance estimation (2/5 reviewers)
- "Variance target D² − τ̂² may be negative" (R1)
- "Positivity constraints not clearly enforced beyond max(σ̂², δ)" (R1)
- "Training σ̂² without log-scale or likelihood stabilization" (R1)
- **Status in new paper:** ✅ FIXED — we use log-σ² parameterization in the code

### 5. Missing baselines (2/5 reviewers)
- "Missing comparisons to stronger network interference baselines" (R1)
- "Missing comparisons to alternative positional encodings (Laplacian PE, RWPE)" (R1)
- **Status in new paper:** ✅ MOSTLY — 8 baselines now. Still missing PE alternatives

---

## Positive Comments (for what reviewers liked)

- "Technical novelty and innovation" — positional encoding in GNN ITE is creative (R1)
- "Addresses a timely and important setting" (R1)
- "Reliable tool for personalized advertising" (R5, scored Accept)
- "Theoretically rigorous — Gauss-Markov proof" (R5)
- "Ablation on GPE showing consistent improvements" (R1)
- "Paper is clear, logically arranged, structurally rigorous" (R5)

---

## Key Action Items for RecSys Paper

### Already addressed:
- ✅ Real-world datasets (6 tabular)
- ✅ More baselines (8 models, 2020-2025)
- ✅ Plug-and-play framing
- ✅ log-σ² parameterization (in code, should mention in paper)
- ✅ Ablation isolating variance weighting contribution

### Still needs paper-level fixes (no experiments):
- ⚠️ Explicitly mention log-σ² parameterization in method section (prevents negative variance)
- ⚠️ Discuss over-smoothing risk with 5 GAT layers and justify depth choice
- ⚠️ Add note about alternative PE methods as future work
- ⚠️ Improve figure self-explanatory power (legends, axis labels)
- ⚠️ Mention cross-fitting details in algorithm description

### Would need new experiments:
- 🔬 Alternative PE comparison (Laplacian, RWPE)
- 🔬 Over-smoothing analysis with varying GAT depth
- 🔬 Visualization (t-SNE of embeddings with/without GPE)
- 🔬 Dynamic graph / scalability tests
