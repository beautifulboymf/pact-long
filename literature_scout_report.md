# Literature Scout Report: Graph-Based ITE / Uplift Modeling for RecSys Submission

**Generated:** 2026-04-09
**Paper Context:** Two plug-and-play modules (GPE = Graph Positional Encoding, Variance-Weighted Learning) that improve ANY graph-based ITE estimator for uplift modeling in recommender systems — targeting ACM RecSys 2025.

---

## CATEGORY 1: RecSys / KDD / WWW / CIKM Papers on Uplift Modeling with GNNs

---

### Paper 1 — E3IR (MUST CITE / POSITION AGAINST)

**Title:** End-to-End Cost-Effective Incentive Recommendation under Budget Constraint with Uplift Modeling

**Authors:** Zexu Sun, Hao Yang, Dugang Liu, Yunpeng Weng, Xing Tang, Xiuqiang He

**Venue & Year:** ACM RecSys 2024

**ArXiv:** https://arxiv.org/abs/2408.11623

**Summary:** Proposes E3IR, a two-module end-to-end architecture combining an uplift prediction head (with monotonic/smooth marketing constraints) and a differentiable integer linear programming (ILP) allocation layer to jointly optimize uplift prediction and budget-constrained incentive assignment.

**Relationship to our work:**
- **Must cite as the closest RecSys peer.** E3IR treats uplift modeling + allocation as an end-to-end problem but does NOT use graph structure. Our GPE module would directly plug into E3IR's uplift prediction head. Frame our work as complementary: E3IR solves the allocation problem; our modules solve the representation quality problem for any graph-based ITE estimator feeding such systems.
- Position against: E3IR is a two-stage improvement (prediction + allocation). Our work improves the prediction stage through modular GPE + variance-weighted learning, applicable across all graph ITE estimators including those used by E3IR.

---

### Paper 2 — GNUM (MUST CITE / COMPARE)

**Title:** Graph Neural Network with Two Uplift Estimators for Label-Scarcity Individual Uplift Modeling

**Authors:** Dingyuan Zhu, Daixin Wang, Zhiqiang Zhang, Kun Kuang, Yan Zhang, Yulin Kang, Jun Zhou

**Venue & Year:** ACM Web Conference (WWW) 2023

**ACM DL:** https://dl.acm.org/doi/abs/10.1145/3543507.3583368  |  **ArXiv:** https://arxiv.org/abs/2403.06489

**Summary:** Proposes GNUM, a GNN framework for uplift modeling under label scarcity, using a class-transformed estimator (for any outcome type) and a partial-label estimator (for discrete outcomes) that leverage social graph structure to improve uplift estimation.

**Relationship to our work:**
- **Must cite and compare against.** GNUM is the most direct prior work — it is a GNN-based uplift estimator for recommendation. Our GPE and variance-weighted modules are explicitly plug-and-play improvements to estimators like GNUM. Use GNUM as one of the "backbone" estimators in experiments.
- Key differentiator: GNUM addresses label scarcity but does NOT use structural positional encodings or variance-aware sample weighting. Our GPE enriches node representations; variance-weighted learning improves training signal quality.

---

### Paper 3 — CDUM (MUST CITE)

**Title:** Enhancing Online Video Recommendation via a Coarse-to-fine Dynamic Uplift Modeling Framework

**Authors:** Chang Meng, Chenhao Zhai, Xueliang Wang, Shuchang Liu, Xiaoqiang Feng, Lantao Hu, Xiu Li, Han Li, Kun Gai (Kuaishou Technology / Tsinghua University)

**Venue & Year:** ACM RecSys 2025

**ACM DL:** https://dl.acm.org/doi/10.1145/3705328.3748070  |  **ArXiv:** https://arxiv.org/abs/2410.16755

**Summary:** Proposes CDUM, a two-module uplift framework for real-time video recommendation at Kuaishou, with a coarse-grained module for long-term user preferences and a fine-grained module for real-time contextual signals; deployed serving hundreds of millions of users.

**Relationship to our work:**
- **Cite as concurrent/complementary RecSys 2025 work on uplift for recommendation.** CDUM addresses the dynamic/temporal aspect of uplift modeling. It does not use GNNs or graph structure. Our work addresses the GNN representation quality problem; the two approaches are orthogonal. Citing CDUM demonstrates awareness of the state of the art at the same venue.

---

### Paper 4 — GNN+Causal Knowledge Uplift

**Title:** Uplift Modeling based on Graph Neural Network Combined with Causal Knowledge

**Authors:** Haowen Wang, Xinyan Ye, Yangze Zhou, Zhiyi Zhang, Longhan Zhang, Jing Jiang

**Venue & Year:** IEEE Conference (IJCNN 2024) — also available via IEEE Xplore

**ArXiv:** https://arxiv.org/abs/2311.08434

**Summary:** Proposes a GCN-based uplift framework that integrates causal knowledge (CATE estimation) with adjacency matrix structure learning to improve uplift value estimation.

**Relationship to our work:**
- **Cite as a related graph-based uplift method.** This paper demonstrates the community's growing interest in combining GNNs with causal knowledge for uplift. Our GPE adds a principled structural encoding component that these methods lack. Differentiate by noting our plug-and-play generality vs. their task-specific causal graph learning.

---

### Paper 5 — S-CIEE (MUST CITE)

**Title:** Inter- and Intra-Similarity Preserved Counterfactual Incentive Effect Estimation for Recommendation Systems

**Authors:** (Authors confirmed via ACM DL — see https://dl.acm.org/doi/10.1145/3722104)

**Venue & Year:** ACM Transactions on Information Systems (TOIS), 2025

**Summary:** Proposes S-CIEE, which preserves intra-similarity within treatment groups and inter-similarity between covariate/representation spaces using Fused Gromov-Wasserstein Optimal Transport, plus a rank-aware learning module for uplift ranking in recommendation.

**Relationship to our work:**
- **Must cite as a direct related work on counterfactual incentive effect estimation for recommendation.** Like our work, S-CIEE addresses the representation learning quality problem for ITE in recommendation. Key differentiator: S-CIEE uses optimal transport for similarity preservation; our variance-weighted learning uses prediction variance as sample weights. Both address representation quality but through different mechanisms. Consider comparing empirically if data permits.

---

### Paper 6 — Uplift Modeling: From Causal Inference to Personalization (Survey/Tutorial)

**Title:** Uplift Modeling: From Causal Inference to Personalization

**Authors:** Felipe Moraes, Hugo Manuel Proença, Anastasiia Kornilova, Javier Albert, Dmitri Goldenberg

**Venue & Year:** CIKM 2023 (Tutorial)

**ACM DL:** https://dl.acm.org/doi/abs/10.1145/3583780.3615298  |  **ArXiv:** https://arxiv.org/abs/2308.09066

**Summary:** A comprehensive tutorial connecting causal inference theory to practical uplift modeling for personalization, covering meta-learners, evaluation metrics (Qini, AUUC), and deployment in e-commerce.

**Relationship to our work:**
- **Cite in related work / background section.** Provides authoritative framing for uplift modeling in the personalization/recommendation context. Use to establish terminology and motivate the problem.

---

## CATEGORY 2: Graph Transformer / Positional Encoding Papers (Reviewer-Requested)

---

### Paper 7 — Graphormer (MUST CITE)

**Title:** Do Transformers Really Perform Badly for Graph Representation?

**Authors:** Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu

**Venue & Year:** NeurIPS 2021

**Proceedings:** https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html  |  **ArXiv:** https://arxiv.org/abs/2106.05234

**Summary:** Proposes Graphormer, a graph Transformer that encodes structural information via centrality encoding (node degree), spatial encoding (shortest-path distances), and edge encoding in attention — achieving top performance on molecular graph benchmarks.

**Relationship to our work:**
- **Must cite — reviewers specifically requested discussion of Graphormer.** Our GPE module draws inspiration from Graphormer's spatial/structural encoding design. Distinguish our approach: Graphormer uses shortest-path distance as PE in the attention mechanism of a full Transformer, while our GPE is a plug-and-play module added to existing GNN backbones without requiring a Transformer architecture. Our approach is computationally lighter and backbone-agnostic.

---

### Paper 8 — GraphGPS (MUST CITE)

**Title:** Recipe for a General, Powerful, Scalable Graph Transformer

**Authors:** Ladislav Rampášek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini

**Venue & Year:** NeurIPS 2022

**Proceedings:** https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html  |  **ArXiv:** https://arxiv.org/abs/2205.12454

**Summary:** Proposes GPS, a modular graph Transformer recipe combining: (i) positional/structural encodings, (ii) local message-passing, and (iii) global attention — achieving linear O(N+E) complexity and SOTA on diverse benchmarks.

**Relationship to our work:**
- **Must cite — reviewers asked for this.** GPS is the most influential modular framework for graph positional encoding. Our GPE module is conceptually aligned with the PE component of GPS but is designed specifically for bipartite user-item graphs in uplift modeling rather than general molecular/citation graphs. Explicitly discuss why our design choices differ from GPS (e.g., PE type selection for heterogeneous graphs, integration with ITE estimation objectives).

---

### Paper 9 — SAN (MUST CITE)

**Title:** Rethinking Graph Transformers with Spectral Attention

**Authors:** Devin Kreuzer, Dominique Beaini, William L. Hamilton, Vincent Létourneau, Prudencio Tossou

**Venue & Year:** NeurIPS 2021

**Proceedings:** https://proceedings.neurips.cc/paper_files/paper/2021/hash/b4fd1d2cb085390fbbadae65e07876a7-Abstract.html  |  **ArXiv:** https://arxiv.org/abs/2106.03893

**Summary:** Proposes SAN, a graph Transformer using Laplacian eigenvectors as positional encodings and Laplacian eigenvalues to re-weight attention — providing theoretically grounded spectral PE and full-graph attention without over-squashing.

**Relationship to our work:**
- **Must cite — reviewers specifically named SAN.** SAN's Laplacian-based PE is the spectral baseline for our GPE. Discuss: (1) sign ambiguity of Laplacian eigenvectors (addressed by SignNet below), (2) scalability of SAN's full spectrum computation for large user-item graphs, and (3) how our GPE adapts or selects from spectral PE methods for the recommendation graph setting.

---

### Paper 10 — SignNet (MUST CITE)

**Title:** Sign and Basis Invariant Networks for Spectral Graph Representation Learning

**Authors:** Derek Lim, Joshua Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, Stefanie Jegelka

**Venue & Year:** ICLR 2023

**OpenReview:** https://openreview.net/forum?id=Q-UHqMorzil  |  **ArXiv:** https://arxiv.org/abs/2202.13013

**Summary:** Proposes SignNet and BasisNet, neural architectures that are provably invariant to sign flips and basis changes in Laplacian eigenvectors, enabling universal approximation of continuous functions of eigenvectors and outperforming prior spectral methods.

**Relationship to our work:**
- **Must cite — reviewers specifically named SignNet.** SignNet directly addresses the sign ambiguity problem of Laplacian eigenvector-based PE (including SAN). If our GPE uses spectral encodings, justify how we handle sign/basis invariance (either by using SignNet-style invariant networks, or by using an alternative PE like random-walk PE that avoids this issue). Explicitly acknowledge this in the positional encoding design discussion.

---

### Paper 11 — PGTR (CITE)

**Title:** Position-aware Graph Transformer for Recommendation

**Authors:** Jiajia Chen, Jiancan Wu, Jiawei Chen, Chongming Gao, Yong Li, Xiang Wang

**Venue & Year:** ACM Transactions on Information Systems (TOIS), 2025 (ArXiv preprint December 2024)

**ACM DL:** https://dl.acm.org/doi/full/10.1145/3757736  |  **ArXiv:** https://arxiv.org/abs/2412.18731

**Summary:** Proposes PGTR, a graph Transformer for collaborative filtering recommendation that augments GCN backbones (e.g., LightGCN) with spectral encoding, degree encoding, PageRank encoding, and type encoding to capture node position and structure in user-item interaction graphs.

**Relationship to our work:**
- **Must cite as the most directly related PE-for-recommendation paper.** PGTR applies PE to recommendation graphs — the same setting as our GPE. Key differentiators: (1) PGTR is focused on standard CF recommendation, not uplift/ITE estimation; (2) PGTR is not plug-and-play in the same sense — it modifies the full architecture. Our GPE is an add-on module to any existing graph ITE estimator. Discuss whether any of PGTR's PE choices informed our design.

---

## CATEGORY 3: Causal Inference / GNN ITE Estimation Papers

---

### Paper 12 — NetDeconf (MUST CITE as foundational baseline)

**Title:** Learning Individual Causal Effects from Networked Observational Data

**Authors:** Ruocheng Guo, Jundong Li, Huan Liu

**Venue & Year:** WSDM 2020

**DBLP:** https://dblp.org/rec/conf/wsdm/GuoLL20.html  |  **ArXiv:** https://arxiv.org/abs/1906.03485

**Summary:** Proposes NetDeconf, the pioneering framework that uses GCN to capture network structure as a proxy for hidden confounders in ITE estimation from observational data, relaxing the strong ignorability assumption.

**Relationship to our work:**
- **Must cite as the foundational GNN-based ITE estimator.** NetDeconf is the origin paper for the GNN+ITE research line. Our GPE module can be directly plugged into NetDeconf. In experiments, NetDeconf + our modules vs. NetDeconf alone is a clean ablation showing the value of structural encoding.

---

### Paper 13 — GIAL (CITE as baseline)

**Title:** Graph Infomax Adversarial Learning for Treatment Effect Estimation with Networked Observational Data

**Authors:** Zhixuan Chu, Stephen L. Rathbun, Sheng Li

**Venue & Year:** KDD 2021

**ACM DL:** https://dl.acm.org/doi/10.1145/3447548.3467302  |  **ArXiv:** https://arxiv.org/abs/2106.02881

**Summary:** Proposes GIAL, which maximizes structure mutual information to extract confounder representations from imbalanced network data, then uses adversarial learning to balance treatment/control distributions and generate counterfactual outcomes.

**Relationship to our work:**
- **Cite as an important baseline in the GNN-based ITE line.** GIAL addresses the imbalanced network structure problem specifically. Our variance-weighted learning module also addresses distributional imbalance but through a different mechanism (prediction variance as sample weights vs. adversarial balancing). Explicitly compare these approaches in related work.

---

### Paper 14 — GDC / WSDM 2025 (MUST CITE)

**Title:** Graph Disentangle Causal Model: Enhancing Causal Inference in Networked Observational Data

**Authors:** Binbin Hu, Zhicheng An, Zhengwei Wu, Ke Tu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Yufei Feng, Jiawei Chen

**Venue & Year:** WSDM 2025

**ACM DL:** https://dl.acm.org/doi/10.1145/3701551.3703525  |  **ArXiv:** https://arxiv.org/abs/2412.03913

**Summary:** Proposes GDC, which disentangles node attributes into adjustment and confounder factors with separate graph aggregators, and enforces disentanglement via a causal constraint module for improved ITE estimation on networked data.

**Relationship to our work:**
- **Must cite as a 2025 concurrent work in the same research area.** GDC addresses representation quality for ITE via disentanglement; our work addresses it via structural PE and variance-weighted training. Both are improving the same pipeline at approximately the same time. In related work, distinguish: GDC modifies the architecture's factorization; our modules add positional context and improve the training objective. Our approach is more modular (plug-and-play) and explicitly designed for the recommendation setting.

---

### Paper 15 — CauGramer / ICLR 2025 (MUST CITE)

**Title:** Causal Graph Transformer for Treatment Effect Estimation Under Unknown Interference

**Authors:** (See https://openreview.net/forum?id=foQ4AeEGG7 — "anpwu" submitter; code at github.com/anpwu/CauGramer)

**Venue & Year:** ICLR 2025

**OpenReview:** https://openreview.net/forum?id=foQ4AeEGG7

**Summary:** Proposes CauGramer, an interference-agnostic Causal Graph Transformer that uses L-order graph Transformer with cross-attention to infer the interference structure and aggregation function simultaneously, integrating confounder balancing via minimax moment constraints.

**Relationship to our work:**
- **Must cite — this is the most closely related ICLR 2025 concurrent work combining graph Transformers and causal ITE.** CauGramer uses a full graph Transformer architecture specifically for unknown interference. Our GPE is not a full Transformer replacement but a plug-and-play encoding module. Key positioning: CauGramer requires a fixed architecture (graph Transformer); our GPE drops into any existing GNN-based ITE backbone. Explicitly discuss this distinction in the introduction and related work.

---

### Paper 16 — TNDVGA (CITE)

**Title:** Disentangled Graph Autoencoder for Treatment Effect Estimation (a.k.a. TNDVGA)

**Authors:** Di Fan, Renlei Jiang, Yunhao Wen, Chuanhou Gao

**Venue & Year:** ArXiv preprint (December 2024, revised February 2025) — no confirmed venue as of search date

**ArXiv:** https://arxiv.org/abs/2412.14497

**Summary:** Proposes TNDVGA, a disentangled variational graph autoencoder that separates latent factors into instrumental, confounding, adjustment, and noisy factors (enforcing independence via HSIC) for ITE estimation on networked observational data.

**Relationship to our work:**
- **Cite to demonstrate awareness of the disentanglement-based ITE line.** Not yet at a top venue; cite if it gets accepted by submission time. Complements GDC as another disentanglement approach. Our variance-weighted learning is orthogonal — it improves training signal quality rather than factorizing latent space.

---

### Paper 17 — SPNet / TKDD 2024 (CITE)

**Title:** Modeling Interference for Individual Treatment Effect Estimation from Networked Observational Data

**Authors:** Qiang Huang, Jing Ma, Jundong Li, Ruocheng Guo, Huiyan Sun, Yi Chang

**Venue & Year:** ACM Transactions on Knowledge Discovery from Data (TKDD) 2024

**ACM DL:** https://dl.acm.org/doi/10.1145/3628449

**Summary:** Proposes SPNet, a two-channel GCN with masked-attention mechanism to capture hidden confounders and model varied magnitude of spillover/interference effects; provides a formal identifiability theorem for ITE under network interference.

**Relationship to our work:**
- **Cite as a rigorous theoretical treatment of network interference for ITE.** SPNet provides the identifiability justification that our work can leverage. Our modules (especially GPE, which encodes structural position) can be integrated into SPNet. If including SPNet as a backbone estimator in experiments, this is an important comparison point.

---

## CATEGORY 4: Variance-Weighted / Sample Weighting in Causal Inference

---

### Paper 18 — R-Learner / Nie & Wager (MUST CITE as methodological foundation)

**Title:** Quasi-Oracle Estimation of Heterogeneous Treatment Effects

**Authors:** Xinkun Nie, Stefan Wager

**Venue & Year:** Biometrika, 2021 (Volume 108, Issue 2)

**Journal:** https://academic.oup.com/biomet/article-abstract/108/2/299/5911092  |  **ArXiv:** https://arxiv.org/abs/1712.04912

**Summary:** Proposes the R-Learner, a two-step meta-learner for CATE estimation that minimizes a Robinson-decomposition-based residualized loss; shown to be equivalent to inverse-variance weighted pseudo-outcome regression, achieving quasi-oracle convergence rates.

**Relationship to our work:**
- **Must cite as the primary theoretical foundation for variance-weighted learning.** The R-Learner's connection to inverse-variance weighting (IVW) is the theoretical justification for our variance-weighted learning module. Explicitly state that our module implements a neural instantiation of the R-Learner's IVW principle within graph-based ITE estimators. The "variance weighting" in our module corresponds exactly to the IVW weights in R-Learner theory.

---

### Paper 19 — CFR / Shalit et al. (MUST CITE as ITE deep learning baseline)

**Title:** Estimating Individual Treatment Effect: Generalization Bounds and Algorithms

**Authors:** Uri Shalit, Fredrik D. Johansson, David Sontag

**Venue & Year:** ICML 2017

**Proceedings:** https://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf  |  **ArXiv:** https://arxiv.org/abs/1606.03976

**Summary:** Proposes CFRNet (Counterfactual Regression) and TARNet — deep neural networks for ITE estimation that learn balanced representations by minimizing IPM (Integral Probability Metric) distance between treated/control distributions; provides generalization bounds for ITE estimation.

**Relationship to our work:**
- **Must cite as the canonical deep learning ITE estimator.** CFR/TARNet is the standard deep learning baseline for ITE. Our variance-weighted learning module can be seen as an alternative to CFR's distribution-balancing approach: instead of minimizing distance between T/C representations, we reweight samples by prediction variance. In related work, discuss these as complementary strategies.

---

## CATEGORY 5: Plug-and-Play / Modular Enhancement for Graph Recommendation

---

### Paper 20 — Lighter-X (CITE for framing)

**Title:** Lighter-X: An Efficient and Plug-and-play Strategy for Graph-based Recommendation through Decoupled Propagation

**Authors:** Yanping Zheng et al. (7 authors)

**Venue & Year:** PVLDB (VLDB), 2025

**ACM DL:** https://dl.acm.org/doi/10.14778/3749646.3749649  |  **ArXiv:** https://arxiv.org/abs/2510.10105

**Summary:** Proposes Lighter-X, a plug-and-play efficiency module for graph-based recommendation that decouples propagation from transformation, reducing parameters to 1% of LightGCN while maintaining performance.

**Relationship to our work:**
- **Cite to validate the "plug-and-play" framing.** Lighter-X demonstrates that the recommendation community accepts and values plug-and-play modular designs for GNN-based recommenders. Use in introduction to motivate our contribution: just as Lighter-X shows plug-and-play efficiency modules are valuable, our work shows plug-and-play GPE + variance-weighted modules are valuable for the ITE estimation dimension.

---

## Summary Table

| # | Paper | Venue | Year | Category | Action |
|---|-------|-------|------|----------|--------|
| 1 | E3IR (Sun et al.) | RecSys | 2024 | Uplift + Rec | MUST CITE, Position Against |
| 2 | GNUM (Zhu et al.) | WWW | 2023 | GNN Uplift | MUST CITE, Compare |
| 3 | CDUM (Meng et al.) | RecSys | 2025 | Uplift + Rec | MUST CITE |
| 4 | GNN+CausalK (Wang et al.) | IEEE | 2024 | GNN Uplift | Cite |
| 5 | S-CIEE | TOIS | 2025 | ITE + Rec | MUST CITE |
| 6 | Uplift Tutorial (Moraes et al.) | CIKM | 2023 | Survey | Background Cite |
| 7 | Graphormer (Ying et al.) | NeurIPS | 2021 | Graph PE | MUST CITE (reviewer request) |
| 8 | GraphGPS (Rampasek et al.) | NeurIPS | 2022 | Graph PE | MUST CITE (reviewer request) |
| 9 | SAN (Kreuzer et al.) | NeurIPS | 2021 | Graph PE | MUST CITE (reviewer request) |
| 10 | SignNet (Lim et al.) | ICLR | 2023 | Graph PE | MUST CITE (reviewer request) |
| 11 | PGTR (Chen et al.) | TOIS | 2025 | PE + Rec | MUST CITE |
| 12 | NetDeconf (Guo et al.) | WSDM | 2020 | GNN ITE | MUST CITE, Baseline |
| 13 | GIAL (Chu et al.) | KDD | 2021 | GNN ITE | Cite, Baseline |
| 14 | GDC (Hu et al.) | WSDM | 2025 | GNN ITE | MUST CITE |
| 15 | CauGramer | ICLR | 2025 | Graph Transformer ITE | MUST CITE |
| 16 | TNDVGA (Fan et al.) | ArXiv | 2024 | GNN ITE | Cite |
| 17 | SPNet (Huang et al.) | TKDD | 2024 | GNN ITE | Cite |
| 18 | R-Learner (Nie & Wager) | Biometrika | 2021 | Variance Weighting | MUST CITE (theory foundation) |
| 19 | CFR/TARNet (Shalit et al.) | ICML | 2017 | Deep ITE | MUST CITE (baseline) |
| 20 | Lighter-X (Zheng et al.) | VLDB | 2025 | Plug-and-Play | Cite (framing) |

---

## Strategic Positioning Notes

### How to frame GPE (Graph Positional Encoding)
- Reviewers asked about Graphormer, GPS, SAN, SignNet. The response must be: "These methods design full graph Transformer architectures with PE for general graph tasks. Our GPE is a lightweight, plug-and-play PE module specifically designed for bipartite user-item uplift graphs, compatible with any existing GNN-based ITE estimator. We adapt the spectral PE paradigm from SAN/SignNet but address its sign-ambiguity and scalability challenges in the recommendation context."
- CauGramer (ICLR 2025) is the most similar in spirit — a graph Transformer for ITE — but requires a full architecture replacement. Emphasize modularity.

### How to frame Variance-Weighted Learning
- The theoretical anchor is the R-Learner (Nie & Wager, Biometrika 2021), which shows that variance-weighted pseudo-outcome regression is equivalent to quasi-oracle ITE estimation.
- Our module is the neural graph instantiation of this principle: instead of the R-Learner's kernel regression, we compute prediction variance over the GNN output distribution and use it to reweight training samples.
- This is more principled than GIAL's adversarial balancing and complementary to CFR's representation-matching.

### Plug-and-play framing
- Lighter-X (VLDB 2025) validates that the RecSys community accepts plug-and-play improvements to graph recommenders. Use it in the introduction.
- Frame our work as: "We contribute two modules that can wrap around any graph-based ITE estimator (NetDeconf, GIAL, GDC, GNUM) without requiring architectural changes, analogous to how plug-and-play efficiency modules (Lighter-X) improve GNN recommenders without changing the backbone."

---

*Report generated by automated literature scout. Verify all venue/year details before submission as ArXiv preprints may have been accepted at conferences between generation date and submission.*
