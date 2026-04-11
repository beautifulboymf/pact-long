# Internal Pre-Submission Review: ACM RecSys 2025

**Paper:** "Graph Positional Encoding and Variance-Weighted Learning as Universal Plug-ins for Uplift Modeling"

**Reviewer:** Internal (simulating senior RecSys reviewer with GNN + causal inference expertise)

**Date:** 2026-04-09

---

## 1. Summary

The paper proposes two modular, architecture-agnostic plug-in modules for graph-based Individual Treatment Effect (ITE) estimation: (1) Graph Positional Encoding (GPE), which fuses node features with node2vec embeddings via multi-head cross-attention before GNN processing, and (2) Variance-Weighted Learning, which estimates per-sample ITE variance and reweights the regression loss to mitigate heteroscedastic noise (with a learned log-variance head for continuous outcomes and an analytical Bernoulli variant for binary outcomes). The authors frame these as input-level and loss-level enhancements that can be "plugged in" to any existing graph ITE estimator without modifying its internal architecture. They evaluate on eight base models (BNN through GDC WSDM 2025 SOTA) across five graph datasets (semi-synthetic) and six real-world tabular uplift benchmarks, totaling 1,100+ experimental runs. Key claims: GPE improves 7/8 base models, TARNet+GPE achieves 11/11 win rate, variance weighting yields 4x Qini improvement on continuous real-world outcomes, and the combination reduces GDC PEHE by up to 31% at high noise.

This is a significant rewrite from the ICML submission (which proposed CAVIN as a monolithic architecture). The reframing as a "plug-and-play paradigm" is strategic and partially addresses the ICML novelty concerns. The addition of 8 base models and 6 real-world tabular datasets substantially strengthens the experimental case.

---

## 2. Strengths

### S1. Compelling reframing as a modular paradigm (strong contribution to RecSys community)
The shift from "here is a new model" to "here are two composable modules that improve any existing model" is genuinely valuable for the RecSys community, where practitioners need to enhance deployed systems, not replace them. The plug-in framing (GPE at input level, Var at loss level) is clean, and the paper makes this case well. This is precisely the kind of work RecSys values: practical, deployable, and compatible with existing infrastructure.

### S2. Breadth of evaluation across base models
Testing on 8 base models spanning 2016--2025 (BNN, TARNet, X-learner, NetDeconf, GIAL, GNUM, GDC, CAVIN) is exceptionally thorough for this subfield. The controlled setup -- all models share the same GNN backbone, hidden dimensions, and training protocol -- makes the comparison fair. The win rate analysis (Table 3) is a simple but effective summary. This directly addresses the ICML criticism of limited baselines.

### S3. Real-world tabular validation
Including six real-world tabular benchmarks (Criteo, Hillstrom, RetailHero, Lenta, X5) for the variance weighting component is a strong addition. The Hillstrom spend result (4x Qini improvement) is striking and grounded in a real dataset that RecSys reviewers know and trust. The honest reporting of no-improvement cases (Criteo, binary outcomes) builds credibility.

### S4. Candid and detailed limitations section
Section 6 explicitly acknowledges the missing graph transformer comparison, the missing alternative PE ablation, the O(n^2) scalability concern, and the semi-synthetic reliance. This level of honesty is unusual and is generally received positively by reviewers, provided the remaining contribution is strong enough.

### S5. Practical guidelines (Section 5.1)
The "when to use GPE" / "when to use variance weighting" / "when to combine both" discussion in Section 5.1 is excellent for a RecSys audience. The insight that BNN (S-learner) does not benefit from GPE because it lacks separate treatment/control heads is a genuinely useful finding for practitioners.

### S6. Addressing the theory gap transparently
Remark 1 (Section 4.2, lines ~161--163) now explicitly states that the Gauss-Markov analysis provides "theoretical motivation rather than a formal guarantee" and defers to empirical evidence. This is a direct and mature response to the ICML criticism about theory for linear case vs. nonlinear model. Much better than overclaiming.

---

## 3. Weaknesses

### W1. (Major) Semi-synthetic graph evaluation still dominates -- and may still favor the proposed method
**Issue:** Although the tabular datasets are real-world, ALL graph-based evaluations remain semi-synthetic. Three of the five graph datasets (CoraFull, DBLP, PubMed) use the authors' own DGP with community-aware features (Louvain detection, intra-community edge boosting, multi-hop spillover), while BlogCatalog and Flickr use Guo et al.'s older protocol. The custom DGP injects exactly the signals that GPE is designed to capture: community structure (which node2vec encodes) and multi-hop treatment spillover (which 5-layer GAT aggregates).

**Why it matters:** RecSys reviewers are increasingly skeptical of semi-synthetic-only graph evaluations. The ICML reviewers raised this exact point (Reviewer ZqhE: "semi-synthetic datasets are generated based on the paper's specific assumptions... which might favor CAVIN over baselines"), and it was echoed at IJCNN (Reviewers 1 and 3). Despite adding tabular benchmarks, the core GPE claim remains validated only on data generated to have community structure.

**Suggestion:** (a) Include at least one graph dataset with a real outcome. Candidates: the LaLonde-Dehejia-Wahba (Jobs) dataset with a constructed social network, or an A/B test log from a social platform with friend-graph structure. Even a proxy evaluation (e.g., predicting with-held conversions on a social graph where treatment is known) would help. (b) Alternatively, test GPE on an *adversarial* DGP where community structure does NOT drive treatment effect heterogeneity, and show graceful degradation rather than harm. (c) At minimum, test on a DGP from another paper (e.g., Chen et al. 2024 or Forastiere et al. 2021's DGP) to demonstrate robustness to DGP choice.

### W2. (Major) The "4x Qini improvement" headline claim is fragile
**Issue:** The 4x Qini claim (0.029 to 0.117 on Hillstrom spend) is the strongest real-world result, but it comes from a single dataset-outcome combination, with large standard deviations (0.065 and 0.076 respectively). The 95% confidence intervals overlap substantially: [~-0.10, 0.16] for S vs. [~0.04, 0.19] for S+Var. On every other dataset, variance weighting provides marginal or zero improvement (and hurts on Criteo).

**Why it matters:** RecSys reviewers will scrutinize headline claims. A 4x improvement that is not statistically significant by conventional standards (overlapping CIs) is problematic. Making this the lead claim in the abstract and throughout the paper invites skepticism.

**Suggestion:** (a) Report p-values or confidence intervals for the Hillstrom comparison. (b) Run more seeds (currently 5; increase to 10+) to tighten the CI. (c) Tone down the "4x" framing in the abstract; instead say "substantial improvement" and provide the actual numbers with CIs. (d) If the result IS significant, make the statistical test explicit. (e) Show the Hillstrom result is not due to a single outlier seed by reporting all 5 individual Qini values.

### W3. (Major) Missing statistical rigor throughout
**Issue:** No error bars, confidence intervals, or significance tests are reported for the graph experiments (Table 2). The paper says "mean over 3 seeds" but does not report standard deviations. Three seeds is also a very small number for drawing conclusions about "69% of all graph settings."

**Why it matters:** Without variance estimates, it is impossible to assess whether the observed improvements (e.g., GDC DBLP rho=5: 3.12 to 2.60) are statistically meaningful or within noise. RecSys 2025 reviewing guidelines explicitly ask for statistical significance assessment. The ICML and IJCNN reviews did not raise this as forcefully as they should have, but RecSys reviewers will.

**Suggestion:** (a) Report mean +/- std for ALL tables, or at minimum for Table 2 (the main result). (b) Increase seeds from 3 to 5 for graph experiments. (c) Run paired t-tests or Wilcoxon signed-rank tests for the "win rate" claims. (d) For the 69% improvement rate claim in the abstract, provide a binomial test or similar.

### W4. (Major) Cross-attention in GPE has O(n^2) complexity -- ignored in body, only in limitations
**Issue:** GPE uses cross-attention where features are queries and ALL nodes' positional embeddings are keys/values (Eq. 5). This is O(n^2) in the number of nodes. For CoraFull (19,793 nodes), this means a 19,793 x 19,793 attention matrix per head. The paper acknowledges this in Section 6 (Limitations) but does not report any runtime numbers, memory usage, or scalability analysis.

**Why it matters:** RecSys systems operate at scales of millions to billions of users. A method that cannot scale beyond 20K nodes is of limited practical interest to the RecSys community. Reviewers will immediately flag this.

**Suggestion:** (a) Report wall-clock training time per epoch for each base model with and without GPE. (b) Report GPU memory consumption. (c) Discuss mini-batch alternatives (GraphSAINT, ClusterGCN-style subgraph sampling) and whether GPE can be applied to sampled subgraphs. (d) If O(n^2) is truly the bottleneck, consider replacing full cross-attention with local cross-attention (only attend to k-nearest neighbors' positional embeddings) and test whether performance degrades.

### W5. (Moderate) Missing alternative PE comparison remains a gap
**Issue:** Despite extensive discussion in Section 5.2 about why node2vec was chosen over Laplacian PE and RWPE, there is no empirical comparison. Section 6 acknowledges this as a limitation, but it was a core ICML criticism (Reviewer dSf9, Reviewer 1 at IJCNN) and remains unaddressed experimentally.

**Why it matters:** Without this ablation, the contribution of GPE is confounded with the choice of node2vec. If Laplacian PE achieves similar gains with less preprocessing cost, the GPE design is less compelling. If RWPE (which is also random-walk based) performs equivalently, the node2vec-specific design choices are irrelevant. RecSys reviewers who know the graph transformer literature will flag this.

**Suggestion:** Run the GPE module with three PE methods (node2vec, Laplacian eigenvectors, RWPE) on at least one dataset (e.g., DBLP) at one noise level (e.g., rho=10). This is a small experiment that would substantially strengthen the paper.

### W6. (Moderate) The tabular experiments test only S-learner, not the full base model suite
**Issue:** Table 5 only compares S-learner (BNN) with and without Bernoulli variance weighting on tabular data. But GPE's best interaction was with TARNet and GDC. Why not test T-learner, X-learner, or DR-learner variants with variance weighting on tabular data?

**Why it matters:** The paper claims variance weighting is a "universal" plug-in, but the tabular validation tests only one base learner. This limits the generalizability claim. RecSys practitioners using T-learner or DR-learner pipelines cannot know whether variance weighting helps their specific setup.

**Suggestion:** Add columns for at least T-learner and X-learner (even without GPE) on the tabular benchmarks. If the results are consistently positive, it strengthens the "universal" claim. If model-dependent, that is also informative.

### W7. (Moderate) Inconsistency between Algorithm 1 and the method description
**Issue:** Algorithm 1 (line 9) says "Update sigma^2_t by regressing on D_t^2 - tau_t^2" but Section 4.2 (line ~153) says the variance head predicts "log sigma^2" with target "log((D - tau_hat)^2 + epsilon)". These are different formulations. The algorithm shows the raw squared residual target; the text describes a log-transformed target. Which is actually implemented?

**Why it matters:** Reproducibility is critical. If a reader follows Algorithm 1, they get a different model than if they follow the text. This exact inconsistency was flagged by ICML Reviewer jpBE ("the sample weight used in Algorithm 1 is different from that in Eq. (21)") and has not been fully resolved.

**Suggestion:** Harmonize Algorithm 1 with the method description. If the implementation uses log-parameterization (as described in text and as better practice), update Algorithm 1 to show `log_sigma^2 <- regress on log((D_t - tau_hat)^2 + epsilon)`.

### W8. (Moderate) GDC+GPE degrades at high noise -- undermining the "universal" claim
**Issue:** At rho=30, GDC+GPE performs WORSE than vanilla GDC on 2 of 3 datasets (CoraFull: 4.93->5.90, PubMed: 4.31->4.80). The paper addresses this by noting that adding Var rescues the degradation, but the standalone GPE claim of "universal improvement" is contradicted. The 69% claim in the abstract should not mask a 31% failure rate that includes failures on the SOTA model.

**Why it matters:** If GPE can hurt the best available model under realistic noise conditions, a practitioner needs clear guidance on when NOT to use it. The paper provides this in Section 5.1 but the abstract overclaims.

**Suggestion:** (a) Soften the abstract language from "GPE improves PEHE in 69% of all graph settings" to something like "GPE improves most base models in most settings, with the strongest and most consistent gains at low-to-moderate noise." (b) Frame the high-noise degradation not as a limitation to be rescued but as a finding about the noise-sensitivity of attention-based feature fusion. (c) Consider a simple heuristic (e.g., validation-based selection of whether to use GPE) and report results with this selection rule.

### W9. (Minor-Moderate) Identification under interference is underspecified
**Issue:** Assumptions 1--3 (Section 3.1) state Network Consistency, Unconfoundedness, and Overlap, but the exposure mapping "agg" in Assumption 1 is never concretely defined. The paper says Y_i = Y_i(T_i, T_i) where T_i = agg(T_{N_i}), but does not specify what "agg" is (mean? sum? presence-based?). Meanwhile, the GAT is implicitly learning this mapping.

**Why it matters:** Without a concrete exposure mapping, the identification argument is incomplete. Theorem 1 (Optimization Equivalence) assumes E[D|X] = tau(X), which is a standard X-learner property, but under interference this requires the exposure mapping to be correctly specified or learned. ICML Reviewer 1 (IJCNN) raised this explicitly.

**Suggestion:** (a) Specify the exposure mapping used in the DGP (it appears to be a weighted sum with decay gamma^r per hop). (b) Briefly discuss whether the GAT architecture can recover this mapping. (c) Acknowledge that Theorem 1 holds only if the base model correctly captures the interference structure.

### W10. (Minor) The "CAVIN" name persists confusingly
**Issue:** The paper was previously titled around "CAVIN" as a monolithic architecture. In the rewrite, CAVIN appears in Table 2's base model list as "CAVIN (X-learner + GPE + Var)" (line 251), in the code URL, and implicitly throughout. But the paper's framing is now about two plug-in modules, not a model called CAVIN. This creates confusion: is CAVIN a model or a framework? The answer seems to be that CAVIN is one specific configuration (X-learner + both plug-ins), but this is not clearly stated.

**Suggestion:** Either (a) drop the CAVIN name entirely and refer to specific configurations as "X-learner+GPE+Var", or (b) define CAVIN explicitly early on as "the specific configuration where both plug-ins are applied to the X-learner base model."

---

## 4. Minor Issues

### M1. Notation inconsistency
- Line 104: T_i (script) is used for the neighbor treatment aggregation, but T_i is also the individual treatment indicator. These share the same letter T with different decorations (calligraphic T vs. italic T), which is confusing. Use a distinct symbol like E_i (for exposure) for the aggregated neighbor treatment.
- Line 115: tau(X_i) conditions on X_i and X_{N}^i, but the notation drops the neighbor features from the conditioning set in subsequent uses.

### M2. Table 2 formatting
- "Underlined values indicate the model failed to converge" -- only the caption mentions this, but no values in the table appear underlined. Either the underlines were lost in formatting, or no model failed to converge in the reported results. Clarify.

### M3. Missing RetailHero from Table 5
- Table 1 lists RetailHero as a tabular dataset, but Table 5 does not report results for it. Was it tested? If excluded, explain why.

### M4. BibTeX issues
- hudgens2008toward uses @inproceedings but is a journal article (JASA). Same for athey2016recursive (PNAS is not a conference).
- velickovic2017graph lists journal as "stat" volume 1050 -- this is a well-known arXiv preprint that was published at ICLR 2018. Should cite the ICLR version.
- chen2024doubly uses @inproceedings but cites an arXiv preprint.

### M5. Figure 2 description
- The caption says "Propensity head" but the method section does not discuss how propensity is used. Is propensity only used for the X-learner combination weights? Clarify the role of the propensity head in the plug-in framework.

### M6. Missing page count check
- RecSys 2025 allows 8 pages of content + 2 pages of references. The current paper appears close to the limit. Verify that the compiled version fits within the page budget.

---

## 5. Questions for Authors (Anticipated Reviewer Questions)

**Q1.** How does GPE scale to real recommendation graphs with millions of nodes? The O(n^2) cross-attention is prohibitive. Have you considered sparse or local attention alternatives?

**Q2.** The DGP uses Louvain communities to generate both treatment spillover and outcome heterogeneity. GPE uses node2vec which captures community structure. Isn't this a self-fulfilling prophecy? Can you demonstrate GPE's value on a DGP that does NOT explicitly inject community effects?

**Q3.** On Table 5, why does variance weighting hurt on Criteo (0.167 -> 0.151)? You attribute this to balanced treatment assignment, but this is an unsatisfying explanation -- the Bernoulli variance should still capture meaningful heterogeneity in predicted probabilities. Could this indicate that the Bernoulli variance formula is too simplistic?

**Q4.** You report 3 seeds for graph experiments. Given that improvements are often small (e.g., DBLP rho=10: 3.19 -> 2.70 for GDC+GPE, but what is the std?), how can reviewers be confident these are real effects rather than noise?

**Q5.** The paper claims GPE captures "community-level structural information." Can you provide empirical evidence for this? For instance, do the learned attention weights in GPE correlate with community membership? A visualization of the attention matrix on a small graph would be informative.

**Q6.** For the variance head: how sensitive is performance to the choice of delta (the variance floor)? You mention 0.01 was selected, but a sensitivity analysis would help practitioners.

**Q7.** The X-learner requires two stages of training. How does the variance weighting interact with the cross-fitting? Specifically, is the variance estimated from the same fold that is used for the final ITE regression, or is there proper sample splitting?

**Q8.** The paper evaluates on citation networks (CoraFull, DBLP, PubMed). These are not social networks and do not naturally exhibit the kind of treatment interference described in the introduction (targeted promotions, peer effects). Why not use actual social network datasets (e.g., Epinions, Ciao, or synthetic social graphs with realistic degree distributions)?

---

## 6. Writing Quality Assessment

### Overall: B+ (Good, with room for improvement)

**Strengths in writing:**
- The introduction clearly motivates the problem with real-world stakes (hundreds of billions in targeted incentives).
- The plug-in framing is consistently maintained throughout.
- Section 5.1 (Practical Guidelines) is a model of clarity for RecSys audiences.
- The Limitations section (Section 6) is admirably honest.

**Areas for improvement:**
- The abstract is dense and tries to pack too many numbers. Consider leading with the conceptual contribution (modular enhancement paradigm) and deferring specific numbers to the body.
- Section 4.2 (Variance-Weighted Learning) mixes implementation details (log-parameterization, gradient detachment) with theoretical motivation (Remark 1). Consider restructuring: theory first, then implementation.
- The Discussion section (Section 5) is unusually long for a RecSys paper and reads partly as a rebuttal to prior reviews rather than a standalone discussion. Trim Section 5.2 (Design Choices) to focus on the most impactful choices; move the rest to supplementary material.
- Several sentences in the introduction are very long (line 69 is one sentence spanning 5 lines). Break these up.

---

## 7. Assessment of ICML Concern Resolution

| ICML Concern | Addressed? | Assessment |
|---|---|---|
| 1. "Just combining existing ideas" | Partially | Reframing as plug-ins is better, but the individual modules (cross-attention fusion, WLS weighting) are still standard techniques applied to a new domain. The contribution is in the systematic evaluation, not the modules themselves. |
| 2. Theory only for linear case | Yes | Remark 1 now explicitly acknowledges this limitation and reframes the theory as motivation. Honest and appropriate. |
| 3. Semi-synthetic data may favor method | Partially | Tabular real-world data added for Var, but GPE still validated only on semi-synthetic graphs. Core concern remains. |
| 4. Missing graph transformer baselines | No | Still missing. Acknowledged in Section 6 as future work. RecSys reviewers may be less concerned about this than ICML, but it remains a gap. |
| 5. Missing alternative PE comparison | No | Still missing experimentally. Discussed qualitatively in Section 5.2 but no empirical data. |

---

## 8. Overall Recommendation

### Score: 5.5/10 (Borderline -- between Weak Reject and Weak Accept)

**Calibrated to RecSys:** This is a solid engineering contribution with practical value. The modular paradigm is the right framing for RecSys. The breadth of evaluation is impressive. However, three issues hold it back from a clear accept:

1. **The core GPE claim lacks real-world validation on graphs.** Every graph experiment is semi-synthetic, and the DGP is designed to have the exact structure GPE captures. Until GPE is shown to help on a real graph outcome (or at least an adversarial DGP), the contribution feels circular.

2. **Statistical rigor is insufficient.** No error bars on the main results table, only 3 seeds, and the headline 4x Qini claim has overlapping confidence intervals. RecSys 2025 requires proper statistical reporting.

3. **Scalability is unaddressed.** O(n^2) cross-attention on nodes is a non-starter for RecSys-scale systems, and no runtime analysis is provided.

**What would tip this to Accept:**
- Add std devs to Table 2 (even with 3 seeds)
- Add one real-world graph experiment (even a proxy task)
- Add runtime/memory analysis showing the overhead of GPE is manageable
- Tighten the 4x Qini claim with proper statistical testing
- Add a Laplacian PE vs. node2vec ablation on one dataset

**What could tip this to Reject:**
- If reviewers focus on the fact that both modules (cross-attention fusion and WLS) are well-established techniques applied without substantial adaptation
- If the O(n^2) scalability concern is deemed fatal for a RecSys venue
- If the semi-synthetic evaluation is viewed as fundamentally unable to validate the GPE claim

---

## 9. Rebuttal Strategy (If Weaknesses Must Be Addressed in Rebuttal)

If time permits experimental additions before submission, prioritize in this order:

1. **[2 days]** Add std devs to all tables. Run 2 more seeds on graph experiments (3 -> 5 seeds). Add paired significance tests for win-rate claims.

2. **[2 days]** Run Laplacian PE and RWPE variants of GPE on DBLP at rho=10. Report in a small ablation table.

3. **[1 day]** Add runtime analysis: wall-clock per epoch and peak GPU memory for each base model +/- GPE. Show the overhead is reasonable for the datasets tested.

4. **[3 days]** Find or construct one real-world graph evaluation. Options:
   - Use the Jobs dataset with a k-NN feature graph (common in causal GNN papers).
   - Use an e-commerce dataset with user co-purchase graph and real conversion outcomes.
   - Run GPE on a DGP from another paper (Forastiere et al. 2021) to show DGP robustness.

5. **[1 day]** Harmonize Algorithm 1 with the text. Fix BibTeX. Add RetailHero to Table 5.

6. **[1 day]** Trim Section 5.2 to half its current length. Move detailed design rationales to supplementary.

---

## 10. Comparison with Prior Review Cycles

**vs. ICML (4x Weak Reject):** The paper has improved substantially. The plug-in reframing, 8 base models, and 6 real-world tabular datasets address the core novelty and evaluation concerns. However, the two most concrete ICML requests (alternative PE comparison, graph transformer baselines) remain unaddressed experimentally. If ICML reviewers were at RecSys, they would likely upgrade to borderline.

**vs. IJCNN (Weak Accept):** The IJCNN version was weaker (monolithic CAVIN, 3 graph datasets only). The current version is substantially stronger. However, IJCNN Reviewer 1's detailed technical concerns (identification under interference, variance target negativity, cross-fitting details) are only partially addressed.

**Net assessment:** The paper is above IJCNN quality but below ICML standards. For RecSys, it is borderline. The practical framing and breadth of evaluation are strengths; the lack of real-world graph validation and statistical rigor are the main risks.
