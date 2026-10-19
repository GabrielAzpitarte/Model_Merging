# PRIOR_WORK_02: prior-work verification for the surviving "uncertainty-gated merge correction" idea

Date: 2026-10-19. Written by a literature-verification pass (no experiments run, no other project files touched; downloaded papers were kept in a scratch directory).

Scope of the question: (i) is the professor's rule `lambda_i[p] = softmax_i(-beta * kappa(W_i[p]))` in prior work; (ii) is a "kappa-capped ridge" for the RegMean solve prior art; (iii) is "empirical joint (bootstrap) uncertainty of the calibration-derived merge correction `F = B G^{-1}`, used to gate/shrink it toward the anchor `R`" prior art.

## 0. How this was done, and versions read

I read primary text myself (curl of the arXiv HTML/PDF, `pdftotext`, then grep plus reading of the method sections and appendices). I used the web-fetch summarizer only for a first look, and it was wrong at least once (it claimed ReForge had no earlier title; the arXiv metadata shows it did), so nothing below rests on a summarizer.

| Paper | arXiv id | Version(s) read | Date of that version | Read how |
|---|---|---|---|---|
| SWUDI, "Closed-Form Spectral Regularization for Multi-Task Model Merging" (Wei, Cheng, Zhang, Shen, Yuan, Cui, Tao) | 2606.07289 | v1 (only version) | 5 Jun 2026 | HTML + PDF; main text and the appendix parts cited below (A-D notation/diagnostics, B-A to B-E, C-B1) read; the assumption lists in App. B-F were not read |
| ReForge, formerly "Bayesian Model Merging" (BMM) (Li, Han, Su, Ji) | 2605.12843 | v3 (latest); v1 main text + appendices B, F; v2 exists, not read separately | v1 13 May 2026 (title "Bayesian Model Merging"); v2 1 Oct 2026 and v3 2 Oct 2026 (title "ReForge: Refining Merged Models with Anchor-Regularized Regression") | HTML + PDF; v3 main text and App. A, B, D read, App. C and E-G skimmed |
| ACE-Merging (Xu, Wu, Lin, Huang, Zhu, Shu, Qin), CVPR 2026 main | 2603.02945 | v2 (latest) | 8 Apr 2026 (v1 3 Mar 2026) | HTML + PDF; main text and App. A.3, B.1 read |
| RegMean++ (Nguyen, Huu-Tien, Suzuki, Nguyen), TMLR | 2508.03121 | v3 (latest) | 25 Apr 2026 (v1 5 Aug 2025, v2 11 Dec 2025) | HTML + PDF; main text and App. C.3, D.9, E read |
| IterIS (Chen, Wang, Li, Zhu, Chen), CVPR 2025 | 2411.15231 | v2 (latest); v1 PDF downloaded, not diffed | 15 Apr 2025 (v1 21 Nov 2024) | HTML + PDF; main text and App. A, C read |
| SLOA, "Training-Free Transformer Merging via Sequential Local Operator Alignment" (Maurya, Lin, Jegelka, Stich, Mulayoff) | 2610.06415 | v1 (only version) | 5 Oct 2026 | HTML + PDF; main text and App. C, D.3, F.3-F.8 read |
| RegMean, "Dataless Knowledge Fusion by Merging Weights of Language Models" (Jin et al.), ICLR 2023 | 2212.09849 | v6 (latest) | 21 May 2025 (v1 19 Dec 2022) | PDF, incl. appendices |

Notation used below, matching the project's: for one module, `D_i` expert updates, `R` the anchor (uniform average), `G_i` an input second-moment ("Gram") matrix, `G = sum_i G_i`, `B = sum_i (D_i - R) G_i`, `F = B G^{-1}`, RegMean `D_RM = R + F`. A useful identity used repeatedly: for ANY anchor `R` and any positive-definite `G`, `(sum_i D_i G_i) G^{-1} = R + [sum_i (D_i - R) G_i] G^{-1}`. So "ridge toward an anchor", "RegMean with modified Grams" and "filtered inverse of a correction" are all the same algebra.

---

## A. Mandatory references

### A.0 Original RegMean (Jin et al., ICLR 2023), arXiv 2212.09849 v6

- Exact method (row-vector convention, `X_i` is N_i x m, `W_i` is m x n): minimize `sum_i ||X_i W - X_i W_i||^2`; closed form `W_M = (sum_i X_i^T X_i)^{-1} sum_i (X_i^T X_i W_i)` (Eq. 2). For transformers (Alg. 1) each Gram is replaced by `G~_i = alpha G_i + (1 - alpha) diag(G_i)` and `W_M = (sum G~_i)^{-1} sum G~_i W_i`; non-linear-layer weights are averaged.
- alpha: off-diagonal entries are multiplied "with a scalar α (set as 0.9 most of the times)"; in the experiments alpha = 0.9 except T5-base where it is 0.1; Gram computed from at most 1,000 training batches of size 16.
- Stated rationale (quote): "We empirically find that directly applying Eq. 2 for merging yields degenerated models in case of some pre-trained LM architectures." Interpretation given: it "corresponds to adding a regularization term ... that penalizes the Euclidean distance between the merged weights W_M and individual model weights" (Appendix A: `Lambda_i = gamma diag(X_i^T X_i)`, `alpha = 1/(1+gamma)`). Section 5.3: with alpha = 1.0 RegMean "yields degenerated performance on T5-base and DeBERTa"; "mostly stable between alpha = 0.1 and 0.9, but suddenly drops at alpha = 1.0".
- A scan of the full text finds NO mention of invertibility, singularity, rank or condition number. The "ill-conditioned inverse" reading is later authors' (MaTS, IterIS, RegMean++), not Jin et al.'s own words.
- Appendix C relevant to point 2: Table 8 compares "adding a constant to diagonals" (best beta = 0.2, in-domain F1 39.13) against "relative scaling of non-diagonals" (alpha = 0.1, 40.32) on T5-base emotion tasks, and says the gap "may be caused by differences in the scale of inputs in different layers, models, and datasets, which makes it difficult to find a single additive regularizer". Table 6: performance "improves as we increase N, but ... soon saturates around N = 100" (batches of 16).
- Relation to the project: this is the baseline the kappa-capped ridge competes with. Note Jin et al. already tried a plain additive ridge and flagged the scale problem; a kappa-defined ridge is scale-free (rho is set relative to the module's own spectrum), so it answers that specific critique, but only a fixed-constant additive ridge was tested in the paper.
- Does it do point 3? Not at all. No uncertainty, no resampling, no per-module gate.

### A.1 SWUDI / SWUDI-A, arXiv 2606.07289 v1

- Exact method. Per linear layer, task vectors `tau_i = W_i - W_0`. Define `A_i = tau_i^T tau_i / ||tau_i||_F^2`, `C = sum_i A_i`, `D = sum_i tau_i A_i`. WUDI loss `L(tau) = sum_i ||(tau - tau_i) tau_i^T||_F^2 / ||tau_i||_F^2 = tr(tau C tau^T) - 2 tr(tau D^T) + const`, so the stationary condition is the normal equation `tau C = D` with min-norm solution `tau_cf = D C^+`. Noise model: `D = tau0 C + E`, and per eigendirection `y_k = D q_k = lambda_k tau0_k + xi_k`, so `tau_cf,k = tau0_k + xi_k / lambda_k` (small `lambda_k` amplify proxy noise `xi_k`). Unified filtered estimator (Eq. 11): `tau_hat_h = tau_init + (D - tau_init C) C_h^+`, `C_h^+ = Q diag(h_k / lambda_k) Q^T`. SWUDI: `h_k = 1[k <= K] * (1 - exp(-t lambda_k))`, `K = ceil(r d_i)` (the exponential factor is exactly gradient flow, Prop. 2). SWUDI-A: `h_k = 1[k <= K_l]` with `K_l = ceil((sum_k sqrt(lambda_k))^2 / sum_k lambda_k)` (participation-square-root rule; CLIP, Flan-T5, MLLM) or a Gavish-Donoho count `|{k : sqrt(lambda_k) > omega_GD(beta) * median_j sqrt(lambda_j)}|` (Llama-3.2-3B). Algorithm 1 uses `tau_init = sum_i tau_i`. Grids: global scale `s in {0.1,0.2,0.3,0.5,1.0}`, `r in {0.55,...,0.85}`, `t in {300,...,1800}`; the text does not say on which split they are chosen (no "validation" or "held-out" anywhere).
- In project notation: `G_i = A_i` (a weights-only, task-vector Gram), `G = C`, `B = D - tau_init C = sum_i (tau_i - tau_init) A_i`, merged `= tau_init + B C_h^+`. So SWUDI is structurally "anchor + filtered-inverse correction". The paper says the filtered estimator "shrinks the residual rather than the absolute estimate, so discarded directions retain the initial point tau_init" (App. B-C).
- Statistics used: weights only. Task vectors stand in for activations ("the task vectors themselves serve as a proxy for hidden activations"). Calibration data appear only in a diagnostic (Table XIII, `I_hat(tau)` "calibration-based estimate of real layer-wise interference"); AdaMerging comparisons use unlabeled test data separately.
- Condition numbers: not used as a rule. Diagnostic only: Table XVII reports the median of `lambda_max / lambda_med` per architecture (85 CLIP-B/32; 1.3e8 Flan-T5 LoRA; 80 Llama-3.2-3B; 4.2e6 Qwen2-VL-7B LoRA; 73 InternVL2.5-1B) and `lambda_90 / lambda_max`. Text: "larger backbones yield better-conditioned task-vector spectra, meaning division by small lambda_k causes less degradation" (closed form vs SWUDI-A gap -3.20 / -1.45 / -0.83 pt on B/32, B/16, L/14; -3.08 on TALL20).
- Spectral filtering / regularization: this is the paper. Tikhonov appears in one sentence only: "Tikhonov regularization corresponds to the classical filter h_k = lambda_k/(lambda_k + alpha)" (Sec. III-B4); there is no Tikhonov experiment. Key evidence: closed-form `D C^+` gets 82.33 on CLIP-B/32 TA8 vs 84.63 for 300 Adam steps and 84.82 at 500 steps ("unimodal pattern ... the empirical signature that early stopping regularizes"); SWUDI 85.55, SWUDI-A 85.53.
- Uncertainty / resampling / Bayes: none in the method. A per-layer noise power law `nu_k^2 ~ sigma0^2 + sigma1^2 lambda_k^alpha` (App. B-C, Eq. 17; mean alpha 1.871 over 72 CLIP layers) is fitted for diagnostics. Fig. 8 caption is a direct caution for any noise-model-driven gate: "the Wiener and pseudoinverse filters attain the lowest predicted risk, yet they are not the best on real merged accuracy, so this risk model is a negative diagnostic and not a selection criterion". UQ-Merge is cited only in related work.
- Relation to points 1-3: (1) no kappa weighting. (2) The nearest published treatment of "regularize the inversion of the Gram operator in a merge", but for a weights-only proxy Gram, with exponential/top-K filters, and no kappa-capped Tikhonov. (3) Not at all in the uncertainty sense; there is a deterministic, eigenvalue-dependent version of "shrink the correction toward the anchor along weakly supported directions".

### A.2 ReForge (v2/v3) = "Bayesian Model Merging" (v1), arXiv 2605.12843

- Titles/versions: v1 (13 May 2026) "Bayesian Model Merging" (BMM); v2 (1 Oct 2026) and v3 (2 Oct 2026) "ReForge: Refining Merged Models with Anchor-Regularized Regression". Confirmed from the arXiv citation metadata of each version.
- Exact method (v3, module-wise, `W_merged = W_pre + s U`). Per task t, activations `x_n^(t)` collected through the task's expert; outputs `y_n^(t) = U^(t) x_n^(t)` with `U^(t) = W^(t) - W_pre`; each task is rescaled by `c_t^{-1/2}`, `c_t = sum_n ||x_n^(t)||^2`, i.e. unit-trace Grams. Likelihood `Y = U X + E`, `E_col ~ N(0, beta^{-1} I)`; prior `U ~ N(U0, alpha^{-1} I)` (element-wise) with anchor `U0 = W_anchor - W_pre` (TA, TIES, RegMean, TSV, WUDI, ISO-CTS, or 0). MAP: `U_MAP = argmin ||Y - U X||_F^2 + lam ||U - U0||_F^2`, `lam = alpha/beta`, so `U_MAP = (Y X^T + lam U0)(X X^T + lam I)^{-1}` (Eq. 8), "equivalent to anchor-centered ridge regression". By the identity in Sec. 0 this is `U0 + B (G + lam I)^{-1}`, and the paper writes this itself in App. A (Eq. 17): `Z_q(lam) = (C_q - U0 G_q)(G_q + lam I)^{-1}`, "the anchor-centered correction of the corresponding ridge solution".
- Data-free variant (Eq. 13-14): `G_t -> B_t = U^(t)T U^(t) / ||U^(t)||_F^2`, `U_DF = (sum_t U^(t) B_t + lam U0)(sum_t B_t + lam I)^{-1}` (the WUDI interference objective). Few-shot variant (Eq. 15): Gram mixing `G_mix(rho) = (1 - rho) B_t + rho G_act,t` with `rho` chosen by BO (v1 App. F: `G_mix(eps) = eps G_few + (1 - eps) G_df`, motivated by "the 1-shot or 5-shot empirical estimates are often noisy and rank-deficient"). Table 4: ISO-CTS 20 tasks, one-shot mixing 82.8 vs one-shot activation-only 80.5, equal to the 128-sample ReForge-DA 82.8.
- Hyperparameters: outer-loop Gaussian-process Bayesian optimization (EI) over `lam` and scale `s`, tied across four module groups (attn-in QKV, attn-out, MLP-in, MLP-out) times B blocks (B = 3 ViT, 1 Llama), score = mean accuracy on a labelled held-out validation set; K = 200 (ViT) / 100 (Llama) trials; `lam in [1e-4, 1]` (ViT), `[1e-3, 100]` (Llama); 128 (ViT) / 1,000 (Llama) unlabeled calibration samples per task. Ablation (20-task, TSV anchor, data-free): shared-lam grid 77.9, random search 79.0, BO 79.7.
- Statistics: activations (data-assisted) or task-vector Grams (data-free); labelled validation data in the outer loop.
- Condition numbers: none (zero hits). Spectral: only as a computational device in App. A (float64 eigendecompositions, ridge-filtered operator `F_lam(G) = G (G + lam I)^{-1}`, top-32 subspace overlap) to compare task-vector Grams with random-PSD/isotropic controls (task-vector Grams give +0.81 to +1.40 pt over random-PSD and +0.81 to +1.29 pt over isotropic Grams in ReForge-DF, Sec. 5.3).
- Uncertainty / Bayes: yes, but not for gating. The Gaussian posterior has row covariance `Sigma_row = beta^{-1} (X X^T + lam I)^{-1}`; "We can sample module weights from this posterior and average predictions ... to improve uncertainty calibration". App. D: only the final MLP output matrix is sampled (S = 10 samples), compared with iid Gaussian noise on accuracy-ECE Pareto fronts, ViT-B/32 8 and 20 tasks, ISO-CTS anchor; "Varying beta with alpha_m = beta lam_m* changes the posterior covariance without changing the MAP estimate". v1 phrased the outer loop as maximizing "predictive evidence" (marginalizing latent task vectors) but, via an explicit "empirical-Bayes approximation" (point estimate in place of the posterior) and a validation-accuracy utility, implemented it as GP-BO on validation accuracy; v3 dropped that framing and the v1 Theorem 1 (cos_F alignment, empirical cos > 0.26) in favour of citing ACE-Merging's Theorem 1.
- Relation to points 1-3: (1) none. (2) Closest published "ridge toward an anchor" for the RegMean-type solve; scale handled by trace normalization; `lam` selected by validation-accuracy BO, not by conditioning. (3) PARTIAL in structure only. The MAP IS an anchor-shrunk correction and a posterior covariance exists, but (a) the shrinkage strength `lam` is a tuned hyperparameter, (b) the posterior is conditional on `X` with one shared noise precision `beta` (i.i.d. Gaussian), (c) it is used only for predictive calibration of one layer, and (d) nothing resamples calibration documents/images or propagates joint (B, G) uncertainty to decide how much correction to trust. So it does not already do point 3, and it is the closest thing to it that I found in the merging literature.

### A.3 ACE-Merging, arXiv 2603.02945 v2 (CVPR 2026)

- Exact method (data-free, per layer). Local linearization gives `W_bar = (sum_t W_t Sigma_t)(sum_t Sigma_t)^{-1}`, `Sigma_t = E[x x^T]` (Eq. 3). Theorem 1: expected gradient Gram is proportional to `Sigma_t`, so `Sigma_hat_t ~ W~_t^T W~_t` with row-centered task vector `W~_t = Delta W_t - 1 mu_t^T`. Heterogeneity `gamma = Var_t[log ||Delta W_t||_F^2] / (E_t[log ||Delta W_t||_F^2])^2`. If `gamma > tau` (0.3): trace-normalize `Sigma_scaled = Sigma_hat / Tr(Sigma_hat)` and add Tikhonov `Sigma_reg = Sigma_scaled + (eps / Tr(Sigma_hat)) I` (Eq. 9); else `Sigma_reg = Sigma_hat + eps I` (Eq. 10). "Collective structural prior" `C_agg = 1 c^T`, `c^T = (1/d_in) 1^T sum_t Sigma~_t` (mean row broadcast). `W_pre = (sum_t W~_t Sigma_reg,t)(sum_t Sigma_reg,t + C_agg)^{-1}` (Eq. 14). If `gamma > tau`, spectral refinement: `Delta_res = sum_t Delta W_t (Sigma~_t - Sigma_reg,mean)`, SVD of `W_pre + Delta_res`, replace the top-k singular values by their mean, `W = W_pre + sigma_iso U_k V_k^T`, `k = 0.3 min(d_in, d_out)`.
- Hyperparameters: `eps = 4e-2` (GPT-2), `2e-4` (RoBERTa-base), `1e-5` (everything else); `tau = 0.3`; `k_frac = 0.3`.
- Statistics: weights only. No calibration data.
- Condition numbers: not used as a rule, but ill-conditioning is the stated motivation: "scale imbalance across tasks, ill-conditioned aggregation, and instability under strong heterogeneity"; and "the preliminary solution W_pre can remain spectrally ill-conditioned" (top 5% of singular values hold >99% of the energy), which motivates the post-hoc top-k flattening.
- Regularization: fixed-per-family Tikhonov. App. B.1: eps is "introduced primarily for numerical stability of the closed-form solve"; as eps -> infinity the merge tends to plain averaging (homogeneous branch) or an energy-weighted average (heterogeneous branch). Sec. 5.3: "the regularization strength eps more directly controls the balance between numerical stability and fidelity, suggesting that a principled strategy for choosing eps could further improve the method." That is an explicit open slot for a principled ridge rule, but in the data-free setting.
- Uncertainty / resampling / posterior: none. The `gamma > tau` switch is a deterministic scalar gate per layer (refinement applied only to heterogeneous layers), driven by weights, not by estimation uncertainty.
- Relation: (2) fixed ridge on a weights-only Gram, with an explicit call for a principled rule; (3) not at all.

### A.4 RegMean++, arXiv 2508.03121 v3 (TMLR)

- Exact method. Same objective and closed form as RegMean: `L = sum_i ||X_i W_M - X_i W_i||^2 + sum_i tr[(W_M - W_i)^T Lambda_i (W_M - W_i)]`, `Lambda_i = ((1-alpha)/alpha) diag(X_i^T X_i)`, `W_M = [sum_i G^_i]^{-1} sum_i G^_i W_i`, `G^_i = alpha G_i + (1 - alpha) diag(G_i)`. The only change: the input features of layer l are computed through the already-merged earlier layers (`X_i^(l) = f_M^(l-1)(X_i^(l-1))`), then run through the candidate's own layer l to get the Gram inputs of each linear sub-layer. About 2x merging time.
- Hyperparameters: alpha = 0.95 for vision (chosen on a 10% held-out slice of training data); LLM grid `{0.1, 0.3, 0.5, 0.7, 0.9}`; 256 samples per task/domain.
- Statistics: activation Grams (task inputs, no labels).
- Condition numbers / ridge / uncertainty: none. Regularization is only RegMean's off-diagonal alpha. Key data point for the project: Table 9 (ViT-B/32, 8-task validation), RegMean 87.6 at alpha = 0.90 and 0.95 vs 3.8 at alpha = 1.00; RegMean++ 89.8/90.1 vs 5.0. Text: "when alpha = 1.0, i.e., no scaling applied, the degradation happens and the overall accuracy is almost zeroed out. This phenomenon is consistent with the insight from Jin et al. (2022)". This matches the observed "plain RegMean collapses".
- Data sensitivity (Sec. 5.7): "merging performance improves modestly as the number of ID samples increases ... Even a small number of ID samples can achieve near-optimal performance"; class imbalance and OOD calibration data hurt regression-based methods (RegMean++ ViT-B/32: 84.4 ID, 76.3 class-imbalanced, 65.5 OOD). Layer analysis: using only middle+deep layers keeps >98% of accuracy (a position-based, not uncertainty-based, notion of which modules matter).
- Relation: (2) it is the off-diagonal-shrinkage comparator in point 2; (3) not at all.

### A.5 IterIS, arXiv 2411.15231 v2 (CVPR 2025)

- Exact method (LoRA merging, layer-wise). Objective `W* = argmin sum_i lam_i ||W_i^T X_i - W^T X~_i||_F^2`, where `X_i` are the inputs of LoRA i in its own model and `X~_i` the inputs of the unified adapter in the current merged model (recomputed every iteration; fixes the "rough assumption" that the merged adapter sees the expert's inputs). Closed form (Eq. 6): `W* = (sum_i lam_i X~_i X~_i^T)^{-1} (sum_i lam_i X~_i X_i^T W_i)`. Few-sample regularization (Eq. 7, Algorithm 1): replace `X~X~^T` by `X~X~^T + alpha ||X~X~^T||_F I` and `X~X^T` by `X~X^T + alpha ||X~X^T||_F I`, "we typically set alpha to 1e-4 or lower". Adaptive weights (Eq. 8) `lam_i = ||W_i||_F^2 / ||W_i^T X_i||_F^2`. Converges in at most J-1 iterations (DAG argument); capped at 20.
- Statistics: unlabeled activations, grouped by task (50 samples in the ablation; claims "only needing 1-5% of the unlabeled samples compared to prior methods").
- Condition numbers: not used. Motivation text: "To ensure the invertibility of sum X_i X_i^T ... they are usually accompanied by massive sample requirements."
- Ablation (App. C, with 50 samples): without the regularizer the LLM composition (COLA, MNLI) is (0.0, 0.0) vs (0.299, 0.780) with it; (MRPC, SST2) (0.272, 0.177) vs (0.824, 0.951); VLM POS CIDEr 0.534 vs 0.794, NEG 0.476 vs 0.794.
- Observation (mine, not in the paper): because `lambda_max(G) <= ||G||_F`, the IterIS ridge guarantees `kappa(G + alpha ||G||_F I) <= (1 + alpha)/alpha`, about 1e4 for alpha = 1e-4. In the near-singular regime the kappa-capped rule gives `rho ~ lambda_max / (kappa_t - 1)`, which is IterIS's `alpha ||G||_F` with `alpha ~ 1/(kappa_t - 1)` up to `lambda_max` vs `||G||_F`. So a fixed scale-relative ridge IS the practical near-equivalent of a kappa cap, and IterIS is the published instance of it in a RegMean-type LoRA merge. The cross term `(X~X^T + rho I) W_i` also pulls toward the (weighted) average of the experts, i.e. exactly the project's `+ rho R` form with `R` the average.
- Relation: (2) HIGH-closeness precedent for the fixed relative ridge control; (3) not at all.

### A.6 SLOA, arXiv 2610.06415 v1

- Exact method. Merges "local functional operators" instead of raw weights: linear map `W` (behavior `X W`), LayerNorm `(gamma, beta)`, and composed attention operators `theta_QK = W_Q W_K^T / sqrt(d_k)` (behavior `X theta X^T`) and `theta_VO = W_V W_O` (behavior `A X theta`). Per operator, in topological order through the partially merged model (Eq. 5): `theta = argmin (1/T) sum_t [ (1/n_t) sum_i ||A(Z^_{t,i}; theta) - A(Z_{t,i}; theta_t)||_F^2 + ||theta - theta_t||_{Lambda_t}^2 ]`, with `Z^` the local input in the partially merged model, `Z` the input in task model t, and `Lambda_t = alpha diag(Cov(phi(chi_t))) + eps I` (QK: `Lambda (x) Lambda`). Closed form for linear/MLP operators (Eq. 98): `theta = (sum_t G^_t)^{-1} (sum_t C^_t)`, `G^_t = (1/n_t) sum_i Z^^T Z^ + Lambda_t`, `C^_t = (1/n_t) sum_i Z^^T Y + Lambda_t theta_t`. QK and VO are solved in the ambient d x d space and then SVD-factorized to a chosen rank r (r > d_k adds capacity; `W_Q = U_r (sqrt(r) Sigma_r)^{1/2}` etc.).
- Why it matters for us beyond RegMean: Example 1 shows that independent averaging of `W_Q` and `W_K` is not invariant to the function-preserving reparametrization `(W_Q R, W_K R^{-T})` (merged score operator scaled by `(1+c)(1+c^{-1})/4` for `R = c I`). My inference: any per-matrix scalar score of `W_Q` or `W_K` alone (such as `kappa(W_Q)`) changes under that same reparametrization when R is not orthogonal, so such scores are gauge-dependent for attention (the same applies to the LoRA `B A` factorization).
- Statistics: unlabeled calibration activations of the task models and of the partially merged model (N = 256 per task default, from training splits), token covariances.
- Condition numbers: not used; `eps I` "provides a ridge term for numerical stability"; "larger calibration sets provide a better-conditioned system for estimating the merged weights" (F.3).
- Calibration-size evidence (F.6): "With few samples, especially 64 or 128, larger alpha improves performance by anchoring the solution toward the task-specific weights. With more samples, 256 or 512, ... strong regularization becomes unnecessary and can reduce performance. The optimal alpha therefore decreases as the calibration set grows, approaching alpha ~ 0.05." F.8: RegMean and RegMean++ "plateau early" in calibration size.
- Relation: (2) data-dependent anchoring strength; (3) PARTIAL and qualitative only: the optimal pull toward the anchor depends on how much data there is, but alpha is a global hand-tuned hyperparameter, not an estimate of per-module uncertainty.

### A.7 Cross-paper matrix (what each uses, and whether it does point 3)

| | RegMean | RegMean++ | IterIS | SLOA | SWUDI | ACE | ReForge |
|---|---|---|---|---|---|---|---|
| Statistics | activation Grams | activation Grams via merged earlier layers | activations (unified-model inputs, iterated) | activations of task and partially merged models | weights only (task-vector Gram) | weights only (task-vector Gram) | activations (DA) or task-vector Gram (DF) + labelled validation for BO |
| Condition number as a rule | no | no | no (ridge implies a loose cap, my derivation) | no | diagnostic only (Table XVII) | no | no |
| Regularizer | off-diagonal alpha (0.9; T5 0.1) | same, alpha 0.95 | `alpha ||G||_F I`, alpha <= 1e-4 | `alpha diag(Cov) + eps I` | spectral filter (exp + top-K or rank rules); Tikhonov only named | fixed Tikhonov eps per family | anchor-centered ridge, lam tuned by BO per module group |
| Spectral filtering | no | no | no | no | yes (core) | post-hoc top-k flattening | no (diagnostics only) |
| Uncertainty / resampling | no | no | no | no | no (noise model is a "negative diagnostic") | no | analytic posterior used for ECE on one layer; no resampling |
| Bayesian | no | no | no | Mahalanobis prior (metric) | no | no | yes (anchor-centered Gaussian prior, MAP) |
| Already does point 3? | not at all | not at all | not at all | partially (sample-size-dependent anchoring, hand-tuned) | not at all (deterministic directional shrinkage toward `tau_init`) | not at all | partially in structure (anchor-shrunk correction + posterior), not in function |

---

## B. Search for closer work

Method: 58 distinct web queries (full list in Appendix 1) plus direct arXiv API lookups of ids, and full-text keyword scans (condition number / ill-conditioned / kappa / exp / softmax / ridge / Tikhonov / bootstrap / posterior / shrink) of 34 downloaded PDFs, with section reads of the relevant ones. A "no" from a keyword scan means "no hit in pdftotext output", not a careful read; equations rendered as images could be missed.

Closeness legend. (i) = professor's rule `softmax(-beta kappa(W))` weights; (ii) = kappa-capped ridge for the RegMean solve; (iii) = bootstrap/posterior uncertainty of the correction used to gate/shrink it. HIGH = same mechanism; MEDIUM = shares the structure or a key ingredient; LOW = adjacent.

### B.1 Condition numbers, exponential weights, scalar scores of matrices (point i)

| Paper (id; venue; date) | Summary | (i) | (ii) | (iii) |
|---|---|---|---|---|
| AMM, "Towards Efficient Multimodal Unified Reasoning Model via Model Merging" (2510.08987 v2; tech report; Oct 2025) | Per-layer, per-task weights `w_i,l ~ alpha_i,l * beta_i,l` with `alpha = ||tau_i||_F^2 / sum_j ||tau_j||_F^2` and `beta_i,l = exp(-gamma ||(tau_m - tau_i) tau_i^T||_F^2 / ||tau_i||_F^2)`, recomputed as the merge evolves (inside a WUDI-style optimization with a gradient-projection regularizer). No condition numbers. | MEDIUM (the only exponential per-matrix weighting I found, but the score is current WUDI interference, not kappa) | LOW | LOW |
| NAN (2505.16148; no venue listed; May 2025) | Coefficient `alpha_i = (1/||W_i||_F) / sum_j (1/||W_j||_F)` on the model weights `W_i`, justified by "variance of learned weights is inversely correlated with training data volume". | MEDIUM (a scalar statistic of each `W_i` turned into normalized merge weights; norm, not kappa) | LOW | LOW |
| FroM (2506.02478 v2; EMNLP 2025 Findings per search; Jun 2025) | Data-free Frobenius-norm-based adaptive weighting presented as an improvement to RegMean, with one extra hyperparameter. Read at abstract/keyword level. | LOW to MEDIUM | LOW | LOW |
| TARA-Merging (2603.26299; CVPR 2026; Mar 2026) | Reweights LoRA directions with a preference-weighted pseudo-loss. Uses the condition number `kappa` of the task-loss Jacobian restricted to the LoRA span, plotted per layer/module (App. Figs. 6-7), purely as an anisotropy diagnostic; also uses effective rank. | LOW to MEDIUM (one of very few merging papers using condition numbers, but of a Jacobian, as a diagnostic) | LOW | LOW |
| SWUDI (2606.07289) | `lambda_max/lambda_med` per architecture as a diagnostic (see A.1). | LOW | MEDIUM | LOW |
| HARC, "When Model Merging Breaks Routing" (2606.03391; Jun 2026) | MoE routing calibration with a Hessian-aware system solved by conjugate gradient; the diagonal-interpolation regularizer (damping 0.9) is justified because it "lifts near-zero eigenvalues and reduces the condition number of the aggregated system"; summing more tasks raises the minimum eigenvalue. | LOW | LOW to MEDIUM | LOW |
| MetaGPT (2406.11385; EMNLP 2024; Jun 2024) | Closed-form scaling coefficients from local linearity and task-vector orthogonality; norm-based, data-free. No condition numbers, no exp. | LOW | LOW | LOW |
| DisTaC (2508.01148 v3; Aug 2025) | Task-vector norm disparity and low source-model confidence break merging; distillation "pre-conditions" task vectors. "Conditioning" here is not kappa. | LOW | LOW | LOW |
| LARV (2602.09413; Feb 2026) | Data-free per-layer scale from effective-rank contrast between update and base. | LOW | LOW | LOW |
| PACT (2606.18627 v3; Jun 2026) | Argues task-critical knowledge stays in the pretrained weights ("Load-Bearing Wall" dimensions) and works with `theta_0 = U0 Sigma0 V0^T`. Tangential: consistent with, but does not test, the observation that `kappa(W_i)` mostly reads the base. No condition numbers. | LOW | LOW | LOW |

Methods in the project's checklist that use neither condition numbers nor exponential/softmax weights (keyword scan of full PDF text; abstracts read): SADA-Merging (2609.24612: spectrum-aware subspace capacity, plasticity-based spectral preservation, depth-dependent anchoring), CT-Merging (2607.20561: projector-averaged consensus directions, per-task residual scale), SVC (2602.05536, ICML 2026: rescales inflated singular values), STAR (2502.10339, NAACL 2025: spectral truncation + nuclear-norm rescale), TSV-M (2412.00081, CVPR 2025), Iso-C/Iso-CTS (2502.04959, ICML 2025: flatten the singular spectrum), WUDI (2503.08099, ICML 2025), OptMerge (2505.19892; its `kappa` is a directional-similarity constant, not a condition number), MetaGPT. Iso-C, SVC and STAR do manipulate the singular spectrum of the merged matrix, which implicitly changes its conditioning, but none weights experts by it. TARA (above) is the only one of the listed names that mentions condition numbers.

### B.2 Conditioning and regularization of RegMean-type solves (point ii)

| Paper (id; venue; date) | Summary | (i) | (ii) | (iii) |
|---|---|---|---|---|
| IterIS (2411.15231) | `alpha ||G||_F I` ridge for few-sample LoRA merging (A.5). | LOW | HIGH (fixed scale-relative ridge; equals a kappa cap in the near-singular regime up to constants) | LOW |
| ReForge/BMM (2605.12843) | Anchor-centered ridge `U0 + B (G + lam I)^{-1}` with unit-trace Grams; `lam` per module group by BO on validation (A.2). | LOW | MEDIUM to HIGH (same algebra as "ridge toward R"; rule is validation-tuned, not kappa) | MEDIUM |
| FeatCal (2605.13030 v2; May 2026, rev. Oct 2026) | Post-merging calibration with a small calibration set: `W* = (sum_i w_i W_i C_i + lam W_anc)(sum_i w_i G_i + lam I)^{-1}`, `w_i = 1/max(||G_i||_F, eps)`, `W_anc = rho W_mer + (1-rho) W_base`; "the ridge sweep is flat over small positive values"; 8 examples per task give 82.9% on CLIP-ViT-B/32 TA. | LOW | MEDIUM to HIGH (anchor ridge with Frobenius-normalized Grams; `lam` a fixed hyperparameter) | LOW to MEDIUM (regularization "keeps the calibrated model close to the merged model" to avoid overfitting a small calibration set) |
| SWUDI (2606.07289) | Spectral filtering of the same normal equation, Tikhonov named as a member (A.1). | LOW | MEDIUM | LOW |
| ACE-Merging (2603.02945) | Fixed Tikhonov `eps`; asks for "a principled strategy for choosing eps" (A.3). | LOW | MEDIUM | LOW |
| MaTS, "Merging by Matching Models in Task Parameter Subspaces" (2312.04339; TMLR 2024; Dec 2023) | Observes RegMean "has to scale the non-diagonal entries of the gram matrix ... to ensure the inverse is not ill-conditioned" and that "the matrix inverse needed in RegMean is numerically unstable and so the closed-form solution does not actually result in the optimal solution"; conjugate gradient started from a task-arithmetic point reaches a lower RegMean objective than the closed form. | LOW | MEDIUM (early-stopped iterative solve from an anchor = implicit regularization; same idea SWUDI formalizes) | LOW |
| RegMean / RegMean++ (A.0, A.4) | Off-diagonal alpha; alpha = 1 collapses (3.8%/5.0% accuracy on the ViT-B/32 8-task validation sweep). | LOW | MEDIUM | LOW |
| SLOA (2610.06415) | `alpha diag(Cov) + eps I`; stronger anchoring when data are scarce (A.6). | LOW | MEDIUM | LOW to MEDIUM |
| ACTMat, "Model Merging via Data-Free Covariance Estimation" (2604.01329 v2; COLM 2026; Apr 2026) | Data-free merge with a pseudo-inverse of the summed estimated covariances; Thm 3.3 bounds negative transfer by layer-wise interference plus covariance-estimation error `||C_t - C_hat_t||_F` multiplied by factors built from `kappa_W = max_t ||W_t||_F`, `||S+||_F`, `max(||S+||_F^2, ||S_hat+||_F^2)` and `sum_t' ||C_hat_t'||_F` (`S = sum_t C_t`, `S+` its pseudo-inverse; exact grouping of the factors is ambiguous in the extracted PDF text), so inverse-Gram norms enter, i.e. ill-conditioning amplifies Gram error. An analytic, worst-case sensitivity of the merge to Gram error. | LOW | LOW to MEDIUM | LOW to MEDIUM (analytic sensitivity to Gram error, not empirical, not used to gate) |
| Won, Lim, Kim, Rajaratnam, "Condition-number-regularized covariance estimation" (JRSSB 75(3):427-450, 2013; PMC3667751) | Maximum likelihood subject to `cond(Sigma) <= kappa_max`; solution clips sample eigenvalues into `[tau*, kappa_max tau*]` (a Stein-type nonlinear shrinkage); if the sample covariance already satisfies the bound it is returned unchanged; `kappa_max` is selected adaptively by cross-validation. Motivation: the true covariance is rarely ill-conditioned but its small-n sample counterpart often is. | LOW | HIGH for the principle (explicit condition-number cap, no-op when already below the cap, CV-selected `kappa_max`); different operator (eigenvalue clipping, not an additive ridge) and a covariance estimation context, not a merge solve | LOW |
| OCReP, Cancelliere, Gai, Gallinari, Rubini (1508.06095; Neural Networks; Aug 2015) | Tikhonov-regularized pseudoinversion training of single-hidden-layer nets; derives the ridge parameter analytically from well-conditioning requirements instead of cross-validation. | LOW | HIGH for "ridge chosen by a conditioning criterion" (different task) | LOW |
| Peeters, van de Wiel, van Wieringen, "The Spectral Condition Number Plot for Regularization Parameter Determination" (1608.04123; Comput. Stat. 35:629-646, 2020) | For ridge-type covariance estimators with p > n, picks the penalty at which the condition number of the regularized estimate stabilizes. | LOW | MEDIUM to HIGH for conditioning-driven penalty selection | LOW |
| Adriazola (2207.01139) | Perturbation theory for how a small Tikhonov term changes the condition number of an SPD matrix. | LOW | MEDIUM (background) | LOW |

The kappa-capped ridge itself is one line: for `G` with extreme eigenvalues `lmax >= lmin >= 0`, `kappa(G + rho I) = (lmax + rho)/(lmin + rho) <= kappa_t` iff `rho >= (lmax - kappa_t lmin)/(kappa_t - 1)`, so the smallest admissible `rho` is `max(0, (lmax - kappa_t lmin)/(kappa_t - 1))`. This derivation is elementary and mine (the search tool's own summary produced the same closed form but cited no source for it); I did not find it written in any merging paper.

### B.3 Uncertainty, posterior, resampling, shrinkage, gating (point iii)

Merging papers:

| Paper (id; venue; date) | Summary | (i) | (ii) | (iii) |
|---|---|---|---|---|
| ReForge/BMM (2605.12843) | Anchor-centered Bayesian linear regression; analytic posterior covariance; posterior sampling for ECE on the last layer; `lam` by validation BO (A.2). | LOW | MEDIUM | MEDIUM (closest found; not a gate, not resampled) |
| Daheim et al., "Model Merging by Uncertainty-Based Gradient Matching" (2310.12808 v2; ICLR 2024; Oct 2023) | Merged parameters are the MAP of a Laplace-approximated posterior with a Gaussian prior centered at the pretrained weights, `N(theta_LLM, H0^{-1})`, combining task Hessians; "tied to the uncertainty of the models, which determines their scaling". Uses "inexpensive Hessian estimates" (abstract), not resampling of calibration data. | LOW | LOW | MEDIUM (Bayesian shrinkage toward an anchor with curvature as uncertainty; not calibration-resampled; not applied to a RegMean correction) |
| Fisher merging, Matena and Raffel (2111.09832 v2; NeurIPS 2022; Nov 2021) | Laplace view: precision-weighted (Fisher) average of expert parameters; fallback to the target model's value when the Fisher is near zero. | LOW | LOW | LOW to MEDIUM |
| MaTS (2312.04339) | Casts RegMean and Fisher merging as linear systems solved by conjugate gradient (see B.2). | LOW | MEDIUM | LOW |
| EpiMer, "Model Merging on Loss Landscape: A Geometry Perspective" (2605.26693; CVPR 2026 Findings; May 2026) | Frechet mean on a Riemannian manifold with the expected Hessian as metric, "a connection between local curvature and epistemic uncertainty of the parameters"; error bound = subspace Frechet variance + residual energy. Curvature, not resampling. | LOW | LOW | LOW to MEDIUM |
| Khan, "Knowledge Adaptation as Posterior Correction" (2506.14262 v3; Jun 2025) | Casts adaptation methods, including model merging, as corrections of old posteriors; "more accurate posteriors often lead to smaller corrections". Conceptual. | LOW | LOW | LOW to MEDIUM |
| PoE-EBM, "Model Merging as Probabilistic Inference in Fine-Tuning Parameter Space" (2607.01689; UAI 2026; Jul 2026) | Product-of-experts view; finds directional residuals heavy-tailed and uses Cauchy experts. Abstract + keyword scan only. | LOW | LOW | LOW |
| UQ-Merge, Qu et al. (Findings of ACL 2025; no arXiv id found) | Label-free uncertainty quantification to decide which MLLM experts to merge, in what order, and how many. Uncertainty gates models, not corrections. | LOW | LOW | LOW to MEDIUM |
| M-Loss (2602.08564; Feb 2026) | Compatibility metric between parameter averaging and output ensembling from limited unlabeled data, at layer and node level. | LOW | LOW | LOW to MEDIUM (a calibration-derived trust/compatibility score, not an uncertainty of a fitted correction) |
| Model Merging by Output-Space Projection (2605.29101; ICML 2026; May 2026) | Convex QP over residual updates minimizing a squared-output calibration loss with ~100 calibration examples per task; a closed-form diagnostic ("fraction of residual energy captured by a chosen basis") predicts merge quality from the calibration set. | LOW | LOW | LOW to MEDIUM |
| BOOM, "Bagging-Based Model Merging for Robust General Text Embeddings" (2602.05787; Feb 2026) | Trains several embedding models on sampled data subsets and merges them: bagging over TRAINING subsets, not over calibration data of a merge correction. | LOW | LOW | LOW |
| Soen and Sun, "On the Variance of the Fisher Information for Deep Learning" (2107.04205; NeurIPS 2021) | Closed-form variance of Fisher estimators; not applied to merging. Abstract only. | LOW | LOW | LOW |

Classical statistics (the mechanism class of point iii; all of these were checked only at abstract or secondary-source level, via search summaries, not from the original papers):

| Source | What it gives | (iii) |
|---|---|---|
| James-Stein / Efron-Morris empirical Bayes shrinkage (Efron and Morris line of work) | Shrink each estimate toward a common/anchor value by a factor estimated from the data. | HIGH as mechanism (not merging) |
| Stein-rule and preliminary-test (pretest) estimators; positive-part Stein rule `max[0, 1 - a e'e / b'Sb] b` (Baranchik; Judge and Bock; years differ across sources, check originals) | Shrink an unrestricted regression estimate toward a restricted (anchor) estimate by a data-dependent factor built from an F/Wald statistic; pretest versions switch hard between the two. This is exactly "scalar gate per module on the correction". | HIGH as mechanism (not merging) |
| Green and Strawderman (JASA 86:1001-1006, 1991) and follow-ups (Rosenman et al., arXiv 2002.06708, 2309.06727) | James-Stein-type combination of an unbiased noisy estimate with a possibly biased one; the weight depends on the estimated discrepancy and noise. | HIGH as mechanism (not merging) |
| Efron, "Estimation and Accuracy After Model Selection" (JASA 109:991-1007, 2014) | Bagging/bootstrap smoothing of discontinuous selection rules, with a formula for the standard error of the smoothed estimate. | MEDIUM to HIGH (bootstrap-smoothed gate) |
| Li (1985, Annals of Statistics) on SURE/GCV for ridge; Nobel, Candes, Boyd (IEEE TSP 2023) | Data-driven ridge strength without a validation set (SURE needs a noise-variance estimate). | MEDIUM |
| Li, Cai, Li, Trans-Lasso (arXiv 2006.10593; JRSSB 2022) | Data-driven transfer: protects the target estimate against uninformative auxiliary sources. | LOW to MEDIUM |
| Case/pairs bootstrap for regression and the bootstrap of Gram matrices (generic) | Resample (X, Y) pairs, recompute the estimator, read the spread; clustered resampling is the standard fix for correlated tokens within a document (general knowledge, not a cited source). | MEDIUM to HIGH |

### B.4 Calibration-set-size sensitivity (what the merging literature already shows)

| Source | Finding |
|---|---|
| RegMean (2212.09849) App. C, Table 6 | Performance improves with N then "saturates around N = 100" batches of 16 (RoBERTa-base emotion tasks). |
| RegMean++ (2508.03121) Sec. 5.7, Fig. 8-9 | Modest gains with more ID samples; small sets already near-optimal; OOD or class-imbalanced calibration data hurt regression-based merging; alpha = 1 collapses. |
| IterIS (2411.15231) | 50 samples with a scale-relative ridge; without it the LLM compositions collapse (0.0, 0.0). |
| SLOA (2610.06415) F.3, F.6, Table 12 | Accuracy rises with calibration samples with diminishing returns; ViT-B/32, 20 tasks, r = 128: 79.77 +- 0.38 (N = 128), 81.03 +- 0.12 (256), 81.63 +- 0.13 (512), 81.92 +- 0.08 (1024), 82.01 +- 0.07 (2048), i.e. the seed-to-seed spread also shrinks with N. The optimal anchoring strength alpha decreases as the set grows (approaching about 0.05); larger alpha helps at 64-128 samples. |
| ReForge/BMM (2605.12843) Table 4, v1 App. F | One-shot or five-shot activation Grams are "noisy and rank-deficient"; mixing with the weights-only Gram recovers 128-sample accuracy. |
| FeatCal (2605.13030) | 8 examples per task reach 82.9% (CLIP-ViT-B/32 TA); ridge sweep flat over small positive values; very large ridge suppresses the update. |
| ProDistill (2502.12706; ICML 2025) | Search-snippet level only: more than 7% gain with 1-shot per task on vision. |
| Expert Merging (2509.25712) App. C.4 | "performance saturates with very few unlabeled samples: best averages occur at N in {5, 10}, and larger N brings marginal or negative returns" (verified in text; learns layer-wise coefficients, not a RegMean solve). |
| MAGIC (2512.19320) | "Even a single additional unlabeled sample per task yields a substantial improvement ... gains become marginal" as data grow (verified in text; magnitude calibration). |

### B.5 Short notes

1. No query found anyone resampling calibration data to estimate the uncertainty of a RegMean-type correction, nor using such an estimate to gate or shrink it toward the anchor. The closest merging papers either tune the shrinkage strength by validation (ReForge, SLOA, RegMean/RegMean++ alpha, FeatCal, IterIS, ACE) or use deterministic/analytic quantities (SWUDI eigenvalue filters, ACTMat inverse-Gram norms, Daheim/Fisher curvature).
2. The anchor-plus-correction decomposition `R + B G^{-1}` is already explicit in the literature: ReForge App. A (Eq. 17) writes the anchor-centered correction `(C - U0 G)(G + lam I)^{-1}`, and SWUDI writes `tau_init + (D - tau_init C) C_h^+`. Even their diagnostic fits a per-module scalar "correction scale" `a_*` (ReForge Eq. 18, a nonnegative scale fitted so that the surrogate correction matches the activation-based one); that is a scalar gate on the correction, but fitted for a diagnostic, not as an uncertainty gate.
3. Two published cautions bear directly on point 3: SWUDI's Fig. 8 (Wiener/Bayes-risk and SNR-gap criteria "do not by themselves pick the best merging filter") and ReForge's finding that 1-5-shot activation Grams are noisy, which they address by mixing with a weights-only Gram rather than by gating.
4. Scale-relative ridges are already routine in merging: IterIS (`alpha ||G||_F`), ReForge (unit trace), FeatCal (Frobenius-normalized), ACE (`eps/Tr`), SLOA (`alpha diag Cov`), RegMean (diagonal-relative). That is why a kappa-capped ridge and an equally tuned fixed relative ridge nearly coincide.
5. Name collision to avoid: ReForge uses `rho` for the Gram-mixing weight and FeatCal uses `rho` for the anchor blend; the project uses `rho` for the ridge.

---

## C. Verdicts

**(1) Is the professor's exact rule, `exp(-beta * kappa(W))` weights, in prior work?** I found no instance. In 58 queries and a keyword scan of 34 full texts, no merging paper weights experts by the condition number of their weight matrices, with or without an exponential/softmax. The nearest forms are different scores: AMM's per-layer `exp(-gamma * normalized WUDI interference)`, NAN's `1/||W_i||_F`, FroM's Frobenius norm, and Fisher/Hessian precision weights. Condition numbers appear in merging papers only as diagnostics (SWUDI's `lambda_max/lambda_med` of the task-vector Gram; TARA's `kappa` of a task-loss Jacobian; HARC's justification of damping). So the rule is not documented elsewhere as far as my search reaches, but that is "no evidence found", not "proved absent", and it is the professor's idea rather than a result; the empirical finding that it equals uniform averaging and shuffled-score nulls stands on its own. One mathematical caveat for the write-up: a per-matrix scalar like `kappa(W_Q)` is not invariant to the function-preserving attention reparametrization `(W_Q R, W_K R^{-T})` (SLOA Example 1; also LoRA `B A`), so per-matrix kappa scores are gauge-dependent for attention and LoRA unless R is orthogonal (my inference).

**(2) Is a kappa-capped ridge for the RegMean solve prior art?** Yes as a family and as a principle; as a specific rule it is a near-duplicate of published practice, but the exact "set rho per module so that kappa(G + rho I) = kappa_target" rule I did not find in any merging paper. The pieces: (a) Tikhonov/ridge on the Gram solve is standard practice and is already used in merging in scale-relative forms (IterIS `alpha ||G||_F I` with the few-sample ablation showing collapse without it; ReForge/FeatCal anchor ridges with normalized Grams; ACE fixed `eps`; SLOA `eps I`); (b) the same normal equation `tau C = D` with spectral regularization is SWUDI, which names Tikhonov as a family member and finds an exponential filter plus hard truncation best, and MaTS showed the closed-form RegMean inverse is "numerically unstable" and can be beaten by conjugate gradient from an anchor; (c) bounding the condition number as the regularization target, with a no-op when already below the bound and a CV-chosen bound, is established statistics (Won et al. 2013 via eigenvalue clipping; Peeters et al. 2020 via the condition-number plot; OCReP 2015 for pseudoinversion training). By my derivation IterIS's `alpha ||G||_F` ridge guarantees `kappa <= (1 + alpha)/alpha` and in the near-singular regime equals the kappa cap with `kappa_t ~ 1 + 1/alpha`. Combined with the observation that kappa-capped beats RegMean's off-diagonal shrinkage by 0.2-0.8 pt but an equally tuned fixed ridge by only 0.1-0.2 pt, I would present the kappa cap as a re-parametrization (a way to set the fixed ridge from a module's own spectrum without per-model tuning), not as a new method. The defensible claim is empirical: how well one `kappa_target` transfers across modules and models compared with tuning a relative ridge.

**(3) Is "empirical joint (bootstrap) uncertainty of the merge correction, used to gate/shrink it" prior art?** Not in the merging literature that I could reach, with these precise exceptions. ReForge (Bayesian linear regression with an anchor-centered prior) contains the algebra (anchor-shrunk correction, analytic posterior covariance `beta^{-1}(X X^T + lam I)^{-1}`) but uses neither resampling nor any uncertainty-dependent gate: `lam` is tuned on labelled validation, and the posterior is sampled on one layer for ECE. SLOA shows qualitatively that the right pull toward the anchor shrinks as calibration data grow, but with a hand-tuned global alpha. SWUDI gives a deterministic, eigenvalue-dependent version of directional shrinkage toward `tau_init`, and warns that risk/noise models did not select the best filter. Daheim et al. and Fisher merging are Bayesian/Laplace shrinkage toward an anchor with curvature (Fisher/Hessian) as the uncertainty, not calibration resampling. In statistics, however, the mechanism class is textbook: James-Stein/Efron-Morris, Stein-rule and pretest estimators that shrink an unrestricted regression estimate toward a restricted one by an F/Wald-based factor, Green-Strawderman combination of unbiased and biased estimates, bagging/bootstrap smoothing, SURE/GCV-tuned ridge. So the idea of "shrink an estimated correction toward an anchor in proportion to its estimated uncertainty" cannot be claimed as new; what could still be new is the specific application (document/image-level bootstrap of the joint `(B, G)` for RegMean-type merges, scalar then directional gates) and a demonstrated gain over: a tuned fixed ridge, ReForge-style validation-tuned `lam`, a closed-form Stein-rule gate computed from the plug-in covariance of `F`, and a shuffled-uncertainty null (the gate must beat shuffled scores, given what happened to the kappa rule).

### What would have to be checked further, and blind spots

1. Not indexed by my tools: OpenReview submissions (ICLR 2027, NeurIPS 2026 reviews), camera-ready ICML/CVPR/COLM 2026 papers not on arXiv, ACL Anthology items without an arXiv id (UQ-Merge), IEEE Xplore (SWUDI looks like a journal extension), patents. The search engine is US-only and snippet-based.
2. Forward citations of RegMean, RegMean++, IterIS, FeatCal, ReForge and SWUDI (Semantic Scholar / Google Scholar) filtered for "bootstrap", "uncertainty", "shrinkage", "Stein", "posterior".
3. Partial reads: ReForge v2 (only v1 and v3 read); ReForge's anonymous code repository (a gate or posterior-based pruning could exist in code but not text); SLOA Fig. 9 data (text only); IterIS supplementary; FeatCal's full ablations over `lam` and `rho`; ACTMat appendix; PoE-EBM, EpiMer, M-Loss and Output-Space Projection were read at abstract/keyword level. Keyword scans of SADA, CT-Merging, SVC, STAR, TSV, Iso-C/CTS, WUDI, OptMerge and MetaGPT are not close reads.
4. Different vocabulary could hide related work: "robust RegMean", "sample covariance noise", "stability", "bagged RegMean", "consensus across calibration subsets", "selective/partial merging", "layer selection by reliability". Two searches along those lines (bagging over calibration subsets; cross-fitting/split-half gates) found nothing in merging.
5. Classical-statistics items marked "secondary" (Baranchik years, Judge and Bock, Green and Strawderman, Efron-Morris, Li 1985, Nobel-Candes-Boyd) were verified only through search summaries; originals should be opened before citing page numbers or exact estimator forms.
6. Wider literature that deserves a targeted look before claiming novelty: multi-task and hierarchical Bayesian regression (Lindley-Smith type shrinkage), transfer learning in regression (Trans-Lasso and descendants), and covariance-shrinkage work on small-sample second-moment matrices (Ledoit-Wolf family), since the problem here is a ridge-regression correction estimated from a small sample.

---

## Appendix 1. Web queries issued (58)

1. IterIS: Iterative Inference-Solving Alignment for LoRA Merging arXiv
2. model merging weights experts by condition number of weight matrix softmax
3. RegMean uncertainty bootstrap resampling calibration data merging correction shrinkage
4. RegMean Gram matrix ill-conditioned few calibration samples ridge regularization model merging
5. Bayesian model merging Laplace approximation posterior shrink toward anchor merged model
6. model merging by uncertainty-based gradient matching Daheim Hessian
7. James-Stein shrinkage model merging task vectors
8. empirical Bayes shrinkage estimator model merging fine-tuned experts
9. condition number regularization model merging closed-form ridge eigenvalue spectral filtering Gram merging
10. SADA model merging arXiv
11. CT-Merging model merging condition number
12. AMM adaptive model merging exponential weights task vectors
13. model merging trust region gated correction toward anchor weight averaging per-layer confidence
14. SVC singular value calibration model merging task vectors
15. TARA model merging task-aware rank adaptation LoRA merging
16. STAR spectral truncation and rescale model merging
17. MetaGPT merging large language models model exclusive task arithmetic closed-form scaling
18. condition-number-regularized covariance estimation Won Lim Kim Boyd
19. Ledoit-Wolf shrinkage Gram matrix activation covariance model merging regression mean
20. Merging by Matching Models in Task Parameter Subspaces regularized covariance conjugate gradient Fisher
21. data-efficient model merging number of calibration samples sensitivity Fisher merging RegMean few samples
22. Stein unbiased risk estimate select ridge shrinkage parameter without validation data
23. Model Merging via Data-Free Covariance Estimation 2604.01329
24. WUDI merging task vectors proxy activations interference data-free optimization
25. noisy Fisher information estimate model merging variance of Fisher estimate sample size
26. model merging bootstrap confidence interval merged weights stability
27. UQ-Merge uncertainty guided multimodal large language model merging
28. FeatCal feature calibration post-merging models closed-form least squares
29. LoRA merging activation Gram closed-form least squares regularization small calibration set underdetermined
30. NAN training-free coefficient estimation model merging inverse variance task vector norm
31. model merging guarantee no worse than weight averaging safe fallback selective merging abstain
32. bootstrap RegMean Gram matrix sampling variability merged weights calibration examples resampled
33. Bayesian ridge regression posterior covariance per-direction shrinkage toward prior mean model merging Gram eigenbasis
34. condition number of weight matrices fine-tuned models merging mergeability ill-conditioned task vector layers
35. Laplace approximation LoRA merging uncertainty Bayesian LoRA adapters merge posterior
36. Tikhonov regularization parameter chosen to achieve target condition number ridge regression numerical stability Gram matrix
37. model merging sensitivity to calibration sample selection seed variance data-dependent merging stability across calibration sets
38. openreview 2026 model merging closed-form Gram ridge regularization calibration uncertainty submission
39. inverse-variance weighting parameter averaging fine-tuned models epistemic variance merge coefficients per-parameter
40. influence functions model merging calibration data which examples drive merged weights sensitivity
41. OCReP optimally conditioned regularization pseudoinversion ridge parameter minimizes condition number
42. Progressive layer-wise distillation ProDistill model merging few-shot validation sampling sensitivity
43. Fisher merging damping regularized Fisher near-zero Fisher numerical issues shrink to uniform averaging
44. spectral norm effective rank weighting experts merging per-layer softmax temperature coefficients data-free merge
45. Model Merging by Output-Space Projection calibration quadratic program arXiv
46. "condition number" "model merging" LoRA singular value spectrum interference weighting adapters
47. well-conditioned merge Gram matrix spectral clipping eigenvalue floor RegMean variant 2026
48. Stein-rule estimator shrink unrestricted regression estimator toward restricted estimator positive-part Baranchik Judge Bock pretest
49. Efron estimation and accuracy after model selection bootstrap smoothing bagging variance of shrinkage estimator
50. Efron Morris empirical Bayes shrinkage toward a common mean James-Stein estimator Stein's paradox regression
51. ProbSurgery probabilistic feature drift model merging representation surgery uncertainty
52. merged model post-merging correction weights calibration set overfit regularization toward merged weights adaptive per-layer strength estimated from data noise
53. weight averaging merging coefficients proportional to exp(-beta * sharpness or Lipschitz or spectral norm) per-model softmax weights model soup
54. well-conditioned fine-tuned weights merge better singular value spectrum minimum singular value fine-tuned layer merging weights stable rank
55. Green Strawderman 1991 James-Stein type estimator combining unbiased and possibly biased estimators
56. Trans-Lasso transfer learning high-dimensional linear regression Li Cai Li data-driven shrinkage toward source estimator
57. bagging ensemble of merged models over calibration subsets average merge solutions subsampled Gram matrices model merging variance reduction
58. cross-fitting split-half calibration stability gate per-layer merge correction apply only if reproducible across calibration splits

## Appendix 2. URLs read

Mandatory references, primary full text (abstract page with version history, HTML, PDF):

- https://arxiv.org/abs/2606.07289
- https://arxiv.org/html/2606.07289v1
- https://arxiv.org/pdf/2606.07289v1
- https://arxiv.org/abs/2605.12843v1
- https://arxiv.org/abs/2605.12843v2
- https://arxiv.org/abs/2605.12843v3
- https://arxiv.org/html/2605.12843v1
- https://arxiv.org/html/2605.12843v3
- https://arxiv.org/pdf/2605.12843v1
- https://arxiv.org/pdf/2605.12843v3
- https://arxiv.org/abs/2603.02945
- https://arxiv.org/html/2603.02945v2
- https://arxiv.org/pdf/2603.02945v2
- https://arxiv.org/abs/2508.03121
- https://arxiv.org/html/2508.03121v3
- https://arxiv.org/pdf/2508.03121v3
- https://arxiv.org/abs/2411.15231v2
- https://arxiv.org/html/2411.15231v2
- https://arxiv.org/pdf/2411.15231v2
- https://arxiv.org/pdf/2411.15231v1
- https://arxiv.org/abs/2610.06415
- https://arxiv.org/html/2610.06415v1
- https://arxiv.org/pdf/2610.06415v1
- https://arxiv.org/abs/2212.09849
- https://arxiv.org/pdf/2212.09849

Classical statistics, full text read:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC3667751/ (Won, Lim, Kim, Rajaratnam, condition-number-regularized covariance estimation)
- https://arxiv.org/pdf/1608.04123 (Peeters, van de Wiel, van Wieringen, spectral condition number plot)

Other papers, PDF downloaded and keyword-scanned and/or partly read (2603.21705 was downloaded but only its abstract was read):

- https://arxiv.org/pdf/2609.24612
- https://arxiv.org/pdf/2607.20561
- https://arxiv.org/pdf/2510.08987
- https://arxiv.org/pdf/2602.05536
- https://arxiv.org/pdf/2603.26299
- https://arxiv.org/pdf/2502.10339
- https://arxiv.org/pdf/2412.00081
- https://arxiv.org/pdf/2502.04959
- https://arxiv.org/pdf/2503.08099
- https://arxiv.org/pdf/2505.19892
- https://arxiv.org/pdf/2406.11385
- https://arxiv.org/pdf/2505.16148
- https://arxiv.org/pdf/2506.02478
- https://arxiv.org/pdf/2602.09413
- https://arxiv.org/pdf/2508.01148
- https://arxiv.org/pdf/2602.08564
- https://arxiv.org/pdf/2601.22285
- https://arxiv.org/pdf/2604.01329
- https://arxiv.org/pdf/2312.04339
- https://arxiv.org/pdf/2310.12808
- https://arxiv.org/pdf/2111.09832
- https://arxiv.org/pdf/2605.13030
- https://arxiv.org/pdf/2603.26317
- https://arxiv.org/pdf/2506.14262
- https://arxiv.org/pdf/2606.18627
- https://arxiv.org/pdf/2607.01689
- https://arxiv.org/pdf/2606.03391
- https://arxiv.org/pdf/2502.18821
- https://arxiv.org/pdf/2410.17961
- https://arxiv.org/pdf/2509.25712
- https://arxiv.org/pdf/2512.19320
- https://arxiv.org/pdf/2607.18026
- https://arxiv.org/pdf/2605.29101
- https://arxiv.org/pdf/2605.26693
- https://arxiv.org/pdf/2603.21705

Abstract / metadata only, via the arXiv API (https://export.arxiv.org/api/query?id_list=...):

- https://arxiv.org/abs/1508.06095
- https://arxiv.org/abs/2207.01139
- https://arxiv.org/abs/2006.00371
- https://arxiv.org/abs/2511.10943
- https://arxiv.org/abs/2605.01580
- https://arxiv.org/abs/2412.12153
- https://arxiv.org/abs/2502.12706
- https://arxiv.org/abs/2411.09263
- https://arxiv.org/abs/2405.07813
- https://arxiv.org/abs/2107.04205
- https://arxiv.org/abs/2507.18807
- https://arxiv.org/abs/2602.05787
- https://arxiv.org/abs/2402.02705

Search-result pages used only through the search tool's snippets (not opened by me; treat as secondary):

- https://arxiv.org/abs/2006.10593 (Li, Cai, Li, Trans-Lasso)
- https://arxiv.org/pdf/2002.06708 (Rosenman et al., shrinkage estimators for combining datasets)
- https://arxiv.org/pdf/2309.06727 (empirical Bayes double shrinkage)
- https://www.asc.ohio-state.edu/statistics/statgen/joul_aut2014/Efron2014.pdf (Efron 2014)
- https://ideas.repec.org/a/taf/jnlasa/v109y2014i507p991-1007.html (Efron 2014 citation record)
- https://business.okstate.edu/site-files/archive/docs/ecls-working-papers/OKSWPS1305.pdf (Stein-rule / pretest estimators)
- https://www.nottingham.ac.uk/economics/documents/discussion-papers/00-19.pdf (context for Green and Strawderman)
- https://bcc.impan.pl/wydawnictwa/preprints_impan/p630.pdf (Reginska, condition number of Tikhonov-regularized operators)
- https://www2.stat.duke.edu/courses/Fall18/sta721/publication/bayesridge (Bayesian ridge notes)
- https://arxiv.org/pdf/2602.09689 (Model soups need only one ingredient; MonoSoup snippet)
- https://arxiv.org/pdf/2605.14084 (CRANE, calibration-subset redraw check)
- UQ-Merge (Qu et al., Findings of ACL 2025): ACL Anthology listing seen in search results only; no arXiv id found
