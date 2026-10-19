# Appendix tables B1-B7: diagnostics of the aa/side track (condition-number merging)

Generated 2026-10-18 23:29 by `make_tables.py` (python 3.10.12, numpy 2.2.6, pandas 2.3.3); every number below is computed by that script from the raw result files under `results/aa` (worktree `/data/home/azpitarte/condmerge_aa`, branch aa/side, read-only). Source paths in each table are relative to the worktree root.

## Conventions used in all tables

- **Rule.** lambda_i^p = softmax_i(-beta * kappa_2(W_i^p)) per linear map p, merged weight W0 + alpha * sum_i lambda_i Delta_i with Delta_i = W_i - W0; alpha = 1 is the normalised average. beta = b / scale with scale = median over maps of the across-expert std of the rule's score; b in {0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, inf}; b = inf is per-map argmin selection; b = 0 is uniform.
- **Banks.** Full fine-tuning: 8 FusionBench CLIP ViT-B/32 experts (sun397, cars, resisc45, eurosat, svhn, gtsrb, mnist, dtd), 73 linear maps (12 x q,k,v,out_proj + 12 x fc1,fc2 + patch embedding; biases tied to their weight, LayerNorm/embeddings get the prior). LoRA: KnOTS rank-16 bank, 48 adapted attention maps (q,k,v,out of 12 blocks) unless a table says another bank (the soup part B uses 13 same-task LoRA variants per task).
- **Splits.** dev = 20% of each official test set (permutation seed 0), used for all fitting/selection/diagnostics (n per task: sun397 3970, cars 1608, resisc45 1260, eurosat 540, svhn 5206, gtsrb 2526, mnist 2000, dtd 376; total 17486; source `results/sanity/expert_matrix_dev.json`, master's file). test = the other 80%, touched only for frozen configurations and for the held-out evaluation of oracle fits. Oracle fits (B3) use dev labels, so their dev numbers are in-sample.
- **Information budgets.** A = weights only; B = pooled unlabeled inputs without task IDs; C = task-separated unlabeled inputs; D = labeled dev data (oracle/diagnostic only). Every table states which budget its rule/score uses.
- **Accuracy.** Every accuracy in B1-B7 is ABSOLUTE accuracy (fraction correct on the task's split, shown in %); 'avg acc' = unweighted mean of the 8 per-task accuracies. NORMALISED accuracy (merged accuracy / own expert's accuracy, averaged) is not used by any file of this scope; it only appears in master's files quoted as external references (B3a legend, X1). Soup tables (B5) score each soup on its own task only. CE = mean cross-entropy of the CLIP logits scaled by 100, in nats.
- **Correlations.** rho = Spearman rank correlation across the 8 experts inside a map, averaged over maps (within-map); 'between-expert' = Pearson correlation over experts of the mean within-map ranks; 'within-expert' = Pearson correlation of the ranks after removing each expert's mean rank (layer-specific part). Permutation p-values: one-sided where two columns '(null <= obs)' / '(null >= obs)' are given, two-sided where a single p is given; the number of shuffles is in each table's description.
- **Scores.** kappa_W = kappa_2(W_i) = sigma_1/sigma_q; smin_W = sigma_min(W_i); fro_D = ||Delta_i||_F; stable_rank_D, entropy_rank_D, kplus_D_0.01 = spread measures of the update's singular values; logkappa_rel = log kappa(W_i) - log kappa(W0); B-budget scores use the pooled input second moment C (B7).
- **Map labels** are shortened to L<layer>.<module> (L3.fc1 = encoder.layers.3.mlp.fc1.weight).
- **Reproduction.** `OMP_NUM_THREADS=4 /data/home/azpitarte/lora_merging_ica/.venv/bin/python make_tables.py` (CPU, about 30 seconds); also writes `csv/soup_full_grid.csv` and keeps scratch files in `_work/`.

## Index of tables

| ID | title | rows |
| :--- | :--- | ---: |
| B1a | Synthetic checks C1-C11 (math of the review on small linear systems) | 11 |
| B1b | All individual sub-checks (console lines of the script) | 41 |
| B1c | C11: condition number of W*+E vs weight-noise level s (synthetic 128x128) | 17 |
| B1d | C11: condition number of real CLIP ViT-B/32 base maps under added weight noise | 6 |
| B2a | Operating point of the gradient-preference diagnostic: the uniform merge at alpha = 1 and 2.4 | 9 |
| B2b | Within-map Spearman between gradient preference and each score (73 linear maps, 8 experts), alpha = 1 and 2.4 | 14 |
| B2c | Extra statistics of the kappa analysis: partial correlation given ‖Δ‖_F, kappa vs update norm, kappa vs sigma_min | 2 |
| B2d | Which 5 maps carry the first-order effect on dev loss, for every score | 14 |
| B2e | Expert-identity vs layer-specific parts of the score / gradient-preference association | 12 |
| B2f | Expert identity: mean within-map rank of each expert (gradient preference and scores) | 8 |
| B2g | Gradient preference vs every score (A: budget A weights-only scores, B: budget B data-supported scores), alpha = 1 and 2.4 | 14 |
| B2h | Per-map summary of gradient preference vs kappa by module type (full 146-row table is the CSV) | 20 |
| B2i | Which expert the dev loss prefers, aggregated over the 73 linear maps | 8 |
| B2j | Task-resolved loss sensitivity: derivative of each task's dev CE when expert i's weight is raised in all 73 linear maps | 16 |
| B3a | Oracle weights fitted with dev labels: headline accuracy, full fine-tuning and LoRA banks | 11 |
| B3b | Oracle weights: per-task accuracy on dev and test | 22 |
| B3c | Uniform-lambda merge: dev accuracy vs the global scale alpha | 11 |
| B3d | Oracle seed agreement (full fine-tuning; two independent fits per family) | 3 |
| B3e | Oracle concentration: effective number of experts n_eff = exp(entropy of lambda) per map, by map type | 9 |
| B3f | Oracle concentration: spread of n_eff and size of the largest weight | 9 |
| B3g | Oracle weights per expert: mean lambda over maps, and how many maps each expert 'owns' (largest lambda) | 9 |
| B3g2 | Oracle ownership: number of attention / MLP maps where each expert has the largest lambda | 10 |
| B3h | Oracle lambda vs kappa and other scores, full fine-tuning (4 layerwise fits) | 20 |
| B3i | Oracle lambda vs kappa and other scores, LoRA bank (48 adapted attention maps) | 14 |
| B4a | Score vs targets: every row of score_vs_targets.json (4 targets x 14 scores) | 56 |
| B4b | Re-derivation of report section 2b: master's tail-robust ratio kappa_q90 (and kappa_q50) against the same targets | 12 |
| B5a | Shared-target soups: the pools (setup) | 6 |
| B5b | Part A: how kappa and the stable rank of the update move with the injected noise | 36 |
| B5c | Part B: the real LoRA variants -- own-task quality and its relation to kappa / update spread | 9 |
| B5d1 | Rule x temperature grid, pool A, first run (mild noise) | 12 |
| B5d2 | Rule x temperature grid, pool A-strong (rules dmedsv, dnoiseshare, mp_invvar, global:mp_invvar not run; values identical to the same rows of A-strong-v2) | 12 |
| B5d3 | Rule x temperature grid, pool A-normmatched (A-nm) | 16 |
| B5d4 | Rule x temperature grid, pool A-strong-v2 (superset of A-strong, identical values on the common rows) | 16 |
| B5d5 | Rule x temperature grid, pool B, first run (interrupted) (interrupted run: only sun397 is complete, the cars rows are partial) | 13 |
| B5d6 | Rule x temperature grid, pool B-v2 (real LoRA variants) (superset of the interrupted first run, identical values on the common rows) | 17 |
| B5e1 | Per-task accuracy, pool A, first run (mild noise) | 22 |
| B5e2 | Per-task accuracy, pool A-strong-v2 | 28 |
| B5e3 | Per-task accuracy, pool A-normmatched (A-nm) | 28 |
| B5e4 | Per-task accuracy, pool B-v2 (real LoRA variants) | 29 |
| B5f1 | Weight each rule puts on the copies, pool A-strong-v2 | 19 |
| B5f2 | Weight each rule puts on the copies, pool A-normmatched (A-nm) | 19 |
| B5g | Frozen TEST evaluation of the shared-target soups (configurations fixed before any test look) | 25 |
| B6a | Tail-stress edit: size of the function-null edit | 2 |
| B6b | Tail-stress edit: the edited expert alone, accuracy on all 8 tasks before and after the edit | 6 |
| B6c | Tail-stress: every row of both JSON files (literal rule on the original and on the tail-edited expert) | 16 |
| B6d | Tail-stress summary: weight on the edited expert and merged accuracy before -> after the edit | 8 |
| B7a | Data-supported scores (budget B): distribution over the 584 (map, expert) rows | 8 |
| B7b | Data-supported scores by module type (median over maps x experts) | 8 |
| B7c | Data-supported scores by expert (median over the 73 maps) | 8 |
| B7d | Spread of each data-supported statistic across the 8 experts inside a map | 7 |
| B7e | Data-weighted vs plain condition number of the BASE map, and numerically singular rows | 8 |
| B7f | Agreement between data-supported statistics: mean within-map Spearman across the 8 experts | 7 |
| X1 | Cross-check of every number of the report that lives in this scope (recomputed from the raw files) | 176 |
| X2 | Discrepancies between the report and the files (all non-OK rows of X1, plus omissions found while building B1-B7) | 23 |
| X3 | Claims of the report that cannot be verified from the files in this scope | 4 |

## B1. Synthetic checks

### B1a. Synthetic checks C1-C11 (math of the review on small linear systems)

**What varies / split / metric / units.** Not a data experiment: each check compares a numerical value with an analytic one in a small synthetic system (seed 0, CPU); the quantity measured is shown per check; no split, no accuracy; 'result' = PASS/FAIL of the script's own asserts.

**Source.** `results/aa/synthetic/synthetic_checks.json`; `scripts/aa_synthetic_checks.py`

| ID | review ref | claim tested (script docstring) | setup | measured quantity | pass criterion | result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| C1 | 3.3 eq.12-13 | shared target, independent noise: optimal lambda = inverse variance (Monte Carlo + QP) | 4 unbiased estimates of one d=50 target with independent Gaussian errors of total variance v=[0.5, 1, 2, 4]; QP over the simplex vs closed form inverse variance; Monte Carlo risk with 20000 draws (opt and uniform weights) | lam_opt=[0.5333, 0.2667, 0.1333, 0.0667]; analytic risk opt 0.2667 vs uniform 0.4688; Monte Carlo opt 0.2666 vs 0.2667, uniform 0.4681 vs 0.4688 (MC vs analytic) | QP optimum == inverse variance (atol 1e-6); MC risk within 3% of analytic, opt and uniform | PASS (3/3 sub-checks) |
| C2 | 3.3 eq.17-22 | controlled LS designs Z_i = diag(1,..,1,1/k_i): v_i = nu^2 (d-1+k_i^2); exp(-beta*kappa) vs power law vs uniform vs oracle; eq. 21 bounds; scale and spectrum counterexamples | LS designs Z_i = diag(1,..,1,1/kappa_i), d=8, nu^2=1, kappa=[1, 2, 4, 8, 16] so v_i=nu^2(d-1+kappa_i^2)=[8, 11, 23, 71, 263]; risk = sum lambda^2 v; compares inverse-variance oracle, uniform, best exp(-beta*kappa) (400 betas in [1e-3,10]), best kappa^-p (401 p in [0,4]) and the reversed rule; eq.21 bounds on 200 random 20xd designs; two counterexamples | risk: oracle 3.6065, uniform 15.0400, best exp(-beta*kappa) 3.6383 (beta=0.3135), best kappa^-p 3.8053 (p=1.06), best reversed 15.2470 | eq.22 variance formula; eq.21 bounds on 200 designs; best decreasing rules < uniform; reversed >= uniform; counterexamples exact | PASS (6/6 sub-checks) |
| C3 | 3.3 eq.14-16 | correlated / duplicated experts: GLS weights can be negative; (1+rho)/2; softmax duplicate bias | 2x2 covariance C=[[1,1.2],[1.2,2]] (GLS weights, negativity, simplex clipping); Var((e1+e2)/2)=(1+rho)/2 at rho in {0,0.5,0.9}; a duplicated expert under uniform softmax over 3 items | GLS weights [1.3333, -0.3333]; simplex-QP [1.0, 0.0] | variance (1+rho)/2 exact; GLS has a negative weight; simplex QP clips to [1,0]; duplicate doubles group mass | PASS (6/6 sub-checks) |
| C4 | 3.2 eq.8 | complementary vs cancelling orientations: equal kappa, opposite merge outcome | W1=diag(1,eps), W2=diag(eps,1) with eps=1e-2: equal kappa, kappa of the average, the cancelling pair W2=-W1, scale invariance kappa(7*W1) | analytic: kappa(W1)=kappa(W2)=1/eps=100, kappa((W1+W2)/2)=1, (W1-W1)/2=0 (no numbers stored in the JSON) | equal kappa; complementary average kappa=1; cancelling pair averages to 0; kappa scale invariant | PASS (4/4 sub-checks) |
| C5 | 3.8 Fig.1L | distinct tasks: penalising the high-kappa expert is strictly worse | two 'tasks' W1=diag(1,1e-4) (kappa 1e4) and W2=I (kappa 1) with isotropic inputs; loss(lam)=0.5(‖W-W1‖²+‖W-W2‖²), W=lam*W1+(1-lam)*W2 on a 101-point grid; the rule softmax(-1e-3*kappa) puts weight lam on the high-kappa expert | lam(high-kappa expert) under softmax(-1e-3*kappa) = 4.54e-05; excess loss over equal weights = 0.2499 | excess == (lam-1/2)^2(1-1e-4)^2 (atol 1e-12); rule moves weight off the high-kappa expert and loses | PASS (2/2 sub-checks) |
| C6 | 3.5 eq.26-27 | heterogeneous quadratics: no scalar mixture reaches the curvature-weighted optimum | two quadratics with targets t1=(1,1), t2=(2,2) and Hessians H1=diag(100,1), H2=diag(1,100) (kappa 100 each); curvature-weighted optimum vs best point of span{t1,t2} with free scale | theta*=[1.0099, 1.9901] (loss 0.4950); best loss in span{t1,t2} = 12.6250 | optimum == (102/101, 201/101); best-span loss > optimum + 1e-3; both kappa(H)=100 | PASS (3/3 sub-checks) |
| C7 | 3.4 eq.23-24 | softmax = unique minimiser of the entropy-regularised linear cost | 5 random costs c~3*N(0,1), random prior q~Dirichlet(1), tau=0.7; SLSQP minimiser of lam.c + tau*KL(lam, q) on the simplex vs prior-weighted softmax | max abs difference SLSQP vs closed form = 3.06e-08 (not in JSON; from re-run) | optimum == prior-weighted softmax (atol 1e-5); shift invariance and scale sensitivity | PASS (2/2 sub-checks) |
| C8 | 3.6 eq.30-35 | regression merge B G^-1, ridge, first-order perturbation bound, kappa(Z^T Z)=kappa(Z)^2 | regression merge B G^-1 with d_out=6, d_in=10, k=3 experts (200 samples each, lognormal column scales, pi=(0.2,0.3,0.5)); vs stacked least squares, identical C_i, first-order perturbation bound (1e-6 perturbations), ridge rho=0.1, and kappa(Z^T Z)=kappa(Z)^2 on a 30x5 matrix | first-order perturbation, norm of change <= bound: 5.669e-06 <= 2.386e-05 (not in JSON; from re-run) | all 5 sub-checks (see B1b), tolerances 1e-8..1e-9, bound x 1.01 | PASS (5/5 sub-checks) |
| C9 | 3.7 eq.36-37 | LoRA gauge: factor kappas change, product spectrum invariant; QR-core spectrum exact | LoRA factors B (64x8), A (8x48), random invertible gauge R = G1 diag(geomspace(1,100,8)) G2 with Gaussian G1, G2: product invariance, factor kappas, QR-core spectrum vs product spectrum, kappa2 of the rank-8 product, averaging factors vs products, concatenated adapter | kappa(A) 2.0->1527.7, kappa(B) 1.7->1696.3; sigma_min/sigma_max of the rank-8 product: 5.9e-18 (not in JSON; from re-run) | all 6 sub-checks (see B1b); factor kappa changes > 10% | PASS (6/6 sub-checks) |
| C10 | 3.2 eq.10 | kappa is coordinate dependent while responses/functional error are invariant | 6x6 random W, W*, input second moment C=LL^T+0.1I and reparameterisation R=diag(geomspace(1,1000,6))Q: functional error tr((W-W*)C(W-W*)ᵀ) vs kappa(W) before/after (W->W R^-1, C->R C R^T) | kappa(W) 12.2 -> 1251.2 (not in JSON; from re-run) | functional error invariant (rel 1e-8); kappa changes by > 50% | PASS (2/2 sub-checks) |
| C11 | new bridge | weight-domain noise: kappa(W* + E) vs noise level s for near-singular W* -- noise can IMPROVE conditioning (smoothed-analysis regime), reversing the rule's intent | n=128, W*=U diag(sigma) V^T with sigma=geomspace(1,1e-6) (near-singular) or geomspace(1,0.5) (well-conditioned), Gaussian noise of per-entry std s in 17 levels geomspace(1e-9,1e-1), 20 draws per level, median kappa; plus 5 real CLIP base maps x 5 noise levels (5 draws) | near-singular: median kappa 6.71e+06 at s=1e-6 -> 1.53e+03 at s=1e-2 (monotone decrease in [1e-6,1e-2]); well-conditioned: 2.01 at s=1e-3 -> 420 at s=0.1; real CLIP L0 k_proj: kappa0 1.04e+08 -> 9.87e+03 at s=1e-2 | near-singular: kappa decreases with s on s in [1e-6,1e-2]; well-conditioned: increases for s >= 1e-3 | PASS (2/2 sub-checks) |

**Legend.**
- review ref = section/equation of the review the check tests (from the script docstring).
- The stored JSON records numbers for C1, C2, C3, C5, C6, C11 and C11_real_base and `fails: []` (no failing check). C4 and C7-C10 store no numbers; their numbers and every individual PASS line come from re-running a COPY of the script in the scratchpad (seed 0): the re-run JSON equals the stored JSON on all shared keys (max relative deviation 0.0e+00); C11_real_base cannot be re-run (needs the base-weight npz passed as argv[2]) and is taken from the stored JSON only.
- Stored `fails` = []; re-run console ends with 'ALL CHECKS PASS'.

### B1b. All individual sub-checks (console lines of the script)

**What varies / split / metric / units.** One row per `check()` call executed; PASS/FAIL against the script's analytic value; no split or accuracy involved.

**Source.** `scripts/aa_synthetic_checks.py`

| check | status | text printed by the script (name + detail) |
| :--- | :--- | :--- |
| C1 | PASS | C1 QP optimum = inverse variance [0.5333 0.2667 0.1333 0.0667] |
| C1 | PASS | C1 MC risk matches analytic (opt) 0.2666 vs 0.2667 |
| C1 | PASS | C1 MC risk matches analytic (uniform) 0.4681 vs 0.4688 |
| C2 | PASS | C2 eq.22 variance formula [  8.  11.  23.  71. 263.] |
| C2 | PASS | C2 eq.21 bounds hold on 200 random designs  |
| C2 | PASS | C2 kappa-decreasing weights beat uniform in the shared-target LS family oracle 3.607 exp 3.638 pow 3.805 uniform 15.040 |
| C2 | PASS | C2 reversed preference never beats uniform here  |
| C2 | PASS | C2 scale counterexample (kappa 10 design far more precise than kappa 1) v(I)=2.0, v(100diag(1,.1))=0.0101 |
| C2 | PASS | C2 same kappa, different variance (201 vs 102)  |
| C3 | PASS | C3 Var((e1+e2)/2) = (1+rho)/2 at rho=0.0  |
| C3 | PASS | C3 Var((e1+e2)/2) = (1+rho)/2 at rho=0.5  |
| C3 | PASS | C3 Var((e1+e2)/2) = (1+rho)/2 at rho=0.9  |
| C3 | PASS | C3 unconstrained GLS has a negative weight [ 1.3333 -0.3333] |
| C3 | PASS | C3 simplex QP clips to a vertex [1. 0.] |
| C3 | PASS | C3 duplicated expert doubles its group mass under uniform softmax  |
| C4 | PASS | C4 equal individual kappa  |
| C4 | PASS | C4 complementary average is perfectly conditioned  |
| C4 | PASS | C4 cancelling pair (W2=-W1) averages to zero with unchanged kappas  |
| C4 | PASS | C4 kappa is scale invariant (cannot see over-large updates)  |
| C5 | PASS | C5 excess loss = (lam-1/2)^2 (1-1e-4)^2  |
| C5 | PASS | C5 the kappa rule moves weight away from the high-kappa expert and loses lam_highk=0.000, excess=0.2499 |
| C6 | PASS | C6 curvature-weighted optimum = (102/101, 201/101)  |
| C6 | PASS | C6 no linear combination of experts reaches it best-span 12.6250 vs opt 0.4950 |
| C6 | PASS | C6 both Hessians have kappa 100  |
| C7 | PASS | C7 entropy-regularised optimum = prior-weighted softmax max err 3.06e-08 |
| C7 | PASS | C7 shift invariance / scale sensitivity  |
| C8 | PASS | C8 RegMean normal-equation solution = stacked least squares  |
| C8 | PASS | C8 identical C_i reduce to pi-weighted averaging  |
| C8 | PASS | C8 eq.34 first-order perturbation bound 5.669e-06 <= 2.386e-05 |
| C8 | PASS | C8 ridge solution solves the regularised objective  |
| C8 | PASS | C8 kappa(Z^T Z) = kappa(Z)^2  |
| C9 | PASS | C9 product invariant under gauge R  |
| C9 | PASS | C9 factor kappas change under gauge kappa(A) 2.0->1527.7, kappa(B) 1.7->1696.3 |
| C9 | PASS | C9 QR-core spectrum equals the product's nonzero spectrum  |
| C9 | PASS | C9 ordinary kappa2 of a rank-r product is numerically infinite sigma_min/sigma_max=5.9e-18 |
| C9 | PASS | C9 averaging factors is not averaging products  |
| C9 | PASS | C9 concatenated adapter is exact (rank <= 2r)  |
| C10 | PASS | C10 functional error eq.10 invariant under reparameterisation  |
| C10 | PASS | C10 kappa changes under the same reparameterisation 12.2 -> 1251.2 |
| C11 | PASS | C11 near-singular W*: kappa decreases as weight noise increases (smoothed analysis regime) kappa 6.71e+06 @s=1e-6 -> 1.53e+03 @s=1e-2 |
| C11 | PASS | C11 well-conditioned W*: kappa increases with weight noise  |

**Legend.**
- Lines come from re-running a copy of the script (seed 0) in the scratchpad (log: report_build/B_diagnostics/_work/synthetic_rerun.log); its JSON equals the stored synthetic_checks.json (B1a legend).

### B1c. C11: condition number of W*+E vs weight-noise level s (synthetic 128x128)

**What varies / split / metric / units.** Varied: Gaussian noise std s per entry (17 levels); metric: kappa_2 of W*+E (median, 10% and 90% quantile over 20 draws), dimensionless; no split.

**Source.** `results/aa/synthetic/synthetic_checks.json`

| s | near-singular W*: kappa median | q10 | q90 | well-conditioned W*: kappa median |
| :--- | ---: | ---: | ---: | ---: |
| 1e-09 | 1e+06 | 9.987e+05 | 1.001e+06 | 2 |
| 3.16e-09 | 9.994e+05 | 9.979e+05 | 1.005e+06 | 2 |
| 1e-08 | 9.996e+05 | 9.874e+05 | 1.014e+06 | 2 |
| 3.16e-08 | 1.009e+06 | 9.639e+05 | 1.039e+06 | 2 |
| 1e-07 | 1.116e+06 | 9.868e+05 | 1.269e+06 | 2 |
| 3.16e-07 | 2.865e+06 | 1.884e+06 | 7.202e+06 | 2 |
| 1e-06 | 6.713e+06 | 2.332e+06 | 2.113e+07 | 2 |
| 3.16e-06 | 1.905e+06 | 1.146e+06 | 7.324e+06 | 2 |
| 1e-05 | 1.055e+06 | 4.493e+05 | 1.165e+07 | 2 |
| 3.16e-05 | 3.71e+05 | 1.7e+05 | 7.887e+05 | 2 |
| 0.0001 | 9.967e+04 | 4.517e+04 | 5.475e+05 | 2.0001 |
| 0.000316 | 6.444e+04 | 2.914e+04 | 4.823e+05 | 2.0002 |
| 0.001 | 2.531e+04 | 8790 | 1.289e+05 | 2.0054 |
| 0.00316 | 4940 | 2525 | 1.208e+04 | 2.0485 |
| 0.01 | 1528 | 842.5 | 4021 | 2.3732 |
| 0.0316 | 832.2 | 332.8 | 2627 | 6.0845 |
| 0.1 | 491.3 | 159.2 | 1441 | 419.94 |

**Legend.**
- near-singular W*: singular values geomspace(1, 1e-6), kappa(W*) = 1e6; well-conditioned: geomspace(1, 0.5), kappa(W*) = 2.

### B1d. C11: condition number of real CLIP ViT-B/32 base maps under added weight noise

**What varies / split / metric / units.** Varied: Gaussian noise std s added to the base weight (5 levels, 5 draws, median); metric: kappa_2, dimensionless; base-model maps, no split.

**Source.** `results/aa/synthetic/synthetic_checks.json`

| map | kappa(W0) | sigma_min(W0) | kappa median, s=1e-05 | kappa median, s=0.0001 | kappa median, s=0.001 | kappa median, s=0.003 | kappa median, s=0.01 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0.k_proj | 1.044e+08 | 3.93e-08 | 3.677e+06 | 5.257e+05 | 9.485e+04 | 2.593e+05 | 9866 |
| L0.v_proj | 3.076e+05 | 9.2e-06 | 9.283e+05 | 1.212e+05 | 4.445e+04 | 1.671e+04 | 1.356e+04 |
| L0.q_proj | 5.769e+07 | 8.65e-08 | 1.265e+07 | 5.487e+05 | 3.041e+05 | 4.327e+04 | 2.18e+04 |
| L6.k_proj | 6792 | 0.000241 | 6760 | 7328 | 9893 | 3290 | 3298 |
| L6.fc1 | 14.38 | 0.229 | 14.38 | 14.38 | 14.03 | 12.42 | 7.548 |
| L11.out_proj | 2.116e+04 | 7.9e-05 | 2.261e+04 | 1.567e+04 | 7833 | 4434 | 3481 |

**Legend.**
- map names are shortened: L<layer>.<module> (e.g. L0.k_proj = encoder.layers.0.self_attn.k_proj.weight).

## B2. Gradient preference vs every score

### B2a. Operating point of the gradient-preference diagnostic: the uniform merge at alpha = 1 and 2.4

**What varies / split / metric / units.** Varied: global scale alpha of the uniform (lambda = 1/8) merge W0 + alpha * mean(Delta); split: dev (20% of each official test set); metric: absolute per-task accuracy (%) and mean dev cross-entropy at CLIP logit scale 100 (nats); budget D (diagnostic).

**Source.** `results/aa/grad_pref/summary.json`

| task | acc (%) alpha=1 | acc (%) alpha=2.4 | CE alpha=1 | CE alpha=2.4 |
| :--- | ---: | ---: | ---: | ---: |
| sun397 | 66.83 | 57.48 | 1.1902 | 1.7136 |
| cars | 63.12 | 57.34 | 1.1338 | 1.4692 |
| resisc45 | 69.76 | 63.10 | 1.0085 | 1.4506 |
| eurosat | 76.11 | 74.07 | 0.6906 | 0.8096 |
| svhn | 65.21 | 78.93 | 1.0999 | 0.6801 |
| gtsrb | 56.73 | 70.03 | 1.4767 | 1.2037 |
| mnist | 86.35 | 96.50 | 0.5999 | 0.1463 |
| dtd | 54.52 | 49.47 | 1.6690 | 1.9182 |
| **mean (avg_acc / avg_ce)** | 67.33 | 68.36 | 1.1086 | 1.1739 |

**Legend.**
- Gradients d(mean dev CE)/d(lambda_i^p) are taken at these two uniform merges (script scripts/aa_grad_preference.py); the objective is the mean over the 8 tasks of each task's mean dev CE.
- avg_acc is the unweighted mean of the 8 absolute accuracies (not normalised by the expert's own accuracy).

### B2b. Within-map Spearman between gradient preference and each score (73 linear maps, 8 experts), alpha = 1 and 2.4

**What varies / split / metric / units.** Varied: score (7), alpha (1, 2.4); split: dev; metric: mean over the 73 maps of the across-expert Spearman rho(pref, score) (dimensionless, +1..-1) with one-sided permutation p-values (10000 shuffles of the scores within each map), first-order effect on dev CE of moving towards the rule, budget D target / budget A scores.

**Source.** `results/aa/grad_pref/analysis_grad_vs_kappa.json`

| alpha | score | mean within-map rho(pref, score) | % maps with rho<0 | p (null <= obs) | p (null >= obs) | first-order dL | p (null dL <= obs) | share of total abs(dL) from top-5 maps | largest-abs(dL) map: dL |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | kappa | +0.0453 | 42.5 | 0.8542 | 0.1458 | 3.039e+05 | 0.4795 | 99.5 | L0.q_proj: 1.22e+06 |
| 1 | log_kappa | +0.0453 | 42.5 | 0.8409 | 0.1592 | -0.00767 | 0.4430 | 36.1 | L3.v_proj: -0.019 |
| 1 | kappa_layer_std | +0.0453 | 42.5 | 0.8492 | 0.1511 | 0.009829 | 0.5226 | 30.6 | L3.fc1: -0.0724 |
| 1 | fro_D | +0.1168 | 32.9 | 0.9965 | 0.0035 | 0.01177 | 0.9088 | 38.1 | L0.fc2: -0.00322 |
| 1 | smin_W | +0.0007 | 52.1 | 0.5108 | 0.4892 | -1.832e-06 | 0.4628 | 38.1 | L8.fc1: -1.93e-05 |
| 1 | stable_rank_D | -0.4273 | 83.6 | 0.0000 | 1.0000 | -9.872 | 0.0000 | 31.7 | L3.fc1: -1.02 |
| 1 | entropy_rank_D | -0.4746 | 82.2 | 0.0000 | 1.0000 | -101.3 | 0.0000 | 29.8 | L4.fc1: -8.01 |
| 2.4 | kappa | +0.0453 | 49.3 | 0.8394 | 0.1614 | 3.243e+06 | 0.8680 | 99.2 | L0.q_proj: 2.97e+06 |
| 2.4 | log_kappa | +0.0453 | 49.3 | 0.8452 | 0.1550 | -0.01888 | 0.4146 | 33.0 | L11.v_proj: 0.0329 |
| 2.4 | kappa_layer_std | +0.0453 | 49.3 | 0.8448 | 0.1554 | 0.9222 | 0.9515 | 57.1 | L1.fc2: 0.355 |
| 2.4 | fro_D | -0.0564 | 56.2 | 0.0988 | 0.9013 | -0.0645 | 0.0133 | 49.0 | L0.fc2: -0.0174 |
| 2.4 | smin_W | -0.1024 | 49.3 | 0.0095 | 0.9907 | -0.0001261 | 0.0099 | 43.7 | L1.fc2: -4.62e-05 |
| 2.4 | stable_rank_D | +0.2420 | 24.7 | 1.0000 | 0.0000 | 13.85 | 1.0000 | 60.6 | L2.fc2: 3.71 |
| 2.4 | entropy_rank_D | +0.2851 | 20.5 | 1.0000 | 0.0000 | 205.9 | 1.0000 | 63.1 | L2.fc2: 47.9 |

**Legend.**
- pref = -(dL/dlambda centred across experts), using the tied weight+bias derivative; positive pref = moving mass to that expert lowers dev CE. The literal rule predicts a NEGATIVE correlation with kappa.
- p (null <= obs): fraction of permutation draws with mean rho <= the observed value (small = evidence in the direction the rule predicts); p (null >= obs): the reverse direction.
- first-order dL = -sum_p sum_i g_i^p (s_i^p - mean_i s^p) (g = tied gradient of the mean dev CE), the derivative of dev CE along the rule's direction up to the positive factor beta/K; negative = the rule's direction lowers dev CE. Its unit is (CE nats) x (score unit), so only the sign, the p-value and the concentration are comparable across scores.
- kappa = kappa_2(W_i); log_kappa; kappa_layer_std = within-map z-score of kappa; fro_D = ‖Δ_i‖_F; smin_W = sigma_min(W_i); stable_rank_D, entropy_rank_D = spread of Delta_i's spectrum.
- map labels are shortened: L<layer>.<module>.

**Notes.**
- kappa, log_kappa and kappa_layer_std have identical within-map ranks, hence identical mean rho; they differ in dL and p(dL).

### B2c. Extra statistics of the kappa analysis: partial correlation given ‖Δ‖_F, kappa vs update norm, kappa vs sigma_min

**What varies / split / metric / units.** Varied: alpha (target only for the first two columns); split: dev; metric: mean (over maps) Spearman correlations, dimensionless.

**Source.** `results/aa/grad_pref/analysis_grad_vs_kappa.json`

| alpha | partial rho(pref, kappa given ‖Δ‖_F) | % maps negative | rho(kappa, ‖Δ‖_F) all maps | ... maps with kappa0 > 1e4 | ... maps with kappa0 <= 1e4 | n maps kappa0 > 1e4 | rho(kappa, sigma_min(W)) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +0.0258 | 46.6 | +0.0408 | -0.0838 | +0.1139 | 27 | -0.9061 |
| 2.4 | +0.0202 | 50.7 | +0.0408 | -0.0838 | +0.1139 | 27 | -0.9061 |

**Legend.**
- The last 5 columns are properties of the experts' weights and do not depend on alpha (identical in both blocks of the JSON).

### B2d. Which 5 maps carry the first-order effect on dev loss, for every score

**What varies / split / metric / units.** Varied: score (7) and alpha (1, 2.4); split: dev; metric: the 5 maps with the largest absolute first-order dL of moving towards softmax(-beta*score) (per-map contribution, up to the factor beta/K; negative lowers dev CE) and their share of the total abs(dL) over the 73 maps; budget A rule / D target.

**Source.** `results/aa/grad_pref/analysis_grad_vs_kappa.json`

| alpha | score | share of total abs(dL) from these 5 maps (%) | map 1: dL | map 2: dL | map 3: dL | map 4: dL | map 5: dL |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- | :--- |
| 1 | kappa | 99.5 | L0.q_proj: 1.22e+06 | L0.k_proj: -9.65e+05 | L2.k_proj: 3.9e+04 | L4.q_proj: 1.43e+04 | L7.q_proj: -4.43e+03 |
| 1 | log_kappa | 36.1 | L3.v_proj: -0.019 | L8.out_proj: -0.0152 | L10.v_proj: 0.0133 | L8.v_proj: 0.0112 | L7.k_proj: -0.00916 |
| 1 | kappa_layer_std | 30.6 | L3.fc1: -0.0724 | L3.fc2: 0.0629 | L8.fc1: 0.0516 | L5.fc2: -0.0511 | L4.fc2: -0.0463 |
| 1 | fro_D | 38.1 | L0.fc2: -0.00322 | embeddings.patch_embedding: -0.00293 | L4.fc2: 0.00278 | L7.fc1: 0.00213 | L6.fc1: 0.00185 |
| 1 | smin_W | 38.1 | L8.fc1: -1.93e-05 | L3.fc1: 6.79e-06 | L7.fc1: -6.52e-06 | L4.fc1: 6.39e-06 | L6.fc1: 6.07e-06 |
| 1 | stable_rank_D | 31.7 | L3.fc1: -1.02 | L4.fc1: -0.869 | L5.fc1: -0.699 | L4.fc2: -0.482 | L3.fc2: -0.468 |
| 1 | entropy_rank_D | 29.8 | L4.fc1: -8.01 | L3.fc1: -7.86 | L5.fc1: -7.37 | L3.fc2: -6.83 | L6.fc1: -6.78 |
| 2.4 | kappa | 99.2 | L0.q_proj: 2.97e+06 | L2.k_proj: 2.56e+05 | L0.k_proj: 2.31e+04 | L5.k_proj: -1.7e+04 | L4.q_proj: 1.3e+04 |
| 2.4 | log_kappa | 33.0 | L11.v_proj: 0.0329 | L9.out_proj: -0.029 | L7.out_proj: -0.026 | L1.v_proj: -0.0249 | L5.k_proj: -0.0182 |
| 2.4 | kappa_layer_std | 57.1 | L1.fc2: 0.355 | L0.fc2: 0.332 | L2.fc2: 0.314 | L6.fc2: -0.265 | L7.fc2: 0.164 |
| 2.4 | fro_D | 49.0 | L0.fc2: -0.0174 | L2.fc2: -0.00853 | L6.fc2: -0.00736 | L1.fc2: 0.00702 | L5.fc2: -0.00608 |
| 2.4 | smin_W | 43.7 | L1.fc2: -4.62e-05 | L8.fc1: -2.24e-05 | L2.fc1: -1.29e-05 | L11.v_proj: -1.19e-05 | L2.fc2: -1.11e-05 |
| 2.4 | stable_rank_D | 60.6 | L2.fc2: 3.71 | L1.fc2: 2.7 | L2.fc1: 2 | L0.fc2: 1.07 | L6.fc2: 0.773 |
| 2.4 | entropy_rank_D | 63.1 | L2.fc2: 47.9 | L1.fc2: 36 | L0.fc2: 34.2 | L6.fc2: 18.5 | L2.fc1: 12.5 |

**Legend.**
- For the literal score kappa these 5 maps carry 99.54% (alpha=1) and 99.16% (alpha=2.4) of the total abs(dL) over the 73 maps; all listed kappa maps are attention q_proj/k_proj maps: True (their kappa(W0) is in B2h / per_map_grad_vs_kappa.csv).
- dL is in units of (CE nats) x (score unit), so magnitudes are comparable only within a score.

### B2e. Expert-identity vs layer-specific parts of the score / gradient-preference association

**What varies / split / metric / units.** Varied: score (6), alpha (1, 2.4); split: dev; metric: Pearson correlation of within-map ranks, dimensionless: between-expert (8 points = expert identity) and within-expert (pooled residual ranks after removing each expert's mean rank = layer-specific part) with permutation p-values (5000 shuffles of maps inside each expert column).

**Source.** `results/aa/grad_pref/decompose_pref.json`

| alpha | score | between-expert corr (identity) | within-expert corr (layer-specific) | p (null <= obs) | p (null >= obs) | null sd | mean per-expert Spearman over maps |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | kappa_W | -0.284 | +0.0954 | 0.9834 | 0.0166 | 0.0443 | +0.0136 |
| 1 | fro_D | +0.154 | +0.0991 | 0.9862 | 0.0138 | 0.0441 | +0.0778 |
| 1 | stable_rank_D | -0.848 | +0.0253 | 0.7090 | 0.2910 | 0.0437 | -0.0084 |
| 1 | entropy_rank_D | -0.850 | +0.0265 | 0.7316 | 0.2684 | 0.0432 | -0.0006 |
| 1 | smin_W | +0.577 | -0.0901 | 0.0214 | 0.9786 | 0.0441 | -0.0123 |
| 1 | kplus_D_0.01 | +0.776 | +0.0220 | 0.7046 | 0.2954 | 0.0436 | -0.0040 |
| 2.4 | kappa_W | -0.305 | +0.0777 | 0.9538 | 0.0462 | 0.0451 | -0.0017 |
| 2.4 | fro_D | -0.268 | +0.1228 | 0.9976 | 0.0024 | 0.0447 | +0.0228 |
| 2.4 | stable_rank_D | +0.619 | +0.0226 | 0.6944 | 0.3056 | 0.0443 | +0.0181 |
| 2.4 | entropy_rank_D | +0.679 | +0.0223 | 0.6940 | 0.3060 | 0.0448 | +0.0179 |
| 2.4 | smin_W | +0.112 | -0.1275 | 0.0020 | 0.9980 | 0.0439 | -0.0025 |
| 2.4 | kplus_D_0.01 | -0.610 | -0.0716 | 0.0524 | 0.9476 | 0.0442 | -0.0432 |

**Legend.**
- Ranks are taken across the 8 experts inside each map; 'between-expert' correlates each expert's mean preference rank with its mean score rank (8 points); 'within-expert' correlates the residual ranks after removing those expert means.
- p (null <= obs) / (null >= obs): one-sided, null = scores permuted over maps independently inside each expert column (keeps each expert's score distribution). Two-sided p-values for the same quantity from a separate 2000-shuffle null are in B2g.
- mean per-expert Spearman over maps: for each expert Spearman(pref / max abs(pref), score) over the 73 maps, averaged over the 8 experts.

### B2f. Expert identity: mean within-map rank of each expert (gradient preference and scores)

**What varies / split / metric / units.** Varied: expert; split: dev; metric: mean over the 73 linear maps of the expert's rank among the 8 experts (1 = lowest value, 8 = highest), dimensionless; preference rank high = the dev loss most wants more weight on this expert.

**Source.** `results/aa/grad_pref/decompose_pref.json`

| expert | pref rank alpha=1 | pref rank alpha=2.4 | kappa_W rank | fro_D rank | stable_rank_D rank | entropy_rank_D rank | smin_W rank | kplus_D_0.01 rank |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sun397 | 2.562 | 4.562 | 5.219 | 7.548 | 7.849 | 7.945 | 3.726 | 3.014 |
| cars | 2.438 | 4.836 | 4.068 | 2.630 | 6.370 | 6.603 | 4.589 | 3.411 |
| resisc45 | 4.329 | 5.466 | 4.534 | 4.836 | 5.726 | 5.452 | 4.562 | 3.658 |
| eurosat | 4.384 | 4.658 | 4.411 | 3.233 | 2.973 | 3.247 | 4.562 | 5.808 |
| svhn | 5.945 | 3.151 | 4.205 | 7.055 | 2.151 | 1.877 | 5.055 | 5.630 |
| gtsrb | 6.260 | 5.301 | 3.959 | 5.753 | 3.685 | 3.726 | 5.288 | 4.932 |
| mnist | 5.699 | 2.644 | 5.014 | 3.781 | 1.945 | 1.288 | 4.068 | 5.753 |
| dtd | 4.384 | 5.384 | 4.589 | 1.164 | 5.301 | 5.863 | 4.151 | 3.795 |

**Legend.**
- Score ranks do not depend on alpha (asserted equal in both blocks of the JSON).

### B2g. Gradient preference vs every score (A: budget A weights-only scores, B: budget B data-supported scores), alpha = 1 and 2.4

**What varies / split / metric / units.** Varied: score (14), alpha (1, 2.4); split: dev; metric: within-map Spearman (mean over 73 maps) and within-expert (layer-specific) rank correlation, both dimensionless, with two-sided permutation p-values in brackets (4000 and 2000 shuffles, seed 0).

**Source.** `results/aa/score_vs_targets.json`; `results/aa/score_vs_targets.md`

| score | alpha=1: within-map rho (p) | alpha=1: within-expert (p) | alpha=2.4: within-map rho (p) | alpha=2.4: within-expert (p) |
| :--- | :--- | :--- | :--- | :--- |
| A:kappa_W | +0.045 (0.307) | +0.095 (0.030) | +0.045 (0.306) | +0.078 (0.082) |
| A:smin_W | +0.001 (0.989) | -0.090 (0.041) | -0.102 (0.023) | -0.128 (0.004) |
| A:fro_D | +0.117 (0.009) | +0.099 (0.024) | -0.056 (0.197) | +0.123 (0.003) |
| A:stable_rank_D | -0.427 (0.000) | +0.025 (0.559) | +0.242 (0.000) | +0.023 (0.595) |
| A:entropy_rank_D | -0.475 (0.000) | +0.026 (0.554) | +0.285 (0.000) | +0.022 (0.617) |
| A:kplus_D_0.01 | +0.232 (0.000) | +0.022 (0.610) | -0.181 (0.000) | -0.072 (0.112) |
| A:logkappa_rel | +0.045 (0.309) | +0.095 (0.026) | +0.045 (0.314) | +0.078 (0.086) |
| B:fnorm_D | +0.330 (0.000) | +0.126 (0.005) | -0.057 (0.186) | +0.119 (0.007) |
| B:rel_fnorm_D | +0.330 (0.000) | +0.126 (0.004) | -0.057 (0.194) | +0.119 (0.007) |
| B:kappa_WC | +0.035 (0.431) | +0.011 (0.833) | -0.058 (0.200) | -0.015 (0.749) |
| B:dlogkappa_WC | +0.035 (0.433) | +0.011 (0.797) | -0.058 (0.198) | -0.015 (0.704) |
| B:srank_DC | -0.308 (0.000) | -0.024 (0.589) | +0.152 (0.002) | +0.049 (0.260) |
| B:erank_DC | -0.389 (0.000) | -0.013 (0.806) | +0.235 (0.000) | +0.075 (0.093) |
| B:kplus_DC | -0.092 (0.035) | +0.048 (0.275) | +0.065 (0.139) | +0.017 (0.691) |

**Legend.**
- gradpref_a1 / gradpref_a2.4 = -(centred tied gradient of mean dev CE at the uniform merge with alpha 1 / 2.4).
- A: scores computed from weights only (kappa_W, smin_W, fro_D = ‖Δ‖_F, stable_rank_D, entropy_rank_D, kplus_D_0.01 = kappa_plus(Delta; tau=0.01), logkappa_rel = log kappa(W_i) - log kappa(W0)). B: pooled unlabeled calibration inputs (second moment C): fnorm_D = tr(Delta C Delta^T), rel_fnorm_D, kappa_WC = kappa_2(W C^1/2), dlogkappa_WC, srank_DC, erank_DC, kplus_DC (see B7).
- Attention/MLP split, between-expert correlations and all other targets are in B4a (same file, 56 rows).

### B2h. Per-map summary of gradient preference vs kappa by module type (full 146-row table is the CSV)

**What varies / split / metric / units.** Varied: alpha (1, 2.4) and module type; split: dev; metric: mean over maps of the across-expert Spearman (dimensionless); agreement = number of maps whose argmin-kappa expert is the max-preference expert (chance = 1/8 = 12.5%).

**Source.** `results/aa/grad_pref/per_map_grad_vs_kappa.csv`

| alpha | module | n maps | mean rho(pref, kappa) | mean rho(pref, ‖Δ‖_F) | mean rho(kappa, ‖Δ‖_F) | argmin-kappa == argmax-pref | median kappa(W0) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | q_proj | 12 | +0.103 | +0.244 | +0.115 | 2/12 | 40045 |
| 1.0 | k_proj | 12 | -0.020 | +0.268 | +0.006 | 1/12 | 79139 |
| 1.0 | v_proj | 12 | +0.075 | +0.143 | +0.002 | 2/12 | 3747 |
| 1.0 | out_proj | 12 | +0.046 | +0.099 | -0.038 | 0/12 | 13961 |
| 1.0 | fc1 | 12 | +0.095 | +0.067 | +0.107 | 2/12 | 18.99 |
| 1.0 | fc2 | 12 | -0.028 | -0.040 | +0.071 | 1/12 | 41.38 |
| 1.0 | patch_embedding | 1 | +0.048 | -0.857 | -0.190 | 0/1 | 531 |
| 1.0 | attention (q,k,v,out) | 48 | +0.051 | +0.188 | +0.021 | 5/48 | 13961 |
| 1.0 | mlp (fc1,fc2) | 24 | +0.034 | +0.014 | +0.089 | 3/24 | 36.45 |
| 1.0 | all 73 maps | 73 | +0.045 | +0.117 | +0.041 | 8/73 | 3379 |
| 2.4 | q_proj | 12 | +0.155 | -0.028 | +0.115 | 2/12 | 40045 |
| 2.4 | k_proj | 12 | -0.016 | -0.032 | +0.006 | 0/12 | 79139 |
| 2.4 | v_proj | 12 | +0.099 | -0.002 | +0.002 | 2/12 | 3747 |
| 2.4 | out_proj | 12 | -0.121 | +0.000 | -0.038 | 1/12 | 13961 |
| 2.4 | fc1 | 12 | -0.062 | -0.149 | +0.107 | 0/12 | 18.99 |
| 2.4 | fc2 | 12 | +0.248 | -0.105 | +0.071 | 2/12 | 41.38 |
| 2.4 | patch_embedding | 1 | -0.333 | -0.333 | -0.190 | 0/1 | 531 |
| 2.4 | attention (q,k,v,out) | 48 | +0.029 | -0.015 | +0.021 | 5/48 | 13961 |
| 2.4 | mlp (fc1,fc2) | 24 | +0.093 | -0.127 | +0.089 | 2/24 | 36.45 |
| 2.4 | all 73 maps | 73 | +0.045 | -0.056 | +0.041 | 7/73 | 3379 |

**Legend.**
- Full data: results/aa/grad_pref/per_map_grad_vs_kappa.csv (one row per alpha x map: kappa0, kappa_min, kappa_max, the three rank correlations, argmin-kappa expert, argmax-preference expert).
- The overall mean rho(pref, kappa) over the 73 maps reproduces B2b (0.0453); the module-level means are not independent samples (each is over 12 maps, 1 for the patch embedding).

### B2i. Which expert the dev loss prefers, aggregated over the 73 linear maps

**What varies / split / metric / units.** Varied: expert and alpha; split: dev; metric: sum over the 73 linear maps of the centred preference pref_tied (CE nats per unit lambda) and number of maps in which the expert has the largest preference (of 73).

**Source.** `results/aa/grad_pref/grad_pref_alpha1.csv`; `results/aa/grad_pref/grad_pref_alpha2.4.csv`

| expert | sum of pref_tied, alpha=1 | n maps argmax-pref, alpha=1 | sum of pref_tied, alpha=2.4 | n maps argmax-pref, alpha=2.4 |
| :--- | ---: | ---: | ---: | ---: |
| sun397 | -0.2874 | 1 | 0.2637 | 2 |
| cars | -0.2836 | 0 | 0.3614 | 4 |
| resisc45 | -0.04095 | 4 | 0.4031 | 19 |
| eurosat | -0.001308 | 3 | 0.02275 | 11 |
| svhn | 0.1906 | 21 | -0.7095 | 8 |
| gtsrb | 0.2874 | 24 | -0.2174 | 18 |
| mnist | 0.1931 | 10 | -0.6874 | 0 |
| dtd | -0.0578 | 10 | 0.5634 | 11 |

**Legend.**
- pref_tied = -(grad_tied - mean over experts of grad_tied) per linear map, grad_tied = d(mean dev CE)/d(lambda_i^p) including the tied bias. The mean pref rank recomputed from these CSVs equals decompose_pref.json (asserted to 1e-3).
- Full data (1592 rows per alpha: param, layer, module, is_linear_map, expert, grad, grad_centered, pref, grad_tied, grad_tied_centered, pref_tied, delta_fro, grad_own_task, numel) is in the two CSVs; the non-linear parameters (LayerNorm, class/position embeddings; is_linear_map = 0) have derivatives there too but get the uniform prior in the rule, so they are not used in any table.

### B2j. Task-resolved loss sensitivity: derivative of each task's dev CE when expert i's weight is raised in all 73 linear maps

**What varies / split / metric / units.** Varied: alpha (1, 2.4), task t (row) and expert i (column); split: dev; metric: d CE_t / d lambda_i summed over the 73 linear maps (tied biases) at the uniform merge, CE nats per unit lambda (negative = raising that expert lowers the task's CE); budget D diagnostic.

**Source.** `results/aa/grad_pref/grad_pref_pertask.npz`; `results/aa/grad_pref/grad_pref_alpha1.csv`; `results/aa/grad_pref/grad_pref_alpha2.4.csv`

| alpha | task (row) | d/dλ(sun397) | d/dλ(cars) | d/dλ(resisc45) | d/dλ(eurosat) | d/dλ(svhn) | d/dλ(gtsrb) | d/dλ(mnist) | d/dλ(dtd) | rank of own expert (1 = most negative) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | sun397 | -0.884 | 0.134 | 0.186 | 0.181 | 0.226 | 0.321 | 0.182 | 0.144 | 1 |
| 1 | cars | 0.103 | -1.12 | 0.092 | 0.193 | 0.3 | 0.355 | 0.198 | 0.0304 | 1 |
| 1 | resisc45 | 0.204 | 0.204 | -2.7 | 0.219 | 0.617 | 0.594 | 0.577 | 0.226 | 1 |
| 1 | eurosat | -0.129 | 0.233 | -0.645 | -4.29 | 1.11 | 0.754 | 0.655 | 0.207 | 1 |
| 1 | svhn | 0.317 | 0.191 | 0.323 | 0.698 | -4.73 | -1.06 | -1.3 | -0.111 | 1 |
| 1 | gtsrb | 0.282 | 0.217 | 0.517 | 0.485 | 0.0383 | -5.76 | 0.0579 | 0.0894 | 1 |
| 1 | mnist | -0.0242 | -0.00429 | 0.0258 | 0.137 | -1.6 | -0.0475 | -4.46 | -0.176 | 1 |
| 1 | dtd | 0.112 | 0.0987 | 0.208 | 0.0689 | 0.195 | 0.221 | 0.224 | -2.27 | 1 |
| 2.4 | sun397 | -1.49 | 1.32 | 2.11 | 2.12 | 3.54 | 4.22 | 2.49 | 1.34 | 1 |
| 2.4 | cars | 0.716 | -2.08 | 1.05 | 1.36 | 2.87 | 3.3 | 1.81 | 0.642 | 1 |
| 2.4 | resisc45 | 0.996 | 0.956 | -6.09 | 1.99 | 4.03 | 3.9 | 3.14 | 0.888 | 1 |
| 2.4 | eurosat | 0.603 | 0.62 | 0.0239 | -6.16 | 3.67 | 4.08 | 2.58 | 1.14 | 1 |
| 2.4 | svhn | 1.07 | 0.664 | 1.67 | 2.12 | -6.04 | -0.322 | 0.274 | 0.364 | 1 |
| 2.4 | gtsrb | 1.18 | 0.894 | 2.39 | 2.49 | 1.93 | -10.5 | 2.21 | 0.596 | 1 |
| 2.4 | mnist | 0.198 | 0.14 | 0.276 | 0.325 | -0.669 | 0.518 | -2.5 | 0.0898 | 1 |
| 2.4 | dtd | 0.757 | 0.736 | 1.48 | 1.7 | 2.48 | 2.68 | 1.64 | -3.43 | 1 |

**Legend.**
- Entry = sum_p alpha * <grad_W CE_t, Delta_i^p> (components stored in the npz, bias gradients added to their weight map). Their mean over tasks reproduces grad_tied in grad_pref_alpha*.csv (asserted to 1e-12).
- Own-task entry (diagonal) is the most negative of its row in 16 of 16 (alpha, task) rows (last column = rank of the own expert among the 8 derivatives, 1 = most negative).

## B3. Oracle weights, full fine-tuning and LoRA

### B3a. Oracle weights fitted with dev labels: headline accuracy, full fine-tuning and LoRA banks

**What varies / split / metric / units.** Varied: oracle family (one lambda per expert + alpha; per-map lambda at alpha = 1; per-map lambda + alpha), seed (0, 1; LoRA seed 0 only); fit on dev labels (budget D, oracle/diagnostic only); evaluation on the full dev split and on the untouched test split (80%); metric: absolute accuracy (%, unweighted mean of the 8 per-task accuracies, NOT normalised), CE in nats at logit scale 100.

**Source.** `results/aa/oracle_lambda/uniform_alpha_grid.json`; `results/aa/oracle_lambda_lora/uniform_alpha_grid.json`; `results/aa/oracle_lambda/global_alpha_seed0.json`; `results/aa/oracle_lambda/global_alpha_seed1.json`; `results/aa/oracle_lambda/layerwise_a1_seed0.json`; `results/aa/oracle_lambda/layerwise_a1_seed1.json`; `results/aa/oracle_lambda/layerwise_alpha_seed0.json`; `results/aa/oracle_lambda/layerwise_alpha_seed1.json`; `results/aa/oracle_lambda_lora/global_alpha_seed0.json`; `results/aa/oracle_lambda_lora/layerwise_a1_seed0.json`; `results/aa/oracle_lambda_lora/layerwise_alpha_seed0.json`; `results/freeze/test.jsonl (cfg.tag = uniform|tuned)`; `results/aa/lora/test_freeze.jsonl (label = uniform)`

| bank | fit | seed | init alpha | fitted alpha | dev acc (%) | test acc (%) | dev - test (pt) | gain vs uniform, dev (pt) | gain vs uniform, test (pt)† | dev CE | test CE | minibatch loss, step 0 -> 399 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-FT | uniform, best alpha on dev (grid) | – | – | 1.60 | 70.28 | 69.29 † | – | 0.00 | 0.00 | 1.0339 | – | – |
| full-FT | global_alpha | 0 | 1.60 | 1.3875 | 70.60 | 69.77 | +0.83 | +0.32 | +0.48 | 1.0069 | 1.0343 | 0.938 -> 0.969 |
| full-FT | global_alpha | 1 | 1.60 | 1.3345 | 70.46 | 69.80 | +0.66 | +0.18 | +0.51 | 1.0053 | 1.0316 | 1.029 -> 0.985 |
| full-FT | layerwise_a1 | 0 | 1.00 | 1.0000 | 75.04 | 73.91 | +1.13 | +4.76 | +4.61 | 0.8632 | 0.8981 | 1.001 -> 0.835 |
| full-FT | layerwise_a1 | 1 | 1.00 | 1.0000 | 75.02 | 73.82 | +1.20 | +4.74 | +4.53 | 0.8627 | 0.9002 | 1.092 -> 0.831 |
| full-FT | layerwise_alpha | 0 | 1.60 | 1.9766 | 78.65 | 77.23 | +1.41 | +8.37 | +7.94 | 0.7530 | 0.7966 | 0.938 -> 0.741 |
| full-FT | layerwise_alpha | 1 | 1.60 | 1.9955 | 78.67 | 77.27 | +1.41 | +8.39 | +7.97 | 0.7532 | 0.7997 | 1.029 -> 0.696 |
| LoRA | uniform, best alpha on dev (grid) | – | – | 0.80 | 52.26 | 51.22 † | – | 0.00 | 0.00 | 1.8010 | – | – |
| LoRA | global_alpha | 0 | 0.80 | 0.2664 | 52.66 | 52.36 | +0.30 | +0.39 | +1.14 | 1.6494 | 1.6668 | 1.722 -> 1.620 |
| LoRA | layerwise_a1 | 0 | 1.00 | 1.0000 | 62.31 | 61.70 | +0.62 | +10.05 | +10.48 | 1.4529 | 1.4726 | 1.813 -> 1.453 |
| LoRA | layerwise_alpha | 0 | 0.80 | 1.7426 | 64.80 | 64.08 | +0.72 | +12.54 | +12.86 | 1.4202 | 1.4413 | 1.722 -> 1.415 |

**Legend.**
- global_alpha: one softmax lambda per expert shared by every linear map + a learned global alpha (8 + 1 free parameters). layerwise_a1: a per-map softmax lambda (73 maps x 8 for full-FT, 48 x 8 for LoRA) with alpha fixed at 1 (the literal normalised-average form). layerwise_alpha: per-map lambda + learned alpha. Adam, 400 steps, lr 0.1 (alpha: lr 0.02 from the script default), 128 images per task and step, logit scale 100 (steps, lr and batch asserted equal in all JSONs); init alpha = dev-best uniform alpha when alpha is learned.
- 'uniform, best alpha on dev (grid)': lambda = 1/8, alpha chosen on the dev grid of uniform_alpha_grid.json (full-FT grid 1..3.2, LoRA grid 0.6..1.6).
- † The test accuracy of tuned uniform is not stored in the oracle files; it is read from results/freeze/test.jsonl (cfg.tag = uniform|tuned) (alpha 1.6; absolute 69.29, normalised 76.68) and results/aa/lora/test_freeze.jsonl (label = uniform) (alpha 0.8; absolute 51.22); the alphas equal the dev-best uniform alphas of the grids (asserted).
- gain = fit minus uniform on the same split, in accuracy points (pt). The oracle lambdas are fitted on dev, so the dev numbers are in-sample; the test numbers are out-of-sample for the lambdas.
- minibatch loss = mean-over-tasks CE of the sampled minibatch at the first and last logged step (noisy; not the dev CE).

### B3b. Oracle weights: per-task accuracy on dev and test

**What varies / split / metric / units.** Varied: oracle family and seed as in B3a, per task; splits dev (in-sample for the fit) and test (80% held out); metric: absolute per-task accuracy (%); last column = unweighted mean of the 8.

**Source.** `results/aa/oracle_lambda/global_alpha_seed0.json`; `results/aa/oracle_lambda/global_alpha_seed1.json`; `results/aa/oracle_lambda/layerwise_a1_seed0.json`; `results/aa/oracle_lambda/layerwise_a1_seed1.json`; `results/aa/oracle_lambda/layerwise_alpha_seed0.json`; `results/aa/oracle_lambda/layerwise_alpha_seed1.json`; `results/aa/oracle_lambda_lora/global_alpha_seed0.json`; `results/aa/oracle_lambda_lora/layerwise_a1_seed0.json`; `results/aa/oracle_lambda_lora/layerwise_alpha_seed0.json`; `results/freeze/test.jsonl (cfg.tag = uniform|tuned)`; `results/aa/lora/test_freeze.jsonl (label = uniform)`

| bank | fit | seed | split | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | mean |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-FT | uniform (best alpha) | – | dev | 65.14 | 62.25 | 69.68 | 79.63 | 75.11 | 64.29 | 92.95 | 53.19 | 70.28 |
| full-FT | uniform (best alpha) † | – | test | 64.06 | 61.04 | 70.77 | 78.01 | 73.58 | 62.40 | 93.03 | 51.46 | 69.29 |
| full-FT | global_alpha | 0 | dev | 60.40 | 54.29 | 73.81 | 73.33 | 76.66 | 79.37 | 90.85 | 56.12 | 70.60 |
| full-FT | global_alpha | 0 | test | 59.23 | 53.85 | 74.94 | 72.18 | 74.93 | 78.52 | 90.61 | 53.92 | 69.77 |
| full-FT | global_alpha | 1 | dev | 61.28 | 55.16 | 72.94 | 76.48 | 76.89 | 73.04 | 91.25 | 56.65 | 70.46 |
| full-FT | global_alpha | 1 | test | 60.05 | 54.84 | 74.01 | 75.69 | 75.21 | 72.39 | 90.96 | 55.25 | 69.80 |
| full-FT | layerwise_a1 | 0 | dev | 63.10 | 58.40 | 79.92 | 82.04 | 81.73 | 84.13 | 94.10 | 56.91 | 75.04 |
| full-FT | layerwise_a1 | 0 | test | 61.76 | 58.17 | 79.40 | 82.31 | 79.99 | 83.70 | 93.65 | 52.26 | 73.91 |
| full-FT | layerwise_a1 | 1 | dev | 63.25 | 59.08 | 77.86 | 83.15 | 82.40 | 83.02 | 94.50 | 56.91 | 75.02 |
| full-FT | layerwise_a1 | 1 | test | 62.15 | 58.06 | 78.12 | 82.87 | 80.80 | 82.08 | 94.10 | 52.39 | 73.82 |
| full-FT | layerwise_alpha | 0 | dev | 62.32 | 63.00 | 83.73 | 84.26 | 86.02 | 89.27 | 95.95 | 64.63 | 78.65 |
| full-FT | layerwise_alpha | 0 | test | 61.57 | 61.00 | 82.48 | 84.63 | 83.95 | 89.02 | 95.71 | 59.51 | 77.23 |
| full-FT | layerwise_alpha | 1 | dev | 62.22 | 62.56 | 81.83 | 87.78 | 86.07 | 88.48 | 96.10 | 64.36 | 78.67 |
| full-FT | layerwise_alpha | 1 | test | 61.48 | 60.92 | 81.07 | 87.69 | 84.12 | 88.09 | 95.88 | 58.91 | 77.27 |
| LoRA | uniform (best alpha) | – | dev | 64.18 | 61.88 | 62.54 | 49.44 | 40.76 | 40.97 | 53.65 | 44.68 | 52.26 |
| LoRA | uniform (best alpha) † | – | test | 62.43 | 60.69 | 63.41 | 47.92 | 40.45 | 38.77 | 53.47 | 42.62 | 51.22 |
| LoRA | global_alpha | 0 | dev | 64.53 | 59.70 | 61.27 | 75.00 | 31.37 | 35.59 | 49.90 | 43.88 | 52.66 |
| LoRA | global_alpha | 0 | test | 62.79 | 59.41 | 61.79 | 76.30 | 30.98 | 33.83 | 49.41 | 44.35 | 52.36 |
| LoRA | layerwise_a1 | 0 | dev | 63.65 | 63.06 | 62.70 | 76.11 | 75.84 | 41.41 | 70.00 | 45.74 | 62.31 |
| LoRA | layerwise_a1 | 0 | test | 62.54 | 61.48 | 64.09 | 77.27 | 73.79 | 39.95 | 70.84 | 43.62 | 61.70 |
| LoRA | layerwise_alpha | 0 | dev | 62.90 | 62.94 | 63.81 | 75.93 | 83.13 | 43.67 | 80.30 | 45.74 | 64.80 |
| LoRA | layerwise_alpha | 0 | test | 62.07 | 62.26 | 64.72 | 75.32 | 81.96 | 42.51 | 80.96 | 42.82 | 64.08 |

**Legend.**
- † rows: tuned-uniform test accuracy from the frozen-test files (see B3a).

### B3c. Uniform-lambda merge: dev accuracy vs the global scale alpha

**What varies / split / metric / units.** Varied: alpha of the uniform merge W0 + alpha * mean(Delta); split: dev; metric: absolute accuracy (%), mean dev CE (nats, logit scale 100).

**Source.** `results/aa/oracle_lambda/uniform_alpha_grid.json`; `results/aa/oracle_lambda_lora/uniform_alpha_grid.json`

| bank | alpha | avg acc (%) | avg CE | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | dev-best |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| full-FT | 1.00 | 67.34 | 1.1087 | 66.88 | 63.12 | 69.68 | 76.11 | 65.25 | 56.77 | 86.35 | 54.52 |  |
| full-FT | 1.60 | 70.28 | 1.0339 | 65.14 | 62.25 | 69.68 | 79.63 | 75.11 | 64.29 | 92.95 | 53.19 | best |
| full-FT | 2.00 | 70.05 | 1.0744 | 62.24 | 61.50 | 66.03 | 77.04 | 78.14 | 67.34 | 95.20 | 52.93 |  |
| full-FT | 2.40 | 68.37 | 1.1738 | 57.48 | 57.34 | 63.10 | 74.07 | 78.95 | 70.07 | 96.50 | 49.47 |  |
| full-FT | 2.80 | 65.22 | 1.3301 | 50.50 | 51.62 | 58.97 | 69.44 | 77.91 | 69.60 | 96.65 | 47.07 |  |
| full-FT | 3.20 | 61.40 | 1.5338 | 42.57 | 44.90 | 53.81 | 65.00 | 76.39 | 66.86 | 96.95 | 44.68 |  |
| LoRA | 0.60 | 51.82 | 1.7438 | 64.43 | 61.94 | 62.54 | 49.81 | 38.80 | 38.99 | 53.35 | 44.68 |  |
| LoRA | 0.80 | 52.26 | 1.8010 | 64.18 | 61.88 | 62.54 | 49.44 | 40.76 | 40.97 | 53.65 | 44.68 | best |
| LoRA | 1.00 | 52.08 | 1.8853 | 63.73 | 62.25 | 61.83 | 47.96 | 42.89 | 41.73 | 52.90 | 43.35 |  |
| LoRA | 1.30 | 49.99 | 2.0619 | 62.95 | 61.07 | 60.16 | 36.85 | 45.58 | 39.98 | 49.95 | 43.35 |  |
| LoRA | 1.60 | 45.40 | 2.2827 | 61.13 | 59.27 | 56.83 | 21.30 | 46.35 | 29.93 | 45.85 | 42.55 |  |

### B3d. Oracle seed agreement (full fine-tuning; two independent fits per family)

**What varies / split / metric / units.** Varied: seed (0 vs 1) of the Adam fit for each family; split: dev (fit); metric: Spearman correlation (dimensionless) of the fitted log lambda over all (map, expert) pairs, mean within-map Spearman, and fraction of maps with the same top expert.

**Source.** `results/aa/oracle_lambda/analysis_oracle_vs_kappa.json`; `results/aa/oracle_lambda/layerwise_alpha_seed0_lambda.csv`; `... (all *_lambda.csv of oracle_lambda)`

| family | Spearman log-lambda, all (map,expert) [recomputed from CSV] | same, from analysis JSON | mean within-map Spearman [recomputed] | same, from analysis JSON | same argmax expert: attention maps | same argmax: MLP maps | same argmax: all 73 maps | alpha seed 0 | alpha seed 1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| layerwise_alpha | 0.9843 | 0.9843 | 0.9423 | 0.9423 | 45/48 | 20/24 | 66/73 | 1.9766 | 1.9955 |
| layerwise_a1 | 0.9796 | 0.9796 | 0.9344 | 0.9344 | 40/48 | 23/24 | 64/73 | 1.0000 | 1.0000 |
| global_alpha (8 expert weights) | 0.9048 | n/a (not in JSON) | n/a | n/a | – | – | argmax gtsrb vs dtd | 1.3875 | 1.3345 |

**Legend.**
- The recomputation from the CSVs equals the values in analysis_oracle_vs_kappa.json (asserted to 1e-9). The JSON has no entry for global_alpha (see B3h legend for the file times): for global_alpha the first column is the Spearman correlation of the 8 expert weights of seed 0 vs seed 1.
- LoRA oracles were fitted with seed 0 only.

### B3e. Oracle concentration: effective number of experts n_eff = exp(entropy of lambda) per map, by map type

**What varies / split / metric / units.** Varied: oracle family, seed and map type; split: dev (fit); metric: mean over the maps of n_eff (1 = one expert owns the map, 8 = uniform), dimensionless; computed here from the *_lambda.csv files.

**Source.** `results/aa/oracle_lambda/global_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda/global_alpha_seed1_lambda.csv`; `results/aa/oracle_lambda/layerwise_a1_seed0_lambda.csv`; `results/aa/oracle_lambda/layerwise_a1_seed1_lambda.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed1_lambda.csv`; `results/aa/oracle_lambda_lora/global_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda_lora/layerwise_a1_seed0_lambda.csv`; `results/aa/oracle_lambda_lora/layerwise_alpha_seed0_lambda.csv`

| bank | fit | seed | q_proj | k_proj | v_proj | out_proj | attention (all) | fc1 | fc2 | MLP (all) | patch emb. (1 map) | all maps |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-FT | global_alpha | 0 | 6.56 | 6.56 | 6.56 | 6.56 | 6.56 | 6.56 | 6.56 | 6.56 | 6.56 | 6.56 |
| full-FT | global_alpha | 1 | 6.75 | 6.75 | 6.75 | 6.75 | 6.75 | 6.75 | 6.75 | 6.75 | 6.75 | 6.75 |
| full-FT | layerwise_a1 | 0 | 1.34 | 1.41 | 1.38 | 1.36 | 1.37 | 2.17 | 2.80 | 2.49 | 5.29 | 1.79 |
| full-FT | layerwise_a1 | 1 | 1.34 | 1.36 | 1.39 | 1.34 | 1.36 | 2.19 | 2.93 | 2.56 | 4.94 | 1.80 |
| full-FT | layerwise_alpha | 0 | 1.63 | 1.51 | 1.81 | 1.84 | 1.70 | 2.97 | 3.40 | 3.18 | 6.28 | 2.25 |
| full-FT | layerwise_alpha | 1 | 1.76 | 1.59 | 1.70 | 1.79 | 1.71 | 3.06 | 3.51 | 3.29 | 5.95 | 2.29 |
| LoRA | global_alpha | 0 | 1.65 | 1.65 | 1.65 | 1.65 | 1.65 | – | – | – | – | 1.65 |
| LoRA | layerwise_a1 | 0 | 1.13 | 1.44 | 1.58 | 1.23 | 1.34 | – | – | – | – | 1.34 |
| LoRA | layerwise_alpha | 0 | 1.37 | 1.62 | 1.90 | 1.39 | 1.57 | – | – | – | – | 1.57 |

**Legend.**
- n_eff = exp(-sum_i lambda_i log lambda_i) with lambda floored at 1e-12 per map. The 'all maps' mean equals n_eff_mean and max_lambda_mean of analysis_oracle_vs_kappa.json (asserted to 1e-9 for the 4 layerwise full-FT fits).
- full-FT: 12 maps per type (72 attention/MLP weights) + the patch embedding; LoRA: only the 48 adapted attention maps (no MLP/patch entries). global_alpha has the same lambda on every map, so its n_eff is the n_eff of the 8 expert weights.

### B3f. Oracle concentration: spread of n_eff and size of the largest weight

**What varies / split / metric / units.** Varied: oracle family, seed, attention vs MLP maps; split: dev (fit); metric: min..max of n_eff over maps, mean over maps of max_i lambda_i (dimensionless), count of attention maps with n_eff < 1.5.

**Source.** `results/aa/oracle_lambda/global_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda/global_alpha_seed1_lambda.csv`; `results/aa/oracle_lambda/layerwise_a1_seed0_lambda.csv`; `results/aa/oracle_lambda/layerwise_a1_seed1_lambda.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed1_lambda.csv`; `results/aa/oracle_lambda_lora/global_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda_lora/layerwise_a1_seed0_lambda.csv`; `results/aa/oracle_lambda_lora/layerwise_alpha_seed0_lambda.csv`

| bank | fit | seed | n_eff range, attention maps | n_eff range, MLP maps | mean max lambda, attention | mean max lambda, MLP | mean max lambda, all maps | attention maps with n_eff < 1.5 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-FT | global_alpha | 0 | 6.56 .. 6.56 | 6.56 .. 6.56 | 0.206 | 0.206 | 0.206 | 0/48 |
| full-FT | global_alpha | 1 | 6.75 .. 6.75 | 6.75 .. 6.75 | 0.199 | 0.199 | 0.199 | 0/48 |
| full-FT | layerwise_a1 | 0 | 1.02 .. 2.78 | 1.04 .. 5.10 | 0.894 | 0.638 | 0.802 | 34/48 |
| full-FT | layerwise_a1 | 1 | 1.02 .. 2.40 | 1.04 .. 5.16 | 0.891 | 0.613 | 0.793 | 33/48 |
| full-FT | layerwise_alpha | 0 | 1.02 .. 3.01 | 1.07 .. 5.28 | 0.814 | 0.522 | 0.711 | 19/48 |
| full-FT | layerwise_alpha | 1 | 1.02 .. 3.11 | 1.05 .. 6.19 | 0.810 | 0.491 | 0.698 | 21/48 |
| LoRA | global_alpha | 0 | 1.65 .. 1.65 | – | 0.841 | – | 0.841 | 0/48 |
| LoRA | layerwise_a1 | 0 | 1.01 .. 3.14 | – | 0.901 | – | 0.901 | 34/48 |
| LoRA | layerwise_alpha | 0 | 1.00 .. 3.32 | – | 0.839 | – | 0.839 | 29/48 |

### B3g. Oracle weights per expert: mean lambda over maps, and how many maps each expert 'owns' (largest lambda)

**What varies / split / metric / units.** Varied: oracle family, seed, expert; split: dev (fit); metric: mean over the maps of lambda_i (rows sum to 1) -- first table; number of maps in which expert i has the largest lambda -- second table.

**Source.** `results/aa/oracle_lambda/global_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda/global_alpha_seed1_lambda.csv`; `results/aa/oracle_lambda/layerwise_a1_seed0_lambda.csv`; `results/aa/oracle_lambda/layerwise_a1_seed1_lambda.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed1_lambda.csv`; `results/aa/oracle_lambda_lora/global_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda_lora/layerwise_a1_seed0_lambda.csv`; `results/aa/oracle_lambda_lora/layerwise_alpha_seed0_lambda.csv`

| bank | fit | seed | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-FT | global_alpha | 0 | 0.0198 | 0.0146 | 0.1883 | 0.1232 | 0.1271 | 0.2055 | 0.1296 | 0.1918 |
| full-FT | global_alpha | 1 | 0.0234 | 0.0198 | 0.1733 | 0.1342 | 0.1373 | 0.1817 | 0.1310 | 0.1993 |
| full-FT | layerwise_a1 | 0 | 0.0197 | 0.0225 | 0.1187 | 0.1405 | 0.2289 | 0.2112 | 0.1123 | 0.1462 |
| full-FT | layerwise_a1 | 1 | 0.0149 | 0.0231 | 0.1177 | 0.1489 | 0.2368 | 0.1945 | 0.1272 | 0.1369 |
| full-FT | layerwise_alpha | 0 | 0.0945 | 0.0778 | 0.1461 | 0.1076 | 0.1530 | 0.1577 | 0.0820 | 0.1812 |
| full-FT | layerwise_alpha | 1 | 0.0927 | 0.0677 | 0.1354 | 0.1156 | 0.1556 | 0.1551 | 0.0855 | 0.1925 |
| LoRA | global_alpha | 0 | 0.0005 | 0.0002 | 0.0098 | 0.8409 | 0.0003 | 0.1441 | 0.0040 | 0.0002 |
| LoRA | layerwise_a1 | 0 | 0.0212 | 0.1528 | 0.1099 | 0.2727 | 0.2405 | 0.0594 | 0.0936 | 0.0500 |
| LoRA | layerwise_alpha | 0 | 0.0220 | 0.1058 | 0.1390 | 0.1935 | 0.1847 | 0.0597 | 0.2071 | 0.0882 |

**Legend.**
- mean lambda for layerwise fits equals expert_mean_lambda of analysis_oracle_vs_kappa.json (4 decimals) where that file has the fit; global_alpha rows are the 8 fitted expert weights themselves.

### B3g2. Oracle ownership: number of attention / MLP maps where each expert has the largest lambda

**What varies / split / metric / units.** Varied: oracle family, seed, map group, expert; split: dev (fit); metric: count of maps (attention: 48, MLP: 24).

**Source.** `results/aa/oracle_lambda/global_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda/global_alpha_seed1_lambda.csv`; `results/aa/oracle_lambda/layerwise_a1_seed0_lambda.csv`; `results/aa/oracle_lambda/layerwise_a1_seed1_lambda.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed1_lambda.csv`; `results/aa/oracle_lambda_lora/global_alpha_seed0_lambda.csv`; `results/aa/oracle_lambda_lora/layerwise_a1_seed0_lambda.csv`; `results/aa/oracle_lambda_lora/layerwise_alpha_seed0_lambda.csv`

| bank | fit | seed | maps | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | n maps |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-FT | layerwise_a1 | 0 | attention | 1 | 1 | 4 | 7 | 15 | 9 | 5 | 6 | 48 |
| full-FT | layerwise_a1 | 0 | mlp | 1 | 0 | 4 | 3 | 2 | 6 | 4 | 4 | 24 |
| full-FT | layerwise_a1 | 1 | attention | 0 | 1 | 5 | 9 | 15 | 7 | 6 | 5 | 48 |
| full-FT | layerwise_a1 | 1 | mlp | 1 | 0 | 4 | 3 | 3 | 6 | 3 | 4 | 24 |
| full-FT | layerwise_alpha | 0 | attention | 5 | 2 | 9 | 5 | 9 | 8 | 2 | 8 | 48 |
| full-FT | layerwise_alpha | 0 | mlp | 1 | 3 | 4 | 2 | 2 | 4 | 1 | 7 | 24 |
| full-FT | layerwise_alpha | 1 | attention | 4 | 0 | 9 | 6 | 9 | 8 | 2 | 10 | 48 |
| full-FT | layerwise_alpha | 1 | mlp | 1 | 4 | 4 | 3 | 3 | 3 | 2 | 4 | 24 |
| LoRA | layerwise_a1 | 0 | attention | 1 | 7 | 6 | 12 | 11 | 3 | 5 | 3 | 48 |
| LoRA | layerwise_alpha | 0 | attention | 1 | 4 | 7 | 9 | 9 | 3 | 10 | 5 | 48 |

### B3h. Oracle lambda vs kappa and other scores, full fine-tuning (4 layerwise fits)

**What varies / split / metric / units.** Varied: oracle fit (family, seed) and score (5); split: dev (fit); metric: Spearman across the 8 experts inside each map, averaged over the 73 maps (attention / MLP split too), between-expert and within-expert (layer-specific) rank correlation; dimensionless; one-sided permutation p-values (5000 shuffles within maps; 2000 for the within-expert null).

**Source.** `results/aa/oracle_lambda/analysis_oracle_vs_kappa.json`

| oracle fit | score | within-map rho(lambda, score) | p (null <= obs) | p (null >= obs) | attention maps | MLP maps | between-expert | within-expert | p (null <= obs) | p (null >= obs) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| layerwise_alpha_seed0 | kappa_W | -0.0117 | 0.4038 | 0.5968 | -0.028 | +0.019 | -0.661 | +0.0096 | 0.5855 | 0.4145 |
| layerwise_alpha_seed0 | smin_W | +0.0147 | 0.6418 | 0.3590 | +0.028 | -0.010 | +0.705 | -0.0119 | 0.3985 | 0.6015 |
| layerwise_alpha_seed0 | fro_D | -0.0375 | 0.1876 | 0.8130 | +0.017 | -0.143 | -0.377 | +0.0533 | 0.8805 | 0.1195 |
| layerwise_alpha_seed0 | stable_rank_D | -0.0998 | 0.0126 | 0.9874 | -0.171 | +0.037 | -0.601 | -0.0154 | 0.3630 | 0.6370 |
| layerwise_alpha_seed0 | entropy_rank_D | -0.1008 | 0.0088 | 0.9914 | -0.181 | +0.053 | -0.538 | -0.0349 | 0.2210 | 0.7790 |
| layerwise_alpha_seed1 | kappa_W | -0.0199 | 0.3150 | 0.6852 | -0.033 | +0.005 | -0.709 | +0.0053 | 0.5285 | 0.4715 |
| layerwise_alpha_seed1 | smin_W | +0.0261 | 0.7232 | 0.2768 | +0.033 | +0.013 | +0.700 | -0.0028 | 0.4895 | 0.5105 |
| layerwise_alpha_seed1 | fro_D | -0.0359 | 0.2056 | 0.7948 | +0.000 | -0.106 | -0.406 | +0.0854 | 0.9690 | 0.0310 |
| layerwise_alpha_seed1 | stable_rank_D | -0.1194 | 0.0032 | 0.9970 | -0.191 | +0.019 | -0.662 | -0.0132 | 0.4010 | 0.5990 |
| layerwise_alpha_seed1 | entropy_rank_D | -0.1129 | 0.0048 | 0.9952 | -0.193 | +0.041 | -0.568 | -0.0228 | 0.2865 | 0.7135 |
| layerwise_a1_seed0 | kappa_W | +0.0173 | 0.6530 | 0.3478 | +0.012 | +0.028 | -0.210 | +0.0374 | 0.8235 | 0.1765 |
| layerwise_a1_seed0 | smin_W | +0.0085 | 0.5824 | 0.4186 | -0.012 | +0.048 | +0.492 | -0.0398 | 0.1965 | 0.8035 |
| layerwise_a1_seed0 | fro_D | +0.0646 | 0.9270 | 0.0738 | +0.134 | -0.069 | +0.091 | +0.0770 | 0.9585 | 0.0415 |
| layerwise_a1_seed0 | stable_rank_D | -0.3206 | 0.0000 | 1.0000 | -0.359 | -0.247 | -0.837 | -0.0266 | 0.2705 | 0.7295 |
| layerwise_a1_seed0 | entropy_rank_D | -0.3330 | 0.0000 | 1.0000 | -0.389 | -0.225 | -0.819 | -0.0066 | 0.4550 | 0.5450 |
| layerwise_a1_seed1 | kappa_W | +0.0124 | 0.6118 | 0.3882 | +0.006 | +0.024 | -0.279 | +0.0366 | 0.8130 | 0.1870 |
| layerwise_a1_seed1 | smin_W | +0.0225 | 0.7022 | 0.2980 | -0.006 | +0.078 | +0.533 | -0.0259 | 0.2790 | 0.7210 |
| layerwise_a1_seed1 | fro_D | +0.0489 | 0.8666 | 0.1338 | +0.090 | -0.030 | +0.021 | +0.1050 | 0.9895 | 0.0105 |
| layerwise_a1_seed1 | stable_rank_D | -0.3232 | 0.0000 | 1.0000 | -0.359 | -0.254 | -0.895 | -0.0123 | 0.3895 | 0.6105 |
| layerwise_a1_seed1 | entropy_rank_D | -0.3444 | 0.0000 | 1.0000 | -0.395 | -0.248 | -0.869 | -0.0218 | 0.3040 | 0.6960 |

**Legend.**
- The literal rule predicts a NEGATIVE correlation between lambda and kappa_W (higher kappa -> lower weight).
- The JSON contains only the four layerwise fits (no global_alpha entry); file modification times (local, same day): analysis_oracle_vs_kappa.json 14:10, global_alpha_seed0.json 14:09, global_alpha_seed1.json 14:14. The global fits are summarised in B3g and B3d.

### B3i. Oracle lambda vs kappa and other scores, LoRA bank (48 adapted attention maps)

**What varies / split / metric / units.** Varied: oracle fit (layerwise_a1 / layerwise_alpha, seed 0) and score (7); split: dev (fit); metric: within-map Spearman (mean over 48 maps), between-expert (8 points) and within-expert rank correlation, dimensionless, two-sided permutation p-values in brackets.

**Source.** `results/aa/lora/oracle_vs_scores.json`; `results/aa/lora/oracle_vs_scores.md`; `results/aa/oracle_lambda_lora/layerwise_a1_seed0.json`

| oracle fit | score | within-map rho (p) | between-expert | within-expert (p) |
| :--- | :--- | ---: | ---: | ---: |
| layerwise_a1_seed0 | kappa_W | +0.034 (0.535) | +0.369 | +0.015 (0.795) |
| layerwise_a1_seed0 | smin_W | -0.034 (0.540) | -0.353 | -0.015 (0.770) |
| layerwise_a1_seed0 | fro_D | +0.045 (0.412) | +0.282 | -0.104 (0.055) |
| layerwise_a1_seed0 | kplus_D_0.01 | +0.031 (0.562) | +0.292 | -0.041 (0.483) |
| layerwise_a1_seed0 | stable_rank_D | +0.105 (0.059) | +0.158 | +0.094 (0.083) |
| layerwise_a1_seed0 | entropy_rank_D | +0.066 (0.218) | +0.041 | +0.074 (0.191) |
| layerwise_a1_seed0 | kappa_A(gauge-dep) | +0.055 (0.324) | +0.260 | +0.002 (0.976) |
| layerwise_alpha_seed0 | kappa_W | +0.025 (0.646) | +0.372 | +0.003 (0.980) |
| layerwise_alpha_seed0 | smin_W | -0.024 (0.655) | -0.355 | -0.004 (0.957) |
| layerwise_alpha_seed0 | fro_D | +0.056 (0.304) | +0.304 | -0.109 (0.044) |
| layerwise_alpha_seed0 | kplus_D_0.01 | -0.023 (0.685) | +0.235 | -0.100 (0.061) |
| layerwise_alpha_seed0 | stable_rank_D | +0.169 (0.001) | +0.276 | +0.143 (0.010) |
| layerwise_alpha_seed0 | entropy_rank_D | +0.136 (0.010) | +0.145 | +0.137 (0.014) |
| layerwise_alpha_seed0 | kappa_A(gauge-dep) | +0.014 (0.788) | +0.194 | -0.036 (0.484) |

**Legend.**
- This file lives in results/aa/lora/ (outside the oracle_lambda_lora directory) but is the analysis of the oracle_lambda_lora fits; its n_eff equals the one recomputed from the CSV (asserted).
- kappa_A(gauge-dep) = condition number of the LoRA factor A, gauge dependent and therefore not a valid score; kappa_W = kappa_2(W0 + B A); kplus_D_0.01 = kappa_plus(Delta; tau = 0.01).
- p-values here are two-sided (|null| >= |obs|), unlike the one-sided p-values of B3h.

## B4. Score vs targets

### B4a. Score vs targets: every row of score_vs_targets.json (4 targets x 14 scores)

**What varies / split / metric / units.** Varied: target (gradient preference at alpha 1 and 2.4; oracle layerwise lambda with learned alpha / with alpha = 1, seed 0) and score (14); split: dev; metric: rank correlations across the 8 experts (dimensionless), two-sided permutation p-values (4000 shuffles within maps for the within-map mean; 2000 shuffles inside expert columns for the within-expert correlation, seed 0).

**Source.** `results/aa/score_vs_targets.json`; `results/aa/score_vs_targets.md`

| target | score | within-map rho | p (two-sided) | attention maps | MLP maps | between-expert (8 pts) | within-expert | p (two-sided) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gradpref_a1 | A:kappa_W | +0.0453 | 0.307 | +0.051 | +0.034 | -0.284 | +0.0954 | 0.030 |
| gradpref_a1 | A:smin_W | +0.0007 | 0.989 | -0.051 | +0.100 | +0.577 | -0.0901 | 0.041 |
| gradpref_a1 | A:fro_D | +0.1168 | 0.009 | +0.188 | -0.021 | +0.154 | +0.0991 | 0.024 |
| gradpref_a1 | A:stable_rank_D | -0.4273 | 0.000 | -0.446 | -0.390 | -0.848 | +0.0253 | 0.559 |
| gradpref_a1 | A:entropy_rank_D | -0.4746 | 0.000 | -0.500 | -0.426 | -0.850 | +0.0265 | 0.554 |
| gradpref_a1 | A:kplus_D_0.01 | +0.2322 | 0.000 | +0.205 | +0.285 | +0.776 | +0.0220 | 0.610 |
| gradpref_a1 | A:logkappa_rel | +0.0453 | 0.309 | +0.051 | +0.034 | -0.284 | +0.0954 | 0.026 |
| gradpref_a1 | B:fnorm_D | +0.3301 | 0.000 | +0.373 | +0.248 | +0.568 | +0.1257 | 0.005 |
| gradpref_a1 | B:rel_fnorm_D | +0.3301 | 0.000 | +0.373 | +0.248 | +0.568 | +0.1257 | 0.004 |
| gradpref_a1 | B:kappa_WC | +0.0352 | 0.431 | +0.057 | -0.006 | +0.242 | +0.0106 | 0.833 |
| gradpref_a1 | B:dlogkappa_WC | +0.0352 | 0.433 | +0.057 | -0.006 | +0.242 | +0.0106 | 0.797 |
| gradpref_a1 | B:srank_DC | -0.3082 | 0.000 | -0.279 | -0.364 | -0.796 | -0.0241 | 0.589 |
| gradpref_a1 | B:erank_DC | -0.3888 | 0.000 | -0.394 | -0.378 | -0.808 | -0.0130 | 0.806 |
| gradpref_a1 | B:kplus_DC | -0.0917 | 0.035 | -0.086 | -0.102 | -0.860 | +0.0482 | 0.275 |
| gradpref_a2.4 | A:kappa_W | +0.0453 | 0.306 | +0.029 | +0.076 | -0.305 | +0.0777 | 0.082 |
| gradpref_a2.4 | A:smin_W | -0.1024 | 0.023 | -0.029 | -0.243 | +0.112 | -0.1275 | 0.004 |
| gradpref_a2.4 | A:fro_D | -0.0564 | 0.197 | -0.015 | -0.135 | -0.268 | +0.1228 | 0.003 |
| gradpref_a2.4 | A:stable_rank_D | +0.2420 | 0.000 | +0.219 | +0.286 | +0.619 | +0.0226 | 0.595 |
| gradpref_a2.4 | A:entropy_rank_D | +0.2851 | 0.000 | +0.237 | +0.378 | +0.679 | +0.0223 | 0.617 |
| gradpref_a2.4 | A:kplus_D_0.01 | -0.1807 | 0.000 | -0.127 | -0.283 | -0.610 | -0.0716 | 0.112 |
| gradpref_a2.4 | A:logkappa_rel | +0.0453 | 0.314 | +0.029 | +0.076 | -0.305 | +0.0777 | 0.086 |
| gradpref_a2.4 | B:fnorm_D | -0.0574 | 0.186 | +0.041 | -0.247 | -0.352 | +0.1193 | 0.007 |
| gradpref_a2.4 | B:rel_fnorm_D | -0.0574 | 0.194 | +0.041 | -0.247 | -0.352 | +0.1193 | 0.007 |
| gradpref_a2.4 | B:kappa_WC | -0.0577 | 0.200 | +0.030 | -0.227 | -0.554 | -0.0147 | 0.749 |
| gradpref_a2.4 | B:dlogkappa_WC | -0.0577 | 0.198 | +0.030 | -0.227 | -0.554 | -0.0147 | 0.704 |
| gradpref_a2.4 | B:srank_DC | +0.1520 | 0.002 | +0.068 | +0.313 | +0.438 | +0.0492 | 0.260 |
| gradpref_a2.4 | B:erank_DC | +0.2352 | 0.000 | +0.168 | +0.364 | +0.565 | +0.0754 | 0.093 |
| gradpref_a2.4 | B:kplus_DC | +0.0652 | 0.139 | +0.004 | +0.183 | +0.464 | +0.0167 | 0.691 |
| oracle_layerwise_alpha | A:kappa_W | -0.0117 | 0.794 | -0.028 | +0.019 | -0.661 | +0.0096 | 0.810 |
| oracle_layerwise_alpha | A:smin_W | +0.0147 | 0.753 | +0.028 | -0.010 | +0.705 | -0.0119 | 0.759 |
| oracle_layerwise_alpha | A:fro_D | -0.0375 | 0.407 | +0.017 | -0.143 | -0.377 | +0.0533 | 0.203 |
| oracle_layerwise_alpha | A:stable_rank_D | -0.0998 | 0.021 | -0.171 | +0.037 | -0.601 | -0.0154 | 0.711 |
| oracle_layerwise_alpha | A:entropy_rank_D | -0.1008 | 0.022 | -0.181 | +0.053 | -0.538 | -0.0349 | 0.441 |
| oracle_layerwise_alpha | A:kplus_D_0.01 | +0.0616 | 0.167 | +0.111 | -0.032 | +0.515 | +0.0216 | 0.643 |
| oracle_layerwise_alpha | A:logkappa_rel | -0.0117 | 0.780 | -0.028 | +0.019 | -0.661 | +0.0096 | 0.789 |
| oracle_layerwise_alpha | B:fnorm_D | +0.0678 | 0.129 | +0.183 | -0.152 | +0.159 | +0.0774 | 0.084 |
| oracle_layerwise_alpha | B:rel_fnorm_D | +0.0678 | 0.134 | +0.183 | -0.152 | +0.159 | +0.0774 | 0.077 |
| oracle_layerwise_alpha | B:kappa_WC | -0.0727 | 0.093 | -0.015 | -0.184 | -0.325 | -0.0641 | 0.140 |
| oracle_layerwise_alpha | B:dlogkappa_WC | -0.0727 | 0.102 | -0.015 | -0.184 | -0.325 | -0.0641 | 0.154 |
| oracle_layerwise_alpha | B:srank_DC | -0.0789 | 0.080 | -0.149 | +0.056 | -0.692 | -0.0038 | 0.921 |
| oracle_layerwise_alpha | B:erank_DC | -0.0740 | 0.094 | -0.151 | +0.074 | -0.604 | +0.0190 | 0.627 |
| oracle_layerwise_alpha | B:kplus_DC | -0.0528 | 0.229 | -0.132 | +0.099 | -0.723 | -0.0214 | 0.638 |
| oracle_layerwise_a1 | A:kappa_W | +0.0173 | 0.689 | +0.012 | +0.028 | -0.210 | +0.0374 | 0.417 |
| oracle_layerwise_a1 | A:smin_W | +0.0085 | 0.844 | -0.012 | +0.048 | +0.492 | -0.0398 | 0.371 |
| oracle_layerwise_a1 | A:fro_D | +0.0646 | 0.144 | +0.134 | -0.069 | +0.091 | +0.0770 | 0.085 |
| oracle_layerwise_a1 | A:stable_rank_D | -0.3206 | 0.000 | -0.359 | -0.247 | -0.837 | -0.0266 | 0.543 |
| oracle_layerwise_a1 | A:entropy_rank_D | -0.3330 | 0.000 | -0.389 | -0.225 | -0.819 | -0.0066 | 0.823 |
| oracle_layerwise_a1 | A:kplus_D_0.01 | +0.2303 | 0.000 | +0.254 | +0.184 | +0.826 | +0.0819 | 0.072 |
| oracle_layerwise_a1 | A:logkappa_rel | +0.0173 | 0.695 | +0.012 | +0.028 | -0.210 | +0.0374 | 0.375 |
| oracle_layerwise_a1 | B:fnorm_D | +0.2779 | 0.000 | +0.340 | +0.158 | +0.557 | +0.1642 | 0.000 |
| oracle_layerwise_a1 | B:rel_fnorm_D | +0.2779 | 0.000 | +0.340 | +0.158 | +0.557 | +0.1642 | 0.000 |
| oracle_layerwise_a1 | B:kappa_WC | +0.0042 | 0.928 | +0.040 | -0.064 | +0.298 | -0.0218 | 0.647 |
| oracle_layerwise_a1 | B:dlogkappa_WC | +0.0042 | 0.923 | +0.040 | -0.064 | +0.298 | -0.0218 | 0.637 |
| oracle_layerwise_a1 | B:srank_DC | -0.2417 | 0.000 | -0.252 | -0.222 | -0.782 | -0.0506 | 0.248 |
| oracle_layerwise_a1 | B:erank_DC | -0.2798 | 0.000 | -0.310 | -0.223 | -0.800 | -0.0162 | 0.723 |
| oracle_layerwise_a1 | B:kplus_DC | -0.1142 | 0.012 | -0.151 | -0.044 | -0.776 | -0.0351 | 0.395 |

**Legend.**
- Targets (budget D, diagnostics): gradpref_a1 / gradpref_a2.4 = -(centred tied dev-CE gradient at the uniform merge, alpha 1 / 2.4); oracle_layerwise_alpha / oracle_layerwise_a1 = fitted per-map lambda, seed 0 (B3).
- Scores A:* use weights only (budget A); B:* use the pooled unlabeled calibration second moment C (budget B): fnorm_D = tr(Delta C Delta^T); rel_fnorm_D = fnorm_D / tr(W0 C W0^T); kappa_WC = kappa_2(W C^1/2); dlogkappa_WC = log kappa_WC(W_i) - log kappa_WC(W0); srank_DC, erank_DC = stable / entropy rank of Delta C^1/2; kplus_DC = kappa_plus(Delta C^1/2; 0.01). A:logkappa_rel = log kappa(W_i) - log kappa(W0).
- within-map rho = mean over the 73 maps of the across-expert Spearman; between-expert = correlation over experts of the mean within-map ranks; within-expert = correlation of ranks after removing each expert's mean rank (layer-specific part).
- The .md file reproduces this JSON exactly in the script's own format (row-by-row string comparison: True). Rows with a score that is a monotone function of another within a map (kappa_W / logkappa_rel; fnorm_D / rel_fnorm_D; kappa_WC / dlogkappa_WC) differ only through the permutation draws.

### B4b. Re-derivation of report section 2b: master's tail-robust ratio kappa_q90 (and kappa_q50) against the same targets

**What varies / split / metric / units.** Varied: score (log kappa_q90, log kappa_q50 of the FULL expert weight; positive rho = a higher ratio coincides with a higher preference, i.e. supports the reversed rule logkappa_q90_rev) and target (6); split: dev; metric: within-map Spearman (mean over 73 maps), attention / MLP split, within-expert correlation, dimensionless, two-sided permutation p-values from 20000 / 5000 new shuffles (seed 0).

**Source.** `results/spectra/stats.csv (master's per-map spectra, kind = full; OUTSIDE the scope list, used only for this check)`; `results/aa/grad_pref/grad_pref_alpha1.csv`; `results/aa/grad_pref/grad_pref_alpha2.4.csv`; `results/aa/oracle_lambda/layerwise_alpha_seed0.json`; `... seed1 and layerwise_a1 seeds 0, 1`

| score | target | within-map rho (p) | attention maps | MLP maps | within-expert (p) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| log kappa_q90 | gradpref_a1 | +0.076 (0.090) | +0.082 | +0.065 | +0.070 (0.125) |
| log kappa_q90 | gradpref_a2.4 | +0.076 (0.087) | +0.178 | -0.118 | +0.059 (0.194) |
| log kappa_q90 | oracle_layerwise_alpha_seed0 | +0.092 (0.039) | +0.102 | +0.073 | +0.077 (0.084) |
| log kappa_q90 | oracle_layerwise_alpha_seed1 | +0.089 (0.046) | +0.094 | +0.079 | +0.074 (0.093) |
| log kappa_q90 | oracle_layerwise_a1_seed0 | +0.031 (0.488) | +0.027 | +0.038 | +0.025 (0.559) |
| log kappa_q90 | oracle_layerwise_a1_seed1 | +0.055 (0.211) | +0.047 | +0.070 | +0.047 (0.291) |
| log kappa_q50 | gradpref_a1 | +0.016 (0.729) | -0.002 | +0.050 | -0.021 (0.636) |
| log kappa_q50 | gradpref_a2.4 | -0.088 (0.044) | -0.033 | -0.194 | -0.076 (0.083) |
| log kappa_q50 | oracle_layerwise_alpha_seed0 | +0.010 (0.820) | +0.030 | -0.029 | +0.005 (0.915) |
| log kappa_q50 | oracle_layerwise_alpha_seed1 | +0.005 (0.912) | +0.016 | -0.016 | -0.003 (0.942) |
| log kappa_q50 | oracle_layerwise_a1_seed0 | +0.036 (0.423) | +0.023 | +0.060 | +0.011 (0.795) |
| log kappa_q50 | oracle_layerwise_a1_seed1 | +0.015 (0.732) | -0.018 | +0.078 | -0.012 (0.788) |

**Legend.**
- Not stored in any result file of the aa track: the report's section 2b table has no producing script in scripts/aa_*.py, so it is re-derived here. The p-values come from this script's own permutation draws and can differ from the report's in the 2nd-3rd decimal.
- Compare with report 2b: gradient pref alpha 1 +0.076 (0.09) / within-expert +0.070 (0.11); alpha 2.4 +0.076 (0.10), attention +0.18, MLP -0.12, within-expert +0.059 (0.19); oracle layerwise + alpha (2 seeds) +0.09 (0.046), within-expert +0.075; oracle alpha = 1 +0.03 (0.48) -- compared in section X (cross-check).

## B5. Shared-target soups

### B5a. Shared-target soups: the pools (setup)

**What varies / split / metric / units.** Varied: pool (A: noisy copies of one FusionBench expert; A-nm: same with norm matching; B: real same-task LoRA variants), where each soup merges the copies/variants of ONE task and is evaluated only on that task's dev split (split: dev; metric: absolute own-task accuracy, %; budget A for the rules, D for the oracles).

**Source.** `results/aa/soup/partA_controlled_noise.jsonl`; `results/aa/soup/partA_controlled_noise_strong.jsonl`; `results/aa/soup/partA_controlled_noise_strong_normmatched.jsonl`; `results/aa/soup/partA_controlled_noise_strong_v2.jsonl`; `results/aa/soup/partB_lora_variants.jsonl`; `results/aa/soup/partB_lora_variants_v2.jsonl`; `results/aa/soup/partA_meta.json`; `results/aa/soup/partA_meta_strong.json`; `results/aa/soup/partA_meta_strong_normmatched.json`; `results/aa/soup/partA_meta_strong_v2.json`; `results/aa/soup/partB_meta_v2.json`; `scripts/aa_soup_shared_target.py`

| file (.jsonl) | report name | what the copies are | noise std per copy (x per-matrix rms of the clean update) | copies / variants per task K | tasks | rows | rows per task | distinct rules |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | :--- | ---: |
| partA_controlled_noise | A, first run (mild noise) | noisy copies of each FusionBench expert | [0.25, 0.5, 1, 2] (script default; not stored in the meta file; relative levels [1, 2, 4, 8] recovered from the oracle_invvar lambda) | 4 | 8 | 736 | 8 x 92 | 12 |
| partA_controlled_noise_strong | A-strong | noisy copies of each FusionBench expert | [1, 4, 8, 16] | 4 | 8 | 736 | 8 x 92 | 12 |
| partA_controlled_noise_strong_normmatched | A-normmatched (A-nm) | noisy copies, each update rescaled to the clean update's Frobenius norm per map | [1, 4, 8, 16] | 4 | 8 | 928 | 8 x 116 | 16 |
| partA_controlled_noise_strong_v2 | A-strong-v2 | noisy copies of each FusionBench expert (same realisations as A-strong) | [1, 4, 8, 16] | 4 | 8 | 928 | 8 x 116 | 16 |
| partB_lora_variants | B, first run (interrupted) | real same-task LoRA variants | n/a | 13 | 2 | 112 | sun397:93, cars:19 | 13 |
| partB_lora_variants_v2 | B-v2 (real LoRA variants) | real same-task LoRA variants | n/a | 13 | 8 | 936 | 8 x 117 | 17 |

**Legend.**
- Part A: for each of the 8 experts, K=4 copies W_e + r_j * rms(Delta_e^p) * G_j on every linear map p (G_j iid Gaussian, seeded by the expert index), non-linear parameters from the clean expert; alpha = 1; oracle_invvar = inverse-variance weights lambda_j ~ 1/r_j^2; cleanest_only / noisiest_only = one-hot on the first / last copy.
- Part B: the 13 LoRA variants per task (seeds, logit scales, label smoothing, step budgets; Alexandre's banks) merged with alpha = 1; the oracles use each variant's own dev accuracy: oracle_best_single = one-hot on the best variant, oracle_quality_softmax{10,30,100} = softmax(q * accuracy).
- A-strong-v2 reproduces A-strong exactly on the 736 common (task, rule, b) accuracies (max abs difference 0.0e+00) and adds the rules dmedsv, dnoiseshare, mp_invvar, global:mp_invvar; B-v2 reproduces the interrupted first B run exactly on the 112 common accuracies (max abs difference 0.0e+00); the first B run covers only sun397 (93 rows) and cars (partial).
- Norm matching: A-nm rescales every noisy update to the clean update's Frobenius norm per map (script flag --norm-matched), so the dnorm scores of the copies of a map should agree up to round-off. Consistent with this, the mean lambda of the dnorm rule in A-nm stays within 0.124 of uniform over all tasks and temperatures (A-strong: up to 0.750), but it is not exactly uniform; the accuracy of dnorm in A-nm (B5d3) is therefore best read as a pseudo-random choice driven by round-off (inference, not measured separately).
- The summary.json / summary.md of the folder are reproduced by this script's re-implementation of scripts/aa_summarize_soups.py: 738 values compared across the six files, max abs difference 0.0e+00.

### B5b. Part A: how kappa and the stable rank of the update move with the injected noise

**What varies / split / metric / units.** Varied: pool and expert; metric: mean over the 73 linear maps of the rank correlation (dimensionless) between the copy's noise level and its kappa_2(W) or the stable rank of its update; '% of maps with kappa decreasing' = share of maps whose kappa correlation with noise is negative; no accuracy involved.

**Source.** `results/aa/soup/partA_meta.json`; `results/aa/soup/partA_meta_strong.json`; `results/aa/soup/partA_meta_strong_normmatched.json`; `results/aa/soup/partA_meta_strong_v2.json`

| file | expert | mean rank corr(kappa, noise) | % maps where kappa decreases with noise | mean rank corr(stable rank of Delta, noise) |
| :--- | :--- | ---: | ---: | ---: |
| partA_controlled_noise | sun397 | -0.444 | 75.3 | +1.000 |
| partA_controlled_noise | cars | -0.386 | 72.6 | +1.000 |
| partA_controlled_noise | resisc45 | -0.414 | 71.2 | +1.000 |
| partA_controlled_noise | eurosat | -0.425 | 72.6 | +1.000 |
| partA_controlled_noise | svhn | -0.425 | 72.6 | +1.000 |
| partA_controlled_noise | gtsrb | -0.405 | 74.0 | +1.000 |
| partA_controlled_noise | mnist | -0.444 | 75.3 | +1.000 |
| partA_controlled_noise | dtd | -0.408 | 74.0 | +1.000 |
| partA_controlled_noise | **mean over experts** | -0.419 | 73.5 | +1.000 |
| partA_controlled_noise_strong | sun397 | -0.441 | 71.2 | +0.937 |
| partA_controlled_noise_strong | cars | -0.408 | 65.8 | +0.973 |
| partA_controlled_noise_strong | resisc45 | -0.526 | 75.3 | +0.964 |
| partA_controlled_noise_strong | eurosat | -0.430 | 72.6 | +0.995 |
| partA_controlled_noise_strong | svhn | -0.526 | 79.5 | +0.992 |
| partA_controlled_noise_strong | gtsrb | -0.389 | 61.6 | +0.981 |
| partA_controlled_noise_strong | mnist | -0.452 | 68.5 | +0.995 |
| partA_controlled_noise_strong | dtd | -0.416 | 72.6 | +0.962 |
| partA_controlled_noise_strong | **mean over experts** | -0.449 | 70.9 | +0.975 |
| partA_controlled_noise_strong_normmatched | sun397 | -0.244 | 69.9 | +0.937 |
| partA_controlled_noise_strong_normmatched | cars | -0.219 | 63.0 | +0.973 |
| partA_controlled_noise_strong_normmatched | resisc45 | -0.274 | 67.1 | +0.964 |
| partA_controlled_noise_strong_normmatched | eurosat | -0.247 | 67.1 | +0.995 |
| partA_controlled_noise_strong_normmatched | svhn | -0.195 | 67.1 | +0.992 |
| partA_controlled_noise_strong_normmatched | gtsrb | -0.137 | 57.5 | +0.981 |
| partA_controlled_noise_strong_normmatched | mnist | -0.208 | 61.6 | +0.995 |
| partA_controlled_noise_strong_normmatched | dtd | -0.101 | 54.8 | +0.962 |
| partA_controlled_noise_strong_normmatched | **mean over experts** | -0.203 | 63.5 | +0.975 |
| partA_controlled_noise_strong_v2 | sun397 | -0.441 | 71.2 | +0.937 |
| partA_controlled_noise_strong_v2 | cars | -0.408 | 65.8 | +0.973 |
| partA_controlled_noise_strong_v2 | resisc45 | -0.526 | 75.3 | +0.964 |
| partA_controlled_noise_strong_v2 | eurosat | -0.430 | 72.6 | +0.995 |
| partA_controlled_noise_strong_v2 | svhn | -0.526 | 79.5 | +0.992 |
| partA_controlled_noise_strong_v2 | gtsrb | -0.389 | 61.6 | +0.981 |
| partA_controlled_noise_strong_v2 | mnist | -0.452 | 68.5 | +0.995 |
| partA_controlled_noise_strong_v2 | dtd | -0.416 | 72.6 | +0.962 |
| partA_controlled_noise_strong_v2 | **mean over experts** | -0.449 | 70.9 | +0.975 |

**Legend.**
- Negative correlation with noise = more noise gives a better-conditioned (lower kappa) weight, the smoothed-analysis effect (math note 3(b)): the literal rule exp(-beta*kappa) then favours the noisiest copy.
- partA_meta_strong_v2.json is identical to partA_meta_strong.json (same noise realisations); the norm-matched pool changes kappa's correlation with noise but not the stable rank's.

### B5c. Part B: the real LoRA variants -- own-task quality and its relation to kappa / update spread

**What varies / split / metric / units.** Varied: task; split: dev; metric: standalone dev accuracy of each of the 13 variants (%, min / mean / max) and Spearman correlation across the 13 variants between quality and the variant's mean log kappa_2 / mean stable rank of its update.

**Source.** `results/aa/soup/partB_meta_v2.json`

| task | variants | quality min (%) | quality mean (%) | quality max (%) | Spearman(quality, mean log kappa) | Spearman(quality, mean stable rank) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| sun397 | 13 | 69.42 | 73.03 | 74.46 | -0.308 | +0.511 |
| cars | 13 | 70.71 | 76.73 | 80.10 | +0.597 | +0.680 |
| resisc45 | 13 | 93.57 | 94.33 | 95.16 | +0.372 | +0.331 |
| eurosat | 13 | 97.41 | 98.39 | 99.07 | +0.316 | +0.430 |
| svhn | 13 | 94.99 | 96.62 | 97.48 | +0.160 | +0.646 |
| gtsrb | 13 | 97.31 | 98.32 | 98.93 | +0.416 | +0.463 |
| mnist | 13 | 99.25 | 99.50 | 99.75 | -0.267 | +0.081 |
| dtd | 13 | 73.94 | 77.88 | 82.18 | +0.385 | +0.758 |
| **mean over tasks** | 13 |  |  |  | +0.209 | +0.487 |

**Legend.**
- Variants (same 13 for every task): lora_c5a, lora_c5a_seed1, lora_c5b, lora_c5b_seed1, lora_c5_ls030, lora_c5_ls030_seed1, lora_c5_s300, lora_c5_s300_seed1, lora_c5_s1000, lora_c5_s3000, lora_c5_cf050, lora_c5_ls010, lora.
- Each variant's dense update is gamma * B A with gamma = 1 on every map that has A_/B_ factors in the variant's npz (script comment: all banks r = 16, alpha = 16, per the banks' README); the variants differ by seed, logit scale, label smoothing and step budget (names).

### B5d1. Rule x temperature grid, pool A, first run (mild noise)

**What varies / split / metric / units.** Varied: rule (rows) and temperature multiplier b (columns; beta = b / median-over-maps across-copy std of the rule's score; b=inf = argmin selection per map); split: dev; metric: absolute own-task accuracy (%) averaged over the 8 tasks, with the number of tasks beating uniform by more than 1e-12 in brackets.

**Source.** `results/aa/soup/partA_controlled_noise.jsonl`; `results/aa/soup/summary.json`; `results/aa/soup/summary.md`

| rule | single config: mean (n tasks > uniform) | b=0.25 | b=0.5 | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 | b=inf |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 90.54 (–) |  |  |  |  |  |  |  |  |  |  |  |
| kappa |  | 90.54 (2) | 90.55 (4) | 90.56 (4) | 90.57 (4) | 90.55 (3) | 90.58 (4) | 90.58 (2) | 90.58 (2) | 90.55 (2) | 90.56 (2) | 90.47 (1) |
| logkappa |  | 90.58 (4) | 90.58 (4) | 90.57 (3) | 90.58 (2) | 90.52 (2) | 90.49 (1) | 90.46 (1) | 90.44 (1) | 90.47 (1) | 90.44 (1) | 90.47 (1) |
| kappa_rev |  | 90.51 (2) | 90.50 (0) | 90.51 (2) | 90.49 (1) | 90.46 (3) | 90.42 (3) | 90.44 (2) | 90.43 (2) | 90.44 (2) | 90.47 (2) | 90.42 (2) |
| global:logkappa |  | 90.55 (4) | 90.54 (2) | 90.55 (2) | 90.50 (2) | 90.51 (2) | 90.54 (3) | 90.54 (3) | 90.52 (3) | 90.52 (3) | 90.52 (3) | 90.52 (3) |
| dsrank |  | 90.57 (2) | 90.58 (4) | 90.56 (4) | 90.50 (2) | 90.47 (2) | 90.46 (2) | 90.49 (2) | 90.51 (3) | 90.51 (3) | 90.51 (3) | 90.51 (3) |
| dsrank_rev |  | 90.55 (4) | 90.54 (2) | 90.50 (1) | 90.48 (2) | 90.49 (2) | 90.47 (2) | 90.47 (2) | 90.47 (2) | 90.47 (2) | 90.47 (2) | 90.47 (2) |
| dnorm |  | 90.53 (1) | 90.56 (4) | 90.53 (3) | 90.50 (2) | 90.50 (2) | 90.52 (3) | 90.50 (2) | 90.51 (3) | 90.51 (3) | 90.50 (3) | 90.51 (3) |
| dnorm_rev |  | 90.57 (3) | 90.49 (1) | 90.51 (1) | 90.49 (2) | 90.47 (2) | 90.46 (2) | 90.47 (2) | 90.47 (2) | 90.47 (2) | 90.47 (2) | 90.47 (2) |
| oracle_invvar | 90.49 (3) |  |  |  |  |  |  |  |  |  |  |  |
| cleanest_only | 90.51 (3) |  |  |  |  |  |  |  |  |  |  |  |
| noisiest_only | 90.47 (2) |  |  |  |  |  |  |  |  |  |  |  |

**Legend.**
- kappa = kappa_2(W_j) (the literal rule); logkappa; kappa_rev = -kappa (reversed); global:logkappa = one score per copy (mean of log kappa over maps); dsrank = stable rank of the update Delta_j (lower spread -> more weight); dsrank_rev = reversed; dnorm = ||Delta_j||_F (lower -> more weight); dnorm_rev; dmedsv = 2 log(median singular value of Delta_j) (Marchenko-Pastur bulk noise level); dnoiseshare = share of the update energy below the MP bulk edge.
- mp_invvar = lambda_j ~ (median sv of Delta_j)^-2 per map, tuning-free (raw beta = 1 on dmedsv); global:mp_invvar = one weight per copy from the total estimated noise energy. Oracles: see B5a.
- A cell is (mean over tasks) (wins). Single-configuration rules (uniform, mp_invvar, global:mp_invvar, the oracles and the one-hot references) have their value in the second column; rules with a b grid have it in the b columns.
- Rules dmedsv, dnoiseshare, mp_invvar, global:mp_invvar were not run in this file.

### B5d2. Rule x temperature grid, pool A-strong (rules dmedsv, dnoiseshare, mp_invvar, global:mp_invvar not run; values identical to the same rows of A-strong-v2)

**What varies / split / metric / units.** Varied: rule (rows) and temperature multiplier b (columns; beta = b / median-over-maps across-copy std of the rule's score; b=inf = argmin selection per map); split: dev; metric: absolute own-task accuracy (%) averaged over the 8 tasks, with the number of tasks beating uniform by more than 1e-12 in brackets.

**Source.** `results/aa/soup/partA_controlled_noise_strong.jsonl`; `results/aa/soup/summary.json`; `results/aa/soup/summary.md`

| rule | single config: mean (n tasks > uniform) | b=0.25 | b=0.5 | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 | b=inf |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 89.76 (–) |  |  |  |  |  |  |  |  |  |  |  |
| kappa |  | 89.53 (2) | 89.44 (3) | 89.42 (3) | 89.39 (3) | 89.18 (1) | 88.93 (0) | 88.77 (0) | 88.47 (0) | 87.53 (0) | 85.22 (0) | 58.79 (0) |
| logkappa |  | 89.45 (1) | 88.53 (0) | 85.66 (0) | 76.57 (0) | 66.17 (0) | 60.32 (0) | 59.03 (0) | 58.80 (0) | 58.77 (0) | 58.85 (0) | 58.79 (0) |
| kappa_rev |  | 89.76 (3) | 89.72 (2) | 89.58 (2) | 89.49 (2) | 89.39 (2) | 89.44 (2) | 89.46 (2) | 89.50 (2) | 89.61 (3) | 89.74 (4) | 89.82 (6) |
| global:logkappa |  | 88.95 (1) | 87.46 (0) | 80.45 (0) | 71.61 (0) | 62.95 (0) | 58.41 (0) | 54.06 (0) | 53.13 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) |
| dsrank |  | 90.19 (5) | 90.32 (6) | 90.44 (6) | 90.50 (8) | 90.55 (8) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) |
| dsrank_rev |  | 88.94 (1) | 87.24 (1) | 81.83 (0) | 75.30 (0) | 71.21 (0) | 68.42 (0) | 65.52 (0) | 61.97 (0) | 58.87 (0) | 56.91 (0) | 55.46 (0) |
| dnorm |  | 90.15 (6) | 90.21 (6) | 90.42 (6) | 90.55 (7) | 90.54 (8) | 90.51 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) |
| dnorm_rev |  | 88.40 (1) | 83.99 (0) | 70.64 (0) | 57.93 (0) | 53.55 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) |
| oracle_invvar | 90.51 (8) |  |  |  |  |  |  |  |  |  |  |  |
| cleanest_only | 90.52 (7) |  |  |  |  |  |  |  |  |  |  |  |
| noisiest_only | 53.05 (0) |  |  |  |  |  |  |  |  |  |  |  |

**Legend.**
- kappa = kappa_2(W_j) (the literal rule); logkappa; kappa_rev = -kappa (reversed); global:logkappa = one score per copy (mean of log kappa over maps); dsrank = stable rank of the update Delta_j (lower spread -> more weight); dsrank_rev = reversed; dnorm = ||Delta_j||_F (lower -> more weight); dnorm_rev; dmedsv = 2 log(median singular value of Delta_j) (Marchenko-Pastur bulk noise level); dnoiseshare = share of the update energy below the MP bulk edge.
- mp_invvar = lambda_j ~ (median sv of Delta_j)^-2 per map, tuning-free (raw beta = 1 on dmedsv); global:mp_invvar = one weight per copy from the total estimated noise energy. Oracles: see B5a.
- A cell is (mean over tasks) (wins). Single-configuration rules (uniform, mp_invvar, global:mp_invvar, the oracles and the one-hot references) have their value in the second column; rules with a b grid have it in the b columns.
- Rules dmedsv, dnoiseshare, mp_invvar, global:mp_invvar were not run in this file.

### B5d3. Rule x temperature grid, pool A-normmatched (A-nm)

**What varies / split / metric / units.** Varied: rule (rows) and temperature multiplier b (columns; beta = b / median-over-maps across-copy std of the rule's score; b=inf = argmin selection per map); split: dev; metric: absolute own-task accuracy (%) averaged over the 8 tasks, with the number of tasks beating uniform by more than 1e-12 in brackets.

**Source.** `results/aa/soup/partA_controlled_noise_strong_normmatched.jsonl`; `results/aa/soup/summary.json`; `results/aa/soup/summary.md`

| rule | single config: mean (n tasks > uniform) | b=0.25 | b=0.5 | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 | b=inf |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 81.27 (–) |  |  |  |  |  |  |  |  |  |  |  |
| kappa |  | 81.26 (4) | 81.18 (2) | 81.15 (3) | 81.24 (4) | 81.18 (3) | 81.18 (3) | 81.21 (4) | 81.19 (4) | 81.23 (3) | 81.18 (3) | 73.96 (0) |
| logkappa |  | 81.34 (4) | 81.25 (3) | 81.17 (4) | 81.18 (3) | 80.92 (1) | 80.76 (2) | 80.14 (0) | 79.24 (0) | 78.06 (0) | 76.64 (0) | 73.96 (0) |
| kappa_rev |  | 81.04 (2) | 81.19 (4) | 81.18 (4) | 81.19 (2) | 81.12 (2) | 81.12 (2) | 81.13 (2) | 81.05 (3) | 80.91 (2) | 80.88 (2) | 86.99 (8) |
| global:logkappa |  | 81.96 (4) | 81.93 (4) | 81.09 (4) | 79.44 (4) | 78.79 (4) | 78.66 (4) | 78.81 (4) | 78.89 (4) | 78.92 (4) | 78.93 (4) | 78.93 (4) |
| dsrank |  | 85.28 (8) | 87.30 (8) | 89.07 (8) | 90.12 (8) | 90.34 (8) | 90.33 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) |
| dsrank_rev |  | 75.73 (0) | 70.86 (0) | 64.87 (0) | 61.82 (0) | 60.64 (0) | 60.27 (0) | 60.04 (0) | 59.87 (0) | 59.73 (0) | 59.69 (0) | 59.66 (0) |
| dnorm |  | 81.07 (2) | 80.83 (3) | 80.32 (1) | 79.42 (1) | 78.89 (1) | 78.58 (1) | 78.37 (1) | 78.15 (1) | 78.09 (1) | 78.05 (1) | 78.03 (1) |
| dnorm_rev |  | 81.47 (5) | 81.59 (5) | 81.59 (5) | 81.67 (5) | 81.78 (6) | 81.85 (6) | 81.84 (5) | 81.75 (5) | 81.70 (6) | 81.64 (5) | 81.66 (5) |
| dmedsv |  | 84.72 (8) | 87.00 (8) | 89.36 (8) | 90.21 (8) | 90.40 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) |
| dnoiseshare |  | 84.70 (8) | 86.93 (8) | 89.24 (8) | 90.22 (8) | 90.40 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) | 90.39 (8) |
| mp_invvar (b=1) | 83.90 (8) |  |  |  |  |  |  |  |  |  |  |  |
| global:mp_invvar (b=1) | 83.89 (8) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_invvar | 90.07 (8) |  |  |  |  |  |  |  |  |  |  |  |
| cleanest_only | 90.39 (8) |  |  |  |  |  |  |  |  |  |  |  |
| noisiest_only | 59.02 (0) |  |  |  |  |  |  |  |  |  |  |  |

**Legend.**
- kappa = kappa_2(W_j) (the literal rule); logkappa; kappa_rev = -kappa (reversed); global:logkappa = one score per copy (mean of log kappa over maps); dsrank = stable rank of the update Delta_j (lower spread -> more weight); dsrank_rev = reversed; dnorm = ||Delta_j||_F (lower -> more weight); dnorm_rev; dmedsv = 2 log(median singular value of Delta_j) (Marchenko-Pastur bulk noise level); dnoiseshare = share of the update energy below the MP bulk edge.
- mp_invvar = lambda_j ~ (median sv of Delta_j)^-2 per map, tuning-free (raw beta = 1 on dmedsv); global:mp_invvar = one weight per copy from the total estimated noise energy. Oracles: see B5a.
- A cell is (mean over tasks) (wins). Single-configuration rules (uniform, mp_invvar, global:mp_invvar, the oracles and the one-hot references) have their value in the second column; rules with a b grid have it in the b columns.

### B5d4. Rule x temperature grid, pool A-strong-v2 (superset of A-strong, identical values on the common rows)

**What varies / split / metric / units.** Varied: rule (rows) and temperature multiplier b (columns; beta = b / median-over-maps across-copy std of the rule's score; b=inf = argmin selection per map); split: dev; metric: absolute own-task accuracy (%) averaged over the 8 tasks, with the number of tasks beating uniform by more than 1e-12 in brackets.

**Source.** `results/aa/soup/partA_controlled_noise_strong_v2.jsonl`; `results/aa/soup/summary.json`; `results/aa/soup/summary.md`

| rule | single config: mean (n tasks > uniform) | b=0.25 | b=0.5 | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 | b=inf |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 89.76 (–) |  |  |  |  |  |  |  |  |  |  |  |
| kappa |  | 89.53 (2) | 89.44 (3) | 89.42 (3) | 89.39 (3) | 89.18 (1) | 88.93 (0) | 88.77 (0) | 88.47 (0) | 87.53 (0) | 85.22 (0) | 58.79 (0) |
| logkappa |  | 89.45 (1) | 88.53 (0) | 85.66 (0) | 76.57 (0) | 66.17 (0) | 60.32 (0) | 59.03 (0) | 58.80 (0) | 58.77 (0) | 58.85 (0) | 58.79 (0) |
| kappa_rev |  | 89.76 (3) | 89.72 (2) | 89.58 (2) | 89.49 (2) | 89.39 (2) | 89.44 (2) | 89.46 (2) | 89.50 (2) | 89.61 (3) | 89.74 (4) | 89.82 (6) |
| global:logkappa |  | 88.95 (1) | 87.46 (0) | 80.45 (0) | 71.61 (0) | 62.95 (0) | 58.41 (0) | 54.06 (0) | 53.13 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) |
| dsrank |  | 90.19 (5) | 90.32 (6) | 90.44 (6) | 90.50 (8) | 90.55 (8) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) |
| dsrank_rev |  | 88.94 (1) | 87.24 (1) | 81.83 (0) | 75.30 (0) | 71.21 (0) | 68.42 (0) | 65.52 (0) | 61.97 (0) | 58.87 (0) | 56.91 (0) | 55.46 (0) |
| dnorm |  | 90.15 (6) | 90.21 (6) | 90.42 (6) | 90.55 (7) | 90.54 (8) | 90.51 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) |
| dnorm_rev |  | 88.40 (1) | 83.99 (0) | 70.64 (0) | 57.93 (0) | 53.55 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) | 53.05 (0) |
| dmedsv |  | 90.02 (5) | 90.27 (5) | 90.43 (6) | 90.53 (8) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) |
| dnoiseshare |  | 90.02 (6) | 90.30 (5) | 90.39 (6) | 90.52 (8) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) | 90.52 (7) |
| mp_invvar (b=1) | 90.53 (8) |  |  |  |  |  |  |  |  |  |  |  |
| global:mp_invvar (b=1) | 90.57 (8) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_invvar | 90.51 (8) |  |  |  |  |  |  |  |  |  |  |  |
| cleanest_only | 90.52 (7) |  |  |  |  |  |  |  |  |  |  |  |
| noisiest_only | 53.05 (0) |  |  |  |  |  |  |  |  |  |  |  |

**Legend.**
- kappa = kappa_2(W_j) (the literal rule); logkappa; kappa_rev = -kappa (reversed); global:logkappa = one score per copy (mean of log kappa over maps); dsrank = stable rank of the update Delta_j (lower spread -> more weight); dsrank_rev = reversed; dnorm = ||Delta_j||_F (lower -> more weight); dnorm_rev; dmedsv = 2 log(median singular value of Delta_j) (Marchenko-Pastur bulk noise level); dnoiseshare = share of the update energy below the MP bulk edge.
- mp_invvar = lambda_j ~ (median sv of Delta_j)^-2 per map, tuning-free (raw beta = 1 on dmedsv); global:mp_invvar = one weight per copy from the total estimated noise energy. Oracles: see B5a.
- A cell is (mean over tasks) (wins). Single-configuration rules (uniform, mp_invvar, global:mp_invvar, the oracles and the one-hot references) have their value in the second column; rules with a b grid have it in the b columns.

### B5d5. Rule x temperature grid, pool B, first run (interrupted) (interrupted run: only sun397 is complete, the cars rows are partial)

**What varies / split / metric / units.** Varied: rule (rows) and temperature multiplier b (columns; beta = b / median-over-maps across-copy std of the rule's score; b=inf = argmin selection per map); split: dev; metric: absolute own-task accuracy (%) on sun397 only, with the number of tasks beating uniform by more than 1e-12 in brackets.

**Source.** `results/aa/soup/partB_lora_variants.jsonl`; `results/aa/soup/summary.json`; `results/aa/soup/summary.md`

| rule | single config: mean (n tasks > uniform) | b=0.25 | b=0.5 | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 | b=inf |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 76.55 (–) |  |  |  |  |  |  |  |  |  |  |  |
| kappa |  | 76.42 (0) | 76.65 (2) | 76.62 (1) | 76.95 (1) | 76.93 (1) | 76.50 (0) | 76.22 (0) | 75.09 (0) | 74.99 (0) | 74.18 (0) | 72.34 (0) |
| logkappa |  | 76.78 (1) | 76.70 (2) | 76.75 (1) | 76.55 (0) | 75.59 (0) | 75.01 (0) | 74.13 (0) | 73.63 (0) | 73.27 (0) | 72.67 (0) | 72.34 (0) |
| kappa_rev |  | 75.54 (0) | 75.26 (0) | 74.66 (0) | 73.60 (0) | 72.27 (0) | 71.79 (0) | 71.31 (0) | 70.91 (0) | 70.71 (0) | 70.71 (0) | 70.68 (0) |
| global:logkappa |  | 76.62 (1) | 76.42 (0) | 76.27 (0) | 76.20 (0) | 75.72 (0) | 74.76 (0) | 74.46 (0) | 74.43 (0) | 74.46 (0) | 74.46 (0) | 74.46 (0) |
| dsrank |  | 76.17 (0) | 75.74 (0) | 75.26 (0) | 74.38 (0) | 72.54 (0) | 71.18 (0) | 70.63 (0) | 70.18 (0) | 70.10 (0) | 69.52 (0) | 68.31 (0) |
| dsrank_rev |  | 76.70 (1) | 76.85 (1) | 76.27 (0) | 75.99 (0) | 74.86 (0) | 74.21 (0) | 73.50 (0) | 73.38 (0) | 73.25 (0) | 73.17 (0) | 73.17 (0) |
| dnorm |  | 76.45 (0) | 76.60 (1) | 76.30 (0) | 74.43 (0) | 72.75 (0) | 72.39 (0) | 72.37 (0) | 72.39 (0) | 72.39 (0) | 72.39 (0) | 72.39 (0) |
| dnorm_rev |  | 76.27 (0) | 76.20 (0) | 75.67 (0) | 74.79 (0) | 73.15 (0) | 72.14 (0) | 71.16 (0) | 70.83 (0) | 70.73 (0) | 70.68 (0) | 70.76 (0) |
| oracle_best_single | 74.46 (0) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_quality_softmax10 | 76.85 (1) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_quality_softmax30 | 76.95 (1) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_quality_softmax100 | 77.36 (1) |  |  |  |  |  |  |  |  |  |  |  |

**Legend.**
- kappa = kappa_2(W_j) (the literal rule); logkappa; kappa_rev = -kappa (reversed); global:logkappa = one score per copy (mean of log kappa over maps); dsrank = stable rank of the update Delta_j (lower spread -> more weight); dsrank_rev = reversed; dnorm = ||Delta_j||_F (lower -> more weight); dnorm_rev; dmedsv = 2 log(median singular value of Delta_j) (Marchenko-Pastur bulk noise level); dnoiseshare = share of the update energy below the MP bulk edge.
- mp_invvar = lambda_j ~ (median sv of Delta_j)^-2 per map, tuning-free (raw beta = 1 on dmedsv); global:mp_invvar = one weight per copy from the total estimated noise energy. Oracles: see B5a.
- A cell is (mean over tasks) (wins). Single-configuration rules (uniform, mp_invvar, global:mp_invvar, the oracles and the one-hot references) have their value in the second column; rules with a b grid have it in the b columns.

### B5d6. Rule x temperature grid, pool B-v2 (real LoRA variants) (superset of the interrupted first run, identical values on the common rows)

**What varies / split / metric / units.** Varied: rule (rows) and temperature multiplier b (columns; beta = b / median-over-maps across-copy std of the rule's score; b=inf = argmin selection per map); split: dev; metric: absolute own-task accuracy (%) averaged over the 8 tasks, with the number of tasks beating uniform by more than 1e-12 in brackets.

**Source.** `results/aa/soup/partB_lora_variants_v2.jsonl`; `results/aa/soup/summary.json`; `results/aa/soup/summary.md`

| rule | single config: mean (n tasks > uniform) | b=0.25 | b=0.5 | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 | b=inf |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 91.04 (–) |  |  |  |  |  |  |  |  |  |  |  |
| kappa |  | 90.95 (1) | 91.00 (4) | 90.97 (4) | 90.91 (4) | 90.90 (2) | 90.69 (1) | 90.41 (0) | 90.04 (0) | 89.63 (0) | 89.06 (1) | 88.29 (1) |
| logkappa |  | 91.06 (2) | 91.06 (5) | 90.97 (3) | 90.73 (1) | 89.97 (0) | 89.37 (1) | 88.69 (1) | 88.59 (1) | 88.44 (1) | 88.32 (1) | 88.29 (1) |
| kappa_rev |  | 90.82 (3) | 90.43 (3) | 90.00 (2) | 89.35 (2) | 88.94 (1) | 88.49 (0) | 88.26 (0) | 88.10 (0) | 87.99 (0) | 88.03 (0) | 88.00 (0) |
| global:logkappa |  | 90.91 (1) | 90.69 (2) | 90.65 (0) | 90.27 (0) | 89.84 (0) | 89.43 (0) | 88.97 (0) | 88.70 (0) | 88.70 (0) | 88.70 (0) | 88.70 (0) |
| dsrank |  | 90.72 (1) | 90.26 (1) | 89.99 (2) | 89.40 (0) | 88.72 (0) | 88.08 (0) | 87.42 (0) | 87.18 (0) | 86.96 (0) | 86.79 (0) | 86.28 (0) |
| dsrank_rev |  | 90.99 (5) | 90.86 (5) | 90.66 (4) | 90.24 (2) | 89.69 (3) | 89.26 (0) | 89.06 (0) | 89.06 (1) | 89.02 (1) | 88.95 (1) | 88.83 (0) |
| dnorm |  | 91.00 (1) | 90.79 (3) | 90.40 (2) | 89.08 (1) | 88.24 (0) | 88.11 (0) | 88.02 (0) | 87.92 (0) | 87.86 (0) | 87.83 (0) | 87.76 (0) |
| dnorm_rev |  | 90.79 (4) | 90.68 (4) | 90.66 (5) | 90.67 (4) | 90.36 (4) | 89.98 (3) | 89.54 (1) | 89.36 (1) | 89.24 (2) | 89.13 (0) | 89.06 (1) |
| dmedsv |  | 90.81 (2) | 90.68 (3) | 89.71 (0) | 87.94 (0) | 87.60 (0) | 87.68 (0) | 87.70 (0) | 87.60 (0) | 87.52 (0) | 87.52 (0) | 87.52 (0) |
| dnoiseshare |  | 90.66 (1) | 90.30 (2) | 90.16 (0) | 89.34 (1) | 88.77 (0) | 88.28 (0) | 88.05 (0) | 88.05 (0) | 88.05 (0) | 88.02 (0) | 88.02 (0) |
| mp_invvar (b=1) | 90.10 (0) |  |  |  |  |  |  |  |  |  |  |  |
| global:mp_invvar (b=1) | 90.48 (2) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_best_single | 90.89 (6) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_quality_softmax10 | 91.06 (6) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_quality_softmax30 | 91.13 (6) |  |  |  |  |  |  |  |  |  |  |  |
| oracle_quality_softmax100 | 91.57 (8) |  |  |  |  |  |  |  |  |  |  |  |

**Legend.**
- kappa = kappa_2(W_j) (the literal rule); logkappa; kappa_rev = -kappa (reversed); global:logkappa = one score per copy (mean of log kappa over maps); dsrank = stable rank of the update Delta_j (lower spread -> more weight); dsrank_rev = reversed; dnorm = ||Delta_j||_F (lower -> more weight); dnorm_rev; dmedsv = 2 log(median singular value of Delta_j) (Marchenko-Pastur bulk noise level); dnoiseshare = share of the update energy below the MP bulk edge.
- mp_invvar = lambda_j ~ (median sv of Delta_j)^-2 per map, tuning-free (raw beta = 1 on dmedsv); global:mp_invvar = one weight per copy from the total estimated noise energy. Oracles: see B5a.
- A cell is (mean over tasks) (wins). Single-configuration rules (uniform, mp_invvar, global:mp_invvar, the oracles and the one-hot references) have their value in the second column; rules with a b grid have it in the b columns.

### B5e1. Per-task accuracy, pool A, first run (mild noise)

**What varies / split / metric / units.** Varied: rule (rows) and task (columns); split: dev; metric: absolute own-task accuracy (%), last two columns = mean over the 8 tasks and number of tasks beating uniform.

**Source.** `results/aa/soup/partA_controlled_noise.jsonl`; `results/aa/soup/summary.md`; `results/aa/soup/summary.json`

| rule | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | mean | tasks > uniform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 74.56 | 78.61 | 95.40 | 98.70 | 97.54 | 99.01 | 99.65 | 80.85 | 90.54 | – |
| kappa b=0.25 | 74.61 | 78.54 | 95.40 | 98.70 | 97.56 | 99.01 | 99.65 | 80.85 | 90.54 | 2 |
| kappa b=1 | 74.61 | 78.73 | 95.48 | 98.70 | 97.56 | 98.93 | 99.65 | 80.85 | 90.56 | 4 |
| kappa b=4 | 74.61 | 78.61 | 95.48 | 98.70 | 97.58 | 98.93 | 99.65 | 80.85 | 90.55 | 3 |
| kappa b=16 | 74.56 | 78.54 | 95.32 | 98.70 | 97.58 | 98.93 | 99.65 | 81.38 | 90.58 | 2 |
| kappa b=inf | 74.08 | 78.54 | 95.40 | 98.70 | 97.60 | 98.97 | 99.65 | 80.85 | 90.47 | 1 |
| logkappa b=4 | 74.48 | 78.36 | 95.32 | 98.70 | 97.62 | 98.93 | 99.65 | 81.12 | 90.52 | 2 |
| global:logkappa b=4 | 74.26 | 78.54 | 95.32 | 98.89 | 97.60 | 98.97 | 99.65 | 80.85 | 90.51 | 2 |
| kappa_rev b=4 | 74.63 | 78.61 | 95.16 | 98.70 | 97.56 | 98.97 | 99.70 | 80.32 | 90.46 | 3 |
| dsrank b=4 | 74.48 | 78.30 | 95.32 | 98.52 | 97.58 | 99.01 | 99.70 | 80.85 | 90.47 | 2 |
| dnorm b=4 | 74.56 | 78.48 | 95.32 | 98.52 | 97.58 | 99.01 | 99.70 | 80.85 | 90.50 | 2 |
| best kappa (per-task max over the b grid) | 74.66 | 78.73 | 95.48 | 98.70 | 97.60 | 99.01 | 99.65 | 81.38 | 90.65 | 5 |
| best logkappa (per-task max over the b grid) | 74.66 | 78.73 | 95.40 | 98.70 | 97.62 | 99.01 | 99.65 | 81.38 | 90.64 | 4 |
| best kappa_rev (per-task max over the b grid) | 74.74 | 78.73 | 95.32 | 98.70 | 97.56 | 99.01 | 99.75 | 80.59 | 90.55 | 4 |
| best global:logkappa (per-task max over the b grid) | 74.61 | 78.73 | 95.56 | 98.89 | 97.60 | 99.05 | 99.65 | 80.85 | 90.62 | 6 |
| best dsrank (per-task max over the b grid) | 74.69 | 78.79 | 95.48 | 98.70 | 97.58 | 99.01 | 99.70 | 80.85 | 90.60 | 5 |
| best dsrank_rev (per-task max over the b grid) | 74.61 | 78.73 | 95.40 | 98.89 | 97.62 | 99.05 | 99.65 | 80.85 | 90.60 | 5 |
| best dnorm (per-task max over the b grid) | 74.76 | 78.61 | 95.48 | 98.70 | 97.58 | 99.01 | 99.70 | 80.85 | 90.59 | 4 |
| best dnorm_rev (per-task max over the b grid) | 74.66 | 78.92 | 95.32 | 98.89 | 97.60 | 99.01 | 99.65 | 80.85 | 90.61 | 4 |
| oracle_invvar | 74.61 | 78.36 | 95.32 | 98.52 | 97.58 | 99.01 | 99.70 | 80.85 | 90.49 | 3 |
| cleanest_only | 74.58 | 78.48 | 95.32 | 98.52 | 97.58 | 99.01 | 99.70 | 80.85 | 90.51 | 3 |
| noisiest_only | 74.26 | 78.36 | 95.16 | 98.89 | 97.60 | 98.97 | 99.65 | 80.85 | 90.47 | 2 |

**Legend.**
- 'best <rule>' = per task maximum over the 11 temperatures (an optimistic oracle over b, as in summary.md); 'rule b=4' rows are single configurations. Rows are a superset of summary.md's columns (which are reproduced exactly, see B5a legend).

### B5e2. Per-task accuracy, pool A-strong-v2

**What varies / split / metric / units.** Varied: rule (rows) and task (columns); split: dev; metric: absolute own-task accuracy (%), last two columns = mean over the 8 tasks and number of tasks beating uniform.

**Source.** `results/aa/soup/partA_controlled_noise_strong_v2.jsonl`; `results/aa/soup/summary.md`; `results/aa/soup/summary.json`

| rule | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | mean | tasks > uniform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 71.76 | 76.12 | 94.92 | 98.33 | 97.45 | 98.73 | 99.65 | 81.12 | 89.76 | – |
| kappa b=0.25 | 71.01 | 76.24 | 94.76 | 98.33 | 97.48 | 98.73 | 99.65 | 80.05 | 89.53 | 2 |
| kappa b=1 | 69.95 | 76.87 | 94.52 | 98.15 | 97.48 | 98.77 | 99.60 | 80.05 | 89.42 | 3 |
| kappa b=4 | 69.14 | 75.81 | 94.44 | 98.33 | 97.46 | 98.69 | 99.50 | 80.05 | 89.18 | 1 |
| kappa b=16 | 67.93 | 74.69 | 93.89 | 98.33 | 97.27 | 98.54 | 99.45 | 80.05 | 88.77 | 0 |
| kappa b=inf | 2.67 | 11.38 | 53.10 | 93.33 | 91.32 | 82.78 | 68.20 | 67.55 | 58.79 | 0 |
| logkappa b=4 | 6.17 | 22.89 | 70.32 | 96.11 | 93.95 | 89.94 | 76.05 | 73.94 | 66.17 | 0 |
| global:logkappa b=4 | 1.21 | 51.43 | 57.14 | 86.30 | 88.71 | 94.26 | 60.75 | 63.83 | 62.95 | 0 |
| kappa_rev b=4 | 71.23 | 74.13 | 94.60 | 99.07 | 97.39 | 98.85 | 99.55 | 80.32 | 89.39 | 2 |
| dsrank b=4 | 74.53 | 78.48 | 95.16 | 98.52 | 97.60 | 99.01 | 99.70 | 81.38 | 90.55 | 8 |
| dnoiseshare b=4 | 74.53 | 78.48 | 95.24 | 98.52 | 97.60 | 99.01 | 99.70 | 81.12 | 90.52 | 7 |
| dmedsv b=4 | 74.53 | 78.48 | 95.24 | 98.52 | 97.60 | 99.01 | 99.70 | 81.12 | 90.52 | 7 |
| dnorm b=4 | 74.46 | 78.36 | 95.32 | 98.52 | 97.60 | 99.01 | 99.70 | 81.38 | 90.54 | 8 |
| best kappa (per-task max over the b grid) | 71.01 | 76.99 | 94.76 | 98.33 | 97.50 | 98.77 | 99.65 | 80.05 | 89.63 | 3 |
| best logkappa (per-task max over the b grid) | 70.05 | 75.50 | 94.92 | 98.33 | 97.39 | 98.77 | 99.55 | 81.12 | 89.45 | 1 |
| best kappa_rev (per-task max over the b grid) | 72.77 | 76.12 | 95.08 | 99.07 | 97.50 | 99.13 | 99.70 | 80.85 | 90.03 | 6 |
| best global:logkappa (per-task max over the b grid) | 67.93 | 74.32 | 94.60 | 98.33 | 97.25 | 98.77 | 99.55 | 80.85 | 88.95 | 1 |
| best dsrank (per-task max over the b grid) | 74.53 | 78.48 | 95.40 | 98.70 | 97.60 | 99.01 | 99.70 | 81.38 | 90.60 | 8 |
| best dsrank_rev (per-task max over the b grid) | 68.61 | 73.20 | 94.60 | 98.52 | 97.31 | 98.69 | 99.40 | 81.38 | 88.97 | 2 |
| best dnorm (per-task max over the b grid) | 74.53 | 78.61 | 95.63 | 98.52 | 97.62 | 99.01 | 99.70 | 81.38 | 90.63 | 8 |
| best dnorm_rev (per-task max over the b grid) | 65.59 | 71.77 | 94.37 | 98.33 | 97.29 | 98.73 | 99.45 | 81.65 | 88.40 | 1 |
| best dmedsv (per-task max over the b grid) | 74.53 | 78.54 | 95.48 | 98.70 | 97.60 | 99.01 | 99.70 | 81.38 | 90.62 | 8 |
| best dnoiseshare (per-task max over the b grid) | 74.53 | 78.54 | 95.40 | 98.70 | 97.60 | 99.01 | 99.70 | 81.38 | 90.61 | 8 |
| mp_invvar | 74.48 | 78.36 | 95.24 | 98.52 | 97.58 | 98.97 | 99.70 | 81.38 | 90.53 | 8 |
| global:mp_invvar | 74.51 | 78.36 | 95.24 | 98.52 | 97.58 | 98.97 | 99.70 | 81.65 | 90.57 | 8 |
| oracle_invvar | 74.51 | 78.17 | 95.24 | 98.52 | 97.58 | 98.97 | 99.70 | 81.38 | 90.51 | 8 |
| cleanest_only | 74.53 | 78.48 | 95.16 | 98.52 | 97.60 | 99.01 | 99.70 | 81.12 | 90.52 | 7 |
| noisiest_only | 1.16 | 5.91 | 50.40 | 83.52 | 88.67 | 74.43 | 57.05 | 63.30 | 53.05 | 0 |

**Legend.**
- 'best <rule>' = per task maximum over the 11 temperatures (an optimistic oracle over b, as in summary.md); 'rule b=4' rows are single configurations. Rows are a superset of summary.md's columns (which are reproduced exactly, see B5a legend).

### B5e3. Per-task accuracy, pool A-normmatched (A-nm)

**What varies / split / metric / units.** Varied: rule (rows) and task (columns); split: dev; metric: absolute own-task accuracy (%), last two columns = mean over the 8 tasks and number of tasks beating uniform.

**Source.** `results/aa/soup/partA_controlled_noise_strong_normmatched.jsonl`; `results/aa/soup/summary.md`; `results/aa/soup/summary.json`

| rule | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | mean | tasks > uniform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 71.31 | 70.21 | 85.40 | 94.44 | 85.96 | 84.28 | 96.60 | 61.97 | 81.27 | – |
| kappa b=0.25 | 71.44 | 70.77 | 85.48 | 94.26 | 86.17 | 83.65 | 96.60 | 61.70 | 81.26 | 4 |
| kappa b=1 | 71.49 | 70.96 | 85.16 | 94.63 | 85.82 | 83.61 | 96.40 | 61.17 | 81.15 | 3 |
| kappa b=4 | 71.28 | 70.58 | 84.76 | 95.37 | 86.42 | 83.65 | 96.20 | 61.17 | 81.18 | 3 |
| kappa b=16 | 71.01 | 70.52 | 84.52 | 95.37 | 86.44 | 84.44 | 96.45 | 60.90 | 81.21 | 4 |
| kappa b=inf | 68.41 | 67.29 | 76.98 | 88.52 | 73.61 | 68.96 | 91.00 | 56.91 | 73.96 | 0 |
| logkappa b=4 | 70.78 | 70.09 | 84.05 | 95.00 | 85.94 | 83.81 | 96.25 | 61.44 | 80.92 | 1 |
| global:logkappa b=4 | 69.32 | 76.12 | 74.29 | 70.19 | 94.51 | 98.93 | 66.65 | 80.32 | 78.79 | 4 |
| kappa_rev b=4 | 71.08 | 69.59 | 84.37 | 94.26 | 86.73 | 84.76 | 96.45 | 61.70 | 81.12 | 2 |
| dsrank b=4 | 76.05 | 77.99 | 94.52 | 98.15 | 97.06 | 98.97 | 99.65 | 80.32 | 90.34 | 8 |
| dnoiseshare b=4 | 76.10 | 77.92 | 94.37 | 98.15 | 97.08 | 99.01 | 99.70 | 80.85 | 90.40 | 8 |
| dmedsv b=4 | 76.10 | 77.92 | 94.37 | 98.15 | 97.08 | 99.01 | 99.70 | 80.85 | 90.40 | 8 |
| dnorm b=4 | 70.48 | 67.29 | 84.92 | 86.11 | 83.19 | 77.32 | 96.10 | 65.69 | 78.89 | 1 |
| best kappa (per-task max over the b grid) | 71.51 | 70.96 | 85.48 | 95.37 | 86.53 | 85.31 | 96.60 | 62.23 | 81.75 | 7 |
| best logkappa (per-task max over the b grid) | 71.46 | 70.90 | 85.40 | 95.19 | 86.38 | 83.81 | 96.60 | 62.23 | 81.50 | 5 |
| best kappa_rev (per-task max over the b grid) | 74.01 | 71.46 | 91.35 | 97.04 | 95.41 | 95.92 | 98.90 | 71.81 | 86.99 | 8 |
| best global:logkappa (per-task max over the b grid) | 70.86 | 77.86 | 82.22 | 91.85 | 97.08 | 99.01 | 94.90 | 80.85 | 86.83 | 4 |
| best dsrank (per-task max over the b grid) | 76.15 | 77.99 | 94.52 | 98.15 | 97.08 | 99.01 | 99.70 | 80.85 | 90.43 | 8 |
| best dsrank_rev (per-task max over the b grid) | 69.40 | 68.78 | 79.76 | 88.52 | 76.10 | 72.84 | 92.75 | 57.71 | 75.73 | 0 |
| best dnorm (per-task max over the b grid) | 71.21 | 69.84 | 85.56 | 93.70 | 86.19 | 83.73 | 96.50 | 66.22 | 81.62 | 3 |
| best dnorm_rev (per-task max over the b grid) | 71.91 | 72.01 | 84.68 | 95.74 | 88.05 | 87.73 | 96.50 | 64.36 | 82.62 | 6 |
| best dmedsv (per-task max over the b grid) | 76.12 | 77.99 | 94.37 | 98.15 | 97.08 | 99.01 | 99.70 | 80.85 | 90.41 | 8 |
| best dnoiseshare (per-task max over the b grid) | 76.12 | 77.92 | 94.37 | 98.15 | 97.08 | 99.01 | 99.70 | 80.85 | 90.40 | 8 |
| mp_invvar | 72.14 | 71.58 | 87.06 | 96.30 | 90.93 | 89.23 | 98.30 | 65.69 | 83.90 | 8 |
| global:mp_invvar | 72.02 | 71.33 | 86.90 | 96.11 | 91.13 | 89.63 | 98.35 | 65.69 | 83.89 | 8 |
| oracle_invvar | 75.99 | 77.55 | 94.13 | 98.15 | 97.00 | 98.85 | 99.65 | 79.26 | 90.07 | 8 |
| cleanest_only | 76.12 | 77.86 | 94.37 | 98.15 | 97.08 | 99.01 | 99.70 | 80.85 | 90.39 | 8 |
| noisiest_only | 66.17 | 63.18 | 66.43 | 67.96 | 47.02 | 44.97 | 66.40 | 50.00 | 59.02 | 0 |

**Legend.**
- 'best <rule>' = per task maximum over the 11 temperatures (an optimistic oracle over b, as in summary.md); 'rule b=4' rows are single configurations. Rows are a superset of summary.md's columns (which are reproduced exactly, see B5a legend).

### B5e4. Per-task accuracy, pool B-v2 (real LoRA variants)

**What varies / split / metric / units.** Varied: rule (rows) and task (columns); split: dev; metric: absolute own-task accuracy (%), last two columns = mean over the 8 tasks and number of tasks beating uniform.

**Source.** `results/aa/soup/partB_lora_variants_v2.jsonl`; `results/aa/soup/summary.md`; `results/aa/soup/summary.json`

| rule | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | mean | tasks > uniform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 76.55 | 81.97 | 94.76 | 98.33 | 97.41 | 98.85 | 99.60 | 80.85 | 91.04 | – |
| kappa b=0.25 | 76.42 | 81.84 | 94.92 | 98.33 | 97.39 | 98.77 | 99.60 | 80.32 | 90.95 | 1 |
| kappa b=1 | 76.62 | 81.97 | 94.92 | 98.33 | 97.43 | 98.81 | 99.65 | 80.05 | 90.97 | 4 |
| kappa b=4 | 76.93 | 81.41 | 95.08 | 98.33 | 97.29 | 98.81 | 99.55 | 79.79 | 90.90 | 2 |
| kappa b=16 | 76.22 | 80.91 | 94.60 | 97.96 | 97.23 | 98.61 | 99.55 | 78.19 | 90.41 | 0 |
| kappa b=inf | 72.34 | 75.19 | 93.25 | 98.70 | 96.52 | 97.90 | 99.50 | 72.87 | 88.29 | 1 |
| logkappa b=4 | 75.59 | 78.98 | 94.05 | 98.33 | 97.04 | 98.30 | 99.55 | 77.93 | 89.97 | 0 |
| global:logkappa b=4 | 75.72 | 78.23 | 94.13 | 97.78 | 96.25 | 98.10 | 99.55 | 78.99 | 89.84 | 0 |
| kappa_rev b=4 | 72.27 | 75.75 | 94.05 | 98.52 | 96.89 | 97.66 | 99.55 | 76.86 | 88.94 | 1 |
| dsrank b=4 | 72.54 | 75.93 | 93.17 | 98.15 | 95.16 | 97.39 | 99.45 | 77.93 | 88.72 | 0 |
| dnoiseshare b=4 | 72.75 | 75.62 | 94.29 | 97.96 | 96.00 | 97.59 | 99.60 | 76.33 | 88.77 | 0 |
| dmedsv b=4 | 72.12 | 70.65 | 93.41 | 98.15 | 93.32 | 97.90 | 99.20 | 76.06 | 87.60 | 0 |
| dnorm b=4 | 72.75 | 71.08 | 93.73 | 98.33 | 95.08 | 98.22 | 99.35 | 77.39 | 88.24 | 0 |
| best kappa (per-task max over the b grid) | 76.95 | 82.15 | 95.08 | 98.70 | 97.43 | 98.89 | 99.65 | 80.32 | 91.15 | 7 |
| best logkappa (per-task max over the b grid) | 76.78 | 82.03 | 94.92 | 98.70 | 97.37 | 98.93 | 99.65 | 80.85 | 91.15 | 6 |
| best kappa_rev (per-task max over the b grid) | 75.54 | 80.66 | 95.08 | 98.52 | 97.33 | 98.69 | 99.85 | 80.85 | 90.82 | 3 |
| best global:logkappa (per-task max over the b grid) | 76.62 | 81.72 | 94.84 | 98.33 | 97.27 | 98.89 | 99.60 | 80.32 | 90.95 | 3 |
| best dsrank (per-task max over the b grid) | 76.17 | 81.53 | 95.08 | 98.52 | 97.20 | 98.89 | 99.60 | 78.99 | 90.75 | 3 |
| best dsrank_rev (per-task max over the b grid) | 76.85 | 80.85 | 95.08 | 98.70 | 97.43 | 98.54 | 99.75 | 81.65 | 91.11 | 6 |
| best dnorm (per-task max over the b grid) | 76.60 | 81.97 | 95.00 | 98.52 | 97.29 | 98.93 | 99.60 | 80.59 | 91.06 | 4 |
| best dnorm_rev (per-task max over the b grid) | 76.27 | 80.47 | 95.08 | 98.70 | 97.54 | 98.93 | 99.75 | 81.65 | 91.05 | 6 |
| best dmedsv (per-task max over the b grid) | 76.27 | 81.84 | 94.92 | 98.52 | 97.18 | 98.89 | 99.60 | 79.52 | 90.84 | 3 |
| best dnoiseshare (per-task max over the b grid) | 75.79 | 81.41 | 95.16 | 98.33 | 97.27 | 98.97 | 99.60 | 79.26 | 90.72 | 2 |
| mp_invvar | 75.64 | 80.10 | 94.29 | 98.33 | 95.66 | 98.81 | 99.55 | 78.46 | 90.10 | 0 |
| global:mp_invvar | 76.22 | 80.35 | 94.44 | 98.70 | 96.64 | 98.89 | 99.60 | 78.99 | 90.48 | 2 |
| oracle_best_single | 74.46 | 80.10 | 95.16 | 99.07 | 97.48 | 98.93 | 99.75 | 82.18 | 90.89 | 6 |
| oracle_quality_softmax10 | 76.85 | 82.03 | 94.84 | 98.52 | 97.43 | 98.81 | 99.65 | 80.32 | 91.06 | 6 |
| oracle_quality_softmax30 | 76.95 | 82.21 | 94.84 | 98.52 | 97.39 | 98.89 | 99.65 | 80.59 | 91.13 | 6 |
| oracle_quality_softmax100 | 77.36 | 82.96 | 95.24 | 98.52 | 97.52 | 98.89 | 99.65 | 82.45 | 91.57 | 8 |

**Legend.**
- 'best <rule>' = per task maximum over the 11 temperatures (an optimistic oracle over b, as in summary.md); 'rule b=4' rows are single configurations. Rows are a superset of summary.md's columns (which are reproduced exactly, see B5a legend).

### B5f1. Weight each rule puts on the copies, pool A-strong-v2

**What varies / split / metric / units.** Varied: rule and temperature; split: dev (weights do not depend on labels for budget-A rules); metric: mean over maps (then over the 8 tasks) of the weight lambda_j on the copy with noise level [1, 4, 8, 16] x rms; dimensionless, rows sum to 1.

**Source.** `results/aa/soup/partA_controlled_noise_strong_v2.jsonl`

| rule | copy 1 (noise 1) | copy 2 (4) | copy 3 (8) | copy 4 (16, noisiest) | weight on noisiest copy: min .. max over the 8 tasks |
| :--- | ---: | ---: | ---: | ---: | ---: |
| uniform | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 .. 0.250 |
| kappa b=1 | 0.211 | 0.225 | 0.254 | 0.310 | 0.273 .. 0.355 |
| kappa b=4 | 0.208 | 0.224 | 0.252 | 0.316 | 0.280 .. 0.392 |
| kappa b=16 | 0.204 | 0.223 | 0.252 | 0.322 | 0.287 .. 0.408 |
| kappa b=inf | 0.128 | 0.139 | 0.161 | 0.572 | 0.534 .. 0.658 |
| logkappa b=1 | 0.170 | 0.192 | 0.249 | 0.389 | 0.361 .. 0.437 |
| logkappa b=4 | 0.134 | 0.147 | 0.188 | 0.531 | 0.494 .. 0.617 |
| logkappa b=16 | 0.129 | 0.140 | 0.163 | 0.568 | 0.532 .. 0.656 |
| logkappa b=inf | 0.128 | 0.139 | 0.161 | 0.572 | 0.534 .. 0.658 |
| global:logkappa b=1 | 0.046 | 0.102 | 0.228 | 0.624 | 0.470 .. 0.760 |
| global:logkappa b=4 | 0.000 | 0.001 | 0.080 | 0.919 | 0.653 .. 1.000 |
| global:logkappa b=16 | 0.000 | 0.000 | 0.010 | 0.990 | 0.927 .. 1.000 |
| global:logkappa b=inf | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 .. 1.000 |
| dsrank b=4 | 0.984 | 0.016 | 0.000 | 0.000 | 0.000 .. 0.000 |
| dnorm b=4 | 0.883 | 0.109 | 0.008 | 0.000 | 0.000 .. 0.000 |
| dmedsv b=4 | 0.994 | 0.005 | 0.000 | 0.000 | 0.000 .. 0.000 |
| mp_invvar | 0.907 | 0.071 | 0.018 | 0.005 | 0.004 .. 0.005 |
| global:mp_invvar | 0.906 | 0.071 | 0.018 | 0.005 | 0.004 .. 0.005 |
| oracle_invvar | 0.924 | 0.058 | 0.014 | 0.004 | 0.004 .. 0.004 |

**Legend.**
- oracle_invvar puts most weight on the cleanest copy. 'Anti-inverse-variance' (shorthand used in the report) = a rule whose weight on the noisiest copy exceeds its weight on the cleanest copy.
- mean over the 8 tasks of the per-task map-averaged weight; the last column gives the spread over tasks of the weight on the noisiest copy.

### B5f2. Weight each rule puts on the copies, pool A-normmatched (A-nm)

**What varies / split / metric / units.** Varied: rule and temperature; split: dev (weights do not depend on labels for budget-A rules); metric: mean over maps (then over the 8 tasks) of the weight lambda_j on the copy with noise level [1, 4, 8, 16] x rms; dimensionless, rows sum to 1.

**Source.** `results/aa/soup/partA_controlled_noise_strong_normmatched.jsonl`

| rule | copy 1 (noise 1) | copy 2 (4) | copy 3 (8) | copy 4 (16, noisiest) | weight on noisiest copy: min .. max over the 8 tasks |
| :--- | ---: | ---: | ---: | ---: | ---: |
| uniform | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 .. 0.250 |
| kappa b=1 | 0.245 | 0.249 | 0.254 | 0.252 | 0.208 .. 0.311 |
| kappa b=4 | 0.251 | 0.248 | 0.253 | 0.248 | 0.193 .. 0.300 |
| kappa b=16 | 0.254 | 0.251 | 0.249 | 0.247 | 0.186 .. 0.300 |
| kappa b=inf | 0.171 | 0.255 | 0.255 | 0.318 | 0.260 .. 0.411 |
| logkappa b=1 | 0.247 | 0.248 | 0.250 | 0.255 | 0.222 .. 0.297 |
| logkappa b=4 | 0.245 | 0.251 | 0.252 | 0.252 | 0.193 .. 0.302 |
| logkappa b=16 | 0.232 | 0.255 | 0.254 | 0.260 | 0.195 .. 0.312 |
| logkappa b=inf | 0.171 | 0.255 | 0.255 | 0.318 | 0.260 .. 0.411 |
| global:logkappa b=1 | 0.288 | 0.251 | 0.205 | 0.256 | 0.035 .. 0.711 |
| global:logkappa b=4 | 0.391 | 0.172 | 0.168 | 0.269 | 0.000 .. 0.997 |
| global:logkappa b=16 | 0.452 | 0.166 | 0.130 | 0.251 | 0.000 .. 1.000 |
| global:logkappa b=inf | 0.500 | 0.125 | 0.125 | 0.250 | 0.000 .. 1.000 |
| dsrank b=4 | 0.984 | 0.016 | 0.000 | 0.000 | 0.000 .. 0.000 |
| dnorm b=4 | 0.243 | 0.268 | 0.229 | 0.259 | 0.200 .. 0.344 |
| dmedsv b=4 | 0.999 | 0.000 | 0.000 | 0.000 | 0.000 .. 0.000 |
| mp_invvar | 0.339 | 0.223 | 0.219 | 0.219 | 0.211 .. 0.229 |
| global:mp_invvar | 0.338 | 0.224 | 0.220 | 0.219 | 0.211 .. 0.230 |
| oracle_invvar | 0.924 | 0.058 | 0.014 | 0.004 | 0.004 .. 0.004 |

**Legend.**
- oracle_invvar puts most weight on the cleanest copy. 'Anti-inverse-variance' (shorthand used in the report) = a rule whose weight on the noisiest copy exceeds its weight on the cleanest copy.
- mean over the 8 tasks of the per-task map-averaged weight; the last column gives the spread over tasks of the weight on the noisiest copy.

### B5g. Frozen TEST evaluation of the shared-target soups (configurations fixed before any test look)

**What varies / split / metric / units.** Varied: pool (A-strong, A-nm, B-v2) and rule; split: TEST (80% of each official test set; each soup scored on its own task); metric: absolute own-task accuracy (%), mean over the 8 tasks and number of tasks beating uniform; the dev mean / wins of the same configuration from B5d/B5e are added for comparison.

**Source.** `results/aa/soup/test_freeze.jsonl`; `results/aa/soup/test_freeze_summary.json`; `results/aa/soup/partA_controlled_noise_strong_v2.jsonl`; `results/aa/soup/partA_controlled_noise_strong_normmatched.jsonl`; `results/aa/soup/partB_lora_variants_v2.jsonl`

| pool | rule | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | TEST mean | TEST tasks > uniform | dev mean | dev tasks > uniform |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-strong | uniform | 72.01 | 76.50 | 94.70 | 98.98 | 97.15 | 98.55 | 99.45 | 78.06 | 89.42 | – | 89.76 | – |
| A-strong | kappa b=1 | 70.66 | 75.67 | 94.64 | 98.89 | 97.03 | 98.47 | 99.40 | 77.46 | 89.03 | 0 | 89.42 | 3 |
| A-strong | kappa b=4 | 69.31 | 74.94 | 94.54 | 98.70 | 97.00 | 98.37 | 99.41 | 77.13 | 88.68 | 0 | 89.18 | 1 |
| A-strong | dsrank b=4 | 74.64 | 78.30 | 95.10 | 99.12 | 97.12 | 98.80 | 99.55 | 79.39 | 90.25 | 7 | 90.55 | 8 |
| A-strong | mp_invvar | 74.65 | 78.31 | 95.02 | 99.21 | 97.13 | 98.83 | 99.56 | 79.45 | 90.27 | 7 | 90.53 | 8 |
| A-strong | global:mp_invvar | 74.68 | 78.28 | 95.02 | 99.21 | 97.14 | 98.83 | 99.56 | 79.45 | 90.27 | 7 | 90.57 | 8 |
| A-strong | oracle_invvar | 74.67 | 78.33 | 95.04 | 99.21 | 97.13 | 98.83 | 99.56 | 79.45 | 90.28 | 7 | 90.51 | 8 |
| A-strong | cleanest_only | 74.60 | 78.35 | 95.04 | 99.12 | 97.13 | 98.83 | 99.55 | 79.32 | 90.24 | 7 | 90.52 | 7 |
| A-strong | noisiest_only | 1.08 | 5.33 | 51.75 | 85.83 | 87.61 | 74.05 | 55.38 | 59.31 | 52.54 | 0 | 53.05 | 0 |
| A-nm | uniform | 69.97 | 69.42 | 84.76 | 95.05 | 84.51 | 83.35 | 96.54 | 60.84 | 80.56 | – | 81.27 | – |
| A-nm | kappa b=1 | 70.00 | 69.75 | 84.44 | 95.60 | 84.54 | 82.74 | 96.45 | 60.51 | 80.50 | 4 | 81.15 | 3 |
| A-nm | kappa b=4 | 69.77 | 69.33 | 84.27 | 95.83 | 85.11 | 83.08 | 96.20 | 60.31 | 80.49 | 2 | 81.18 | 3 |
| A-nm | dsrank b=4 | 74.97 | 78.25 | 94.84 | 98.94 | 97.04 | 98.41 | 99.49 | 77.93 | 89.98 | 8 | 90.34 | 8 |
| A-nm | mp_invvar | 70.63 | 70.79 | 87.96 | 96.71 | 89.66 | 89.32 | 98.04 | 63.43 | 83.32 | 8 | 83.90 | 8 |
| A-nm | global:mp_invvar | 70.60 | 70.98 | 87.68 | 96.71 | 89.89 | 89.53 | 98.05 | 63.43 | 83.36 | 8 | 83.89 | 8 |
| A-nm | oracle_invvar | 74.80 | 77.88 | 94.68 | 98.80 | 96.90 | 98.35 | 99.46 | 76.93 | 89.73 | 8 | 90.07 | 8 |
| A-nm | cleanest_only | 74.96 | 78.21 | 94.86 | 98.89 | 97.04 | 98.42 | 99.49 | 77.73 | 89.95 | 8 | 90.39 | 8 |
| A-nm | noisiest_only | 64.63 | 61.67 | 67.98 | 68.43 | 46.63 | 43.22 | 67.53 | 47.61 | 58.46 | 0 | 59.02 | 0 |
| B-v2 | uniform | 75.14 | 81.05 | 95.22 | 99.26 | 97.20 | 98.81 | 99.62 | 78.26 | 90.57 | – | 91.04 | – |
| B-v2 | kappa b=1 | 75.39 | 80.97 | 95.18 | 99.21 | 97.07 | 98.80 | 99.59 | 78.06 | 90.53 | 1 | 90.97 | 4 |
| B-v2 | kappa b=4 | 75.54 | 80.93 | 95.08 | 99.21 | 97.06 | 98.67 | 99.60 | 78.06 | 90.52 | 1 | 90.90 | 2 |
| B-v2 | dsrank b=4 | 71.73 | 75.84 | 93.15 | 98.89 | 94.67 | 97.32 | 99.46 | 74.80 | 88.23 | 0 | 88.72 | 0 |
| B-v2 | mp_invvar | 74.10 | 78.83 | 94.58 | 98.89 | 95.19 | 98.44 | 99.46 | 77.06 | 89.57 | 0 | 90.10 | 0 |
| B-v2 | global:mp_invvar | 74.72 | 78.97 | 94.38 | 98.89 | 96.48 | 98.49 | 99.56 | 77.46 | 89.87 | 0 | 90.48 | 2 |
| B-v2 | oracle_best_single_by_dev | 72.79 | 81.13 | 94.90 | 99.26 | 97.32 | 98.49 | 99.45 | 75.93 | 89.91 | 2 | 90.89 | 6 |

**Legend.**
- Frozen rules: uniform; kappa (literal) at b=1 and b=4; dsrank (update stable rank) at b=4; mp_invvar (tuning-free lambda ~ median-sv(Delta)^-2) and global:mp_invvar; references oracle_invvar / cleanest_only / noisiest_only (A pools) and oracle_best_single_by_dev (B: best variant chosen by DEV accuracy, evaluated on test).
- Recomputed means and win counts equal test_freeze_summary.json (25 configurations; max abs difference of the means 0.0e+00; win-count mismatches: 0). No script in scripts/ writes test_freeze_summary.json; it is reproduced here from test_freeze.jsonl.
- Noise realisations are seeded exactly as in the dev runs (script docstring); consistent with that, the mean-lambda vectors stored in test_freeze.jsonl equal those of the dev runs for every frozen configuration compared (max abs difference 0.0e+00). A pools use noise levels 1, 4, 8, 16.

## B6. Tail-stress test

### B6a. Tail-stress edit: size of the function-null edit

**What varies / split / metric / units.** Varied: the edited expert (SVHN or SUN397); the edit floors the singular values of the expert's full weight at floor x sigma_min in every map with kappa(W0) > 1e3 (full fine-tuning, 8 FusionBench experts); metric: ‖W' - W‖_F / ‖Δ_j‖_F per edited map (%), median and max over the edited maps; no split involved.

**Source.** `results/aa/tail_stress_svhn.json`; `results/aa/tail_stress_sun397.json`; `scripts/aa_tail_stress.py`

| edited expert | floor (x sigma_min) | maps edited (of 73) | median ‖edit‖/‖Δ_j‖ (%) | max ‖edit‖/‖Δ_j‖ (%) |
| :--- | ---: | ---: | ---: | ---: |
| svhn | 10 | 47 | 0.468 | 8.61 |
| sun397 | 10 | 47 | 0.346 | 5.89 |

**Legend.**
- W' = U diag(max(s, floor * s_min)) V^T per edited map: kappa of the edited expert falls by about the floor factor while the weight changes by a tiny fraction of its fine-tuning update.

### B6b. Tail-stress edit: the edited expert alone, accuracy on all 8 tasks before and after the edit

**What varies / split / metric / units.** Varied: before / after the edit; split: dev; metric: absolute accuracy (%) of the single expert (lambda = e_j, alpha = 1) on each task, and the change in points.

**Source.** `results/aa/tail_stress_svhn.json`; `results/aa/tail_stress_sun397.json`

| edited expert | version | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svhn | before | 40.71 | 35.88 | 17.38 | 7.41 | 97.58 | 27.75 | 82.15 | 26.33 |
| svhn | after | 40.76 | 35.88 | 17.38 | 7.41 | 97.58 | 27.75 | 82.20 | 26.33 |
| svhn | after - before (pt) | +0.050 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.050 | +0.000 |
| sun397 | before | 74.56 | 46.21 | 55.63 | 48.89 | 28.45 | 27.16 | 44.05 | 43.88 |
| sun397 | after | 74.61 | 46.21 | 55.63 | 48.89 | 28.45 | 27.24 | 44.05 | 43.88 |
| sun397 | after - before (pt) | +0.050 | +0.000 | +0.000 | +0.000 | +0.000 | +0.079 | +0.000 | +0.000 |

**Legend.**
- Largest absolute change on any task: SVHN edit 0.050 pt, SUN397 edit 0.079 pt (dev sets of 376..5206 images, so one image = 0.02..0.27 pt on a task).

### B6c. Tail-stress: every row of both JSON files (literal rule on the original and on the tail-edited expert)

**What varies / split / metric / units.** Varied: edited expert, temperature b (rule scale: beta = b / median across-expert std of kappa), alpha, version (original / tail-edited); split: dev (8 tasks); metric: absolute merged accuracy (%) per task and mean, mean over maps of the weight lambda_j given to the edited expert j (all 73 maps and the 48 attention maps).

**Source.** `results/aa/tail_stress_svhn.json`; `results/aa/tail_stress_sun397.json`

| edited expert | b | alpha | version | avg acc (%) | sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd | mean lambda_j (all maps) | mean lambda_j (attention maps) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svhn | 4 | 1.6 | original | 70.03 | 64.56 | 63.50 | 71.75 | 75.19 | 75.95 | 65.84 | 90.80 | 52.66 | 0.1338 | 0.1384 |
| svhn | 4 | 1.6 | tail-edited | 68.71 | 63.05 | 60.76 | 64.68 | 61.30 | 93.45 | 60.25 | 93.55 | 52.66 | 0.4488 | 0.6175 |
| svhn | 16 | 1.6 | original | 69.63 | 63.83 | 63.68 | 70.40 | 71.67 | 79.73 | 66.23 | 89.35 | 52.13 | 0.1570 | 0.1738 |
| svhn | 16 | 1.6 | tail-edited | 64.95 | 61.66 | 57.77 | 59.29 | 47.78 | 95.56 | 55.98 | 93.45 | 48.14 | 0.5974 | 0.8436 |
| svhn | inf | 1.6 | original | 50.56 | 43.00 | 44.15 | 35.08 | 30.19 | 89.17 | 34.72 | 76.55 | 51.60 | 0.2192 | 0.1667 |
| svhn | inf | 1.6 | tail-edited | 42.24 | 33.65 | 33.96 | 18.25 | 14.26 | 95.58 | 29.18 | 79.30 | 33.78 | 0.6849 | 0.8750 |
| svhn | 4 | 1.0 | original | 67.33 | 66.90 | 63.50 | 71.03 | 74.44 | 66.23 | 58.19 | 84.35 | 53.99 | 0.1338 | 0.1384 |
| svhn | 4 | 1.0 | tail-edited | 67.48 | 65.99 | 61.88 | 66.19 | 65.00 | 85.17 | 55.34 | 87.85 | 52.39 | 0.4488 | 0.6175 |
| sun397 | 4 | 1.6 | original | 70.03 | 64.56 | 63.50 | 71.75 | 75.19 | 75.95 | 65.84 | 90.80 | 52.66 | 0.1131 | 0.1069 |
| sun397 | 4 | 1.6 | tail-edited | 68.37 | 68.84 | 61.32 | 67.30 | 75.37 | 71.07 | 60.65 | 88.65 | 53.72 | 0.3959 | 0.5370 |
| sun397 | 16 | 1.6 | original | 69.63 | 63.83 | 63.68 | 70.40 | 71.67 | 79.73 | 66.23 | 89.35 | 52.13 | 0.1155 | 0.1106 |
| sun397 | 16 | 1.6 | tail-edited | 66.36 | 70.83 | 59.20 | 64.60 | 72.04 | 67.98 | 57.05 | 86.50 | 52.66 | 0.5304 | 0.7416 |
| sun397 | inf | 1.6 | original | 50.56 | 43.00 | 44.15 | 35.08 | 30.19 | 89.17 | 34.72 | 76.55 | 51.60 | 0.0822 | 0.1250 |
| sun397 | inf | 1.6 | tail-edited | 48.63 | 55.94 | 42.16 | 33.81 | 29.26 | 79.52 | 26.76 | 72.40 | 49.20 | 0.5205 | 0.7917 |
| sun397 | 4 | 1.0 | original | 67.33 | 66.90 | 63.50 | 71.03 | 74.44 | 66.23 | 58.19 | 84.35 | 53.99 | 0.1131 | 0.1069 |
| sun397 | 4 | 1.0 | tail-edited | 65.49 | 69.37 | 62.00 | 68.10 | 73.33 | 62.12 | 53.29 | 82.55 | 53.19 | 0.3959 | 0.5370 |

**Legend.**
- Rule: lambda_i^p = softmax_i(-beta * kappa_2(W_i^p)) per linear map p, merge W0 + alpha * sum lambda_i Delta_i; b = inf is argmin selection (the expert with the smallest kappa in each map gets weight 1, so mean lambda_j = share of maps it wins: 16/73 = 0.2192 for SVHN before the edit).
- The 'original' rows of the two files are identical except for lambda_j: the merge before the edit does not depend on which expert is later edited (same accuracies in both files).

### B6d. Tail-stress summary: weight on the edited expert and merged accuracy before -> after the edit

**What varies / split / metric / units.** Varied: edited expert, b, alpha; split: dev; metric: mean lambda on the edited expert (all maps; attention maps), absolute merged accuracy (%) and its change (pt): overall, on the edited expert's own task, mean over the other 7 tasks, and the task with the largest drop (pt).

**Source.** `results/aa/tail_stress_svhn.json`; `results/aa/tail_stress_sun397.json`

| edited expert | b | alpha | mean lambda_j, all maps | mean lambda_j, attention maps | avg acc (%) | change, 8-task mean (pt) | change on own task (pt) | mean change on the other 7 tasks (pt) | largest drop: task (pt) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| svhn | 4 | 1.6 | 0.134 -> 0.449 | 0.138 -> 0.618 | 70.03 -> 68.71 | -1.32 | +17.50 | -4.00 | eurosat (-13.89) |
| svhn | 16 | 1.6 | 0.157 -> 0.597 | 0.174 -> 0.844 | 69.63 -> 64.95 | -4.67 | +15.83 | -7.60 | eurosat (-23.89) |
| svhn | inf | 1.6 | 0.219 -> 0.685 | 0.167 -> 0.875 | 50.56 -> 42.24 | -8.31 | +6.42 | -10.42 | dtd (-17.82) |
| svhn | 4 | 1.0 | 0.134 -> 0.449 | 0.138 -> 0.618 | 67.33 -> 67.48 | +0.15 | +18.94 | -2.54 | eurosat (-9.44) |
| sun397 | 4 | 1.6 | 0.113 -> 0.396 | 0.107 -> 0.537 | 70.03 -> 68.37 | -1.66 | +4.28 | -2.51 | gtsrb (-5.19) |
| sun397 | 16 | 1.6 | 0.116 -> 0.530 | 0.111 -> 0.742 | 69.63 -> 66.36 | -3.27 | +7.00 | -4.74 | svhn (-11.76) |
| sun397 | inf | 1.6 | 0.082 -> 0.521 | 0.125 -> 0.792 | 50.56 -> 48.63 | -1.92 | +12.95 | -4.05 | svhn (-9.64) |
| sun397 | 4 | 1.0 | 0.113 -> 0.396 | 0.107 -> 0.537 | 67.33 -> 65.49 | -1.84 | +2.47 | -2.45 | gtsrb (-4.91) |

**Legend.**
- Reading: a function-null edit (B6a, B6b) raises the literal rule's weight on the edited expert and moves the merge towards that expert: its own task gains, the other tasks lose.
- alpha = 1.0 rows (b = 4 only) are in the JSON files but not in report section 5, which lists alpha = 1.6 only.

## B7. Data-supported score statistics

### B7a. Data-supported scores (budget B): distribution over the 584 (map, expert) rows

**What varies / split / metric / units.** Varied: statistic; no split of the data is evaluated (second moments C come from the pooled unlabeled calibration inputs of the 8 tasks, no task IDs, no labels); metric: min / 10% / median / 90% / max over 73 maps x 8 experts; units as in the legend.

**Source.** `results/aa/data_supported/data_supported_stats.csv`; `scripts/aa_data_supported_scores.py`

| statistic | definition (script docstring) | n rows | min | q10 | median | q90 | max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| fnorm_D | tr(Delta C Delta^T) -- functional update energy on the data (review eq. 10) | 584 | 0.00105 | 0.00637 | 0.201 | 1.75 | 66.9 |
| rel_fnorm_D | fnorm_D / tr(W0 C W0^T) -- relative change of the layer's output energy | 584 | 5.39e-05 | 0.00029 | 0.000762 | 0.00842 | 0.0995 |
| kappa_WC | kappa2(W_i C^{1/2}) -- data-weighted condition number of the full layer | 584 | 230 | 708 | 5.95e+04 | 2.17e+07 | 1.01e+18 |
| dlogkappa_WC | log kappa_WC(W_i) - log kappa_WC(W0) -- base-relative version | 584 | -5.21 | -1.14 | 0.00216 | 0.795 | 5.98 |
| srank_DC | stable rank of Delta C^{1/2} -- data-supported update spread | 584 | 1 | 1.11 | 1.94 | 3.8 | 7.98 |
| erank_DC | entropy rank of Delta C^{1/2} | 584 | 1.03 | 1.96 | 9.97 | 35.7 | 78.6 |
| kplus_DC | kappa_plus(Delta C^{1/2}; 1e-2) | 584 | 67.9 | 98.6 | 99.6 | 99.9 | 100 |
| kappa_W0C | kappa_2(W0 C^{1/2}) of the BASE map (repeated for the 8 experts) | 584 | 233 | 693 | 5.3e+04 | 8.42e+06 | 9.28e+15 |

**Legend.**
- C_p = E[x x^T] of the map's input (base-model activations on the union of the 8 calibration splits); C^{1/2} by eigh in float64 with eigenvalues clipped at 0; the patch embedding uses the 3072-dim patch Gram.
- kappa_WC and kappa_W0C are condition numbers (dimensionless); fnorm_D is an expected squared output change (tr(Delta C Delta^T)); rel_fnorm_D is that divided by tr(W0 C W0^T); srank_DC, erank_DC, kplus_DC are dimensionless spread measures of the update's data-weighted spectrum.
- The full 584-row table is the CSV (columns: param, expert, the 8 statistics); 73 maps x 8 experts in the order of fullft_stats.npz.

### B7b. Data-supported scores by module type (median over maps x experts)

**What varies / split / metric / units.** Varied: module type; no split; metric: median of each statistic over the rows of that module type (12 maps x 8 experts; patch embedding: 1 map x 8).

**Source.** `results/aa/data_supported/data_supported_stats.csv`

| module | maps | rows | fnorm_D | rel_fnorm_D | kappa_WC | dlogkappa_WC | srank_DC | erank_DC | kplus_DC | kappa_W0C |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 12 | 96 | 0.279 | 0.00055 | 1.01e+06 | -0.124 | 1.79 | 8.77 | 99.5 | 1.45e+06 |
| k_proj | 12 | 96 | 0.272 | 0.0005 | 2.33e+05 | -0.0704 | 2.37 | 13.3 | 99.7 | 8.48e+05 |
| v_proj | 12 | 96 | 0.132 | 0.000834 | 5.29e+04 | 0.00974 | 2.8 | 19.7 | 99.7 | 5e+04 |
| out_proj | 12 | 96 | 0.00508 | 0.00101 | 1.43e+05 | -0.159 | 2.44 | 11.8 | 99.6 | 2.37e+05 |
| fc1 | 12 | 96 | 1.99 | 0.000557 | 2.55e+04 | 0.00171 | 1.39 | 5.26 | 99.7 | 2.55e+04 |
| fc2 | 12 | 96 | 0.122 | 0.00894 | 664 | 0.0172 | 1.1 | 1.9 | 99.4 | 647 |
| patch_embedding | 1 | 8 | 43 | 0.064 | 2.96e+04 | -0.373 | 1.05 | 1.35 | 91.4 | 4.14e+04 |
| all | 73 | 584 | 0.201 | 0.000762 | 5.95e+04 | 0.00216 | 1.94 | 9.97 | 99.6 | 5.3e+04 |

### B7c. Data-supported scores by expert (median over the 73 maps)

**What varies / split / metric / units.** Varied: expert; no split; metric: median over the 73 maps of each statistic for that expert.

**Source.** `results/aa/data_supported/data_supported_stats.csv`

| expert | fnorm_D | rel_fnorm_D | kappa_WC | dlogkappa_WC | srank_DC | erank_DC | kplus_DC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sun397 | 0.208 | 0.000786 | 6.49e+04 | 0.00402 | 2.88 | 29.9 | 99.7 |
| cars | 0.112 | 0.000471 | 5.43e+04 | -3.06e-05 | 2.43 | 16.3 | 99.6 |
| resisc45 | 0.203 | 0.000897 | 5.48e+04 | 0.00161 | 2.18 | 11.9 | 99.6 |
| eurosat | 0.258 | 0.000835 | 5.64e+04 | 0.00293 | 1.61 | 6.81 | 99.5 |
| svhn | 0.335 | 0.00108 | 6.35e+04 | 0.00343 | 1.61 | 6.17 | 99.3 |
| gtsrb | 0.27 | 0.000788 | 5.78e+04 | 0.000911 | 1.96 | 10.8 | 99.5 |
| mnist | 0.206 | 0.000675 | 6.81e+04 | 0.013 | 1.87 | 7.51 | 99.5 |
| dtd | 0.0849 | 0.000407 | 5.2e+04 | 0.000554 | 2.03 | 12.3 | 99.7 |

**Legend.**
- Compare with the expert-identity structure of B2f: the update-energy and update-spread statistics differ systematically between experts, which is why they correlate with the expert-level preference in B4a.

### B7d. Spread of each data-supported statistic across the 8 experts inside a map

**What varies / split / metric / units.** Varied: statistic; no split; metric: per map, max/min (or max-min for the signed dlogkappa_WC) over the 8 experts; quantiles over the 73 maps give the dynamic range the softmax rule has to work with.

**Source.** `results/aa/data_supported/data_supported_stats.csv`

| statistic | spread measure | q10 over maps | median over maps | q90 over maps |
| :--- | :--- | ---: | ---: | ---: |
| fnorm_D | max / min | 2.29 | 3.84 | 7.33 |
| rel_fnorm_D | max / min | 2.29 | 3.84 | 7.33 |
| kappa_WC | max / min | 1 | 5.16 | 53.4 |
| dlogkappa_WC | max - min (log units) | 0.00447 | 1.64 | 3.98 |
| srank_DC | max / min | 1.13 | 1.95 | 3.07 |
| erank_DC | max / min | 1.94 | 4.03 | 7.72 |
| kplus_DC | max / min | 1.01 | 1.01 | 1.03 |

### B7e. Data-weighted vs plain condition number of the BASE map, and numerically singular rows

**What varies / split / metric / units.** Varied: module type; no split; metric: median kappa_2(W0) (from per_map_grad_vs_kappa.csv) vs median kappa_2(W0 C^{1/2}), median ratio, Spearman correlation across maps, and the number of maps whose data-weighted base kappa exceeds 1e12.

**Source.** `results/aa/data_supported/data_supported_stats.csv`; `results/aa/grad_pref/per_map_grad_vs_kappa.csv`

| module | maps | median kappa(W0) | median kappa(W0 C^1/2) | median ratio | Spearman across maps | maps with kappa(W0 C^1/2) > 1e12 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 12 | 40045 | 1.45e+06 | 15.6 | +0.993 | 1 |
| k_proj | 12 | 79139 | 848075 | 20.3 | +0.930 | 2 |
| v_proj | 12 | 3747 | 50039 | 11.2 | +0.930 | 1 |
| out_proj | 12 | 13961 | 237156 | 16.9 | +0.804 | 0 |
| fc1 | 12 | 18.99 | 25491 | 1.03e+03 | +0.168 | 0 |
| fc2 | 12 | 41.38 | 647 | 16.2 | +0.706 | 0 |
| patch_embedding | 1 | 531 | 41384 | 77.9 | – | 0 |
| all | 73 | 3379 | 52998 | 17.7 | +0.842 | 4 |

**Legend.**
- Rows with kappa_WC > 1e12: 25 of 584, in 4 maps (L0.k_proj x8, L0.v_proj x8, L0.q_proj x8, L1.k_proj x1); values of order 1e15 and above are at the float64 round-off floor (1/eps = 4.5e+15), i.e. not meaningful condition numbers; the cause is not recorded in these files (plausibly a numerically rank-deficient pooled input Gram C for the layer-0 inputs). Statistics derived from those rows (kappa_WC, dlogkappa_WC) are round-off in B4a/B4b.
- median kappa(W0): the base-map condition number from fp64 SVD of the stored fp32 matrix.

### B7f. Agreement between data-supported statistics: mean within-map Spearman across the 8 experts

**What varies / split / metric / units.** Varied: pair of statistics; no split; metric: Spearman correlation across the 8 experts inside each map, averaged over the 73 maps (dimensionless).

**Source.** `results/aa/data_supported/data_supported_stats.csv`

| statistic | fnorm_D | rel_fnorm_D | kappa_WC | dlogkappa_WC | srank_DC | erank_DC | kplus_DC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fnorm_D | +1.00 | +1.00 | +0.13 | +0.13 | -0.47 | -0.53 | -0.14 |
| rel_fnorm_D | +1.00 | +1.00 | +0.13 | +0.13 | -0.47 | -0.53 | -0.14 |
| kappa_WC | +0.13 | +0.13 | +1.00 | +1.00 | -0.03 | -0.06 | -0.04 |
| dlogkappa_WC | +0.13 | +0.13 | +1.00 | +1.00 | -0.03 | -0.06 | -0.04 |
| srank_DC | -0.47 | -0.47 | -0.03 | -0.03 | +1.00 | +0.86 | +0.25 |
| erank_DC | -0.53 | -0.53 | -0.06 | -0.06 | +0.86 | +1.00 | +0.23 |
| kplus_DC | -0.14 | -0.14 | -0.04 | -0.04 | +0.25 | +0.23 | +1.00 |

**Legend.**
- fnorm_D and rel_fnorm_D have the same within-map ranks (the divisor tr(W0 C W0^T) is constant inside a map); kappa_WC and dlogkappa_WC likewise (the subtracted log kappa_WC(W0) is constant inside a map).

## X. Cross-check against docs/aa/AA_SIDE_REPORT.md

### X1. Cross-check of every number of the report that lives in this scope (recomputed from the raw files)

**What varies / split / metric / units.** For each quoted number of docs/aa/AA_SIDE_REPORT.md (sections 1-5) and docs/aa/MATH_kappa_of_finetuned_weights.md the value is parsed out of the report text by the script and compared with the value recomputed from the result files; metric and units are those of the quoted number (accuracy in %, absolute unless stated; correlations dimensionless).

**Source.** `docs/aa/AA_SIDE_REPORT.md`; `docs/aa/MATH_kappa_of_finetuned_weights.md`

| # | report location | claim (as written) | reported | recomputed | status | source of the recomputed value / note |
| ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | §3 table l.102, full-FT, uniform (best alpha), dev | 70.3 / – | 70.3 | 70.280 | OK | uniform_alpha_grid.json |
| 2 | §3 table l.102, full-FT, global oracle, dev | 70.6 / 69.8 | 70.6 | 70.605 | OK | global_alpha_seed0.json (dev) |
| 3 | §3 table l.102, full-FT, global oracle, test | 70.6 / 69.8 | 69.8 | 69.772 | OK | global_alpha_seed0.json (test) |
| 4 | §3 table l.102, full-FT, layerwise oracle, alpha = 1, dev | 75.0 / 73.9 | 75.0 | 75.041 | OK | layerwise_a1_seed0.json (dev) |
| 5 | §3 table l.102, full-FT, layerwise oracle, alpha = 1, test | 75.0 / 73.9 | 73.9 | 73.907 | OK | layerwise_a1_seed0.json (test) |
| 6 | §3 table l.102, full-FT, layerwise oracle + alpha, dev | 78.7 / 77.2 | 78.7 | 78.646 | ROUNDING (OK for seed 1, 2-seed mean) | layerwise_alpha_seed0.json (dev) -- ; other candidates: seed 1 78.675, 2-seed mean 78.660 |
| 7 | §3 table l.102, full-FT, layerwise oracle + alpha, test | 78.7 / 77.2 | 77.2 | 77.234 | OK | layerwise_alpha_seed0.json (test) |
| 8 | §3 table l.103, LoRA, uniform (best alpha), dev | 52.3 / 51.2 | 52.3 | 52.264 | OK | uniform_alpha_grid.json |
| 9 | §3 table l.103, LoRA, uniform (best alpha), test | 52.3 / 51.2 | 51.2 | 51.220 | OK | results/aa/lora/test_freeze.jsonl (label = uniform) |
| 10 | §3 table l.103, LoRA, global oracle, dev | 52.7 / 52.4 | 52.7 | 52.656 | OK | global_alpha_seed0.json (dev) |
| 11 | §3 table l.103, LoRA, global oracle, test | 52.7 / 52.4 | 52.4 | 52.357 | OK | global_alpha_seed0.json (test) |
| 12 | §3 table l.103, LoRA, layerwise oracle, alpha = 1, dev | 62.3 / 61.7 | 62.3 | 62.314 | OK | layerwise_a1_seed0.json (dev) |
| 13 | §3 table l.103, LoRA, layerwise oracle, alpha = 1, test | 62.3 / 61.7 | 61.7 | 61.696 | OK | layerwise_a1_seed0.json (test) |
| 14 | §3 table l.103, LoRA, layerwise oracle + alpha, dev | 64.8 / 64.1 | 64.8 | 64.802 | OK | layerwise_alpha_seed0.json (dev) |
| 15 | §3 table l.103, LoRA, layerwise oracle + alpha, test | 64.8 / 64.1 | 64.1 | 64.078 | OK | layerwise_alpha_seed0.json (test) |
| 16 | §4 table l.119, A, 'uniform' | 89.8 | 89.8 | 89.760 | OK | partA_controlled_noise_strong_v2.jsonl |
| 17 | §4 table l.119, A-nm, 'uniform' | 81.3 | 81.3 | 81.272 | OK | partA_controlled_noise_strong_normmatched.jsonl |
| 18 | §4 table l.119, B, 'uniform' | 91.0 | 91.0 | 91.040 | OK | partB_lora_variants_v2.jsonl |
| 19 | §4 table l.120, A, 'oracle (inverse variance; quality softma' | 90.5 | 90.5 | 90.509 | OK | partA_controlled_noise_strong_v2.jsonl, rule oracle_invvar |
| 20 | §4 table l.120, A-nm, 'oracle (inverse variance; quality softma' | 90.1 | 90.1 | 90.073 | OK | partA_controlled_noise_strong_normmatched.jsonl, rule oracle_invvar |
| 21 | §4 table l.120, B, 'oracle (inverse variance; quality softma' | 91.6 | 91.6 | 91.573 | OK | partB_lora_variants_v2.jsonl, rule oracle_quality_softmax100 |
| 22 | §4 table l.121, A, 'literal κ, b = 1 / 16 / ∞', b=1 | 89.4 / 88.8 / **58.8** | 89.4 | 89.425 | OK | partA_controlled_noise_strong_v2.jsonl, rule kappa |
| 23 | §4 table l.121, A, 'literal κ, b = 1 / 16 / ∞', b=16 | 89.4 / 88.8 / **58.8** | 88.8 | 88.770 | OK | partA_controlled_noise_strong_v2.jsonl, rule kappa |
| 24 | §4 table l.121, A, 'literal κ, b = 1 / 16 / ∞', b=inf | 89.4 / 88.8 / **58.8** | 58.8 | 58.791 | OK | partA_controlled_noise_strong_v2.jsonl, rule kappa |
| 25 | §4 table l.121, A-nm, 'literal κ, b = 1 / 16 / ∞', b=1 | 81.2 / 81.2 / 74.0 | 81.2 | 81.155 | OK | partA_controlled_noise_strong_normmatched.jsonl, rule kappa |
| 26 | §4 table l.121, A-nm, 'literal κ, b = 1 / 16 / ∞', b=16 | 81.2 / 81.2 / 74.0 | 81.2 | 81.207 | OK | partA_controlled_noise_strong_normmatched.jsonl, rule kappa |
| 27 | §4 table l.121, A-nm, 'literal κ, b = 1 / 16 / ∞', b=inf | 81.2 / 81.2 / 74.0 | 74.0 | 73.961 | OK | partA_controlled_noise_strong_normmatched.jsonl, rule kappa |
| 28 | §4 table l.121, B, 'literal κ, b = 1 / 16 / ∞', b=1 | 91.0 / 90.4 / 88.3 | 91.0 | 90.973 | OK | partB_lora_variants_v2.jsonl, rule kappa |
| 29 | §4 table l.121, B, 'literal κ, b = 1 / 16 / ∞', b=16 | 91.0 / 90.4 / 88.3 | 90.4 | 90.411 | OK | partB_lora_variants_v2.jsonl, rule kappa |
| 30 | §4 table l.121, B, 'literal κ, b = 1 / 16 / ∞', b=inf | 91.0 / 90.4 / 88.3 | 88.3 | 88.286 | OK | partB_lora_variants_v2.jsonl, rule kappa |
| 31 | §4 table l.122, A, 'tuning-free GD/MP inverse variance λ ∝ σ' | 90.5 | 90.5 | 90.529 | OK | partA_controlled_noise_strong_v2.jsonl, rule mp_invvar |
| 32 | §4 table l.122, A-nm, 'tuning-free GD/MP inverse variance λ ∝ σ' | 83.9 | 83.9 | 83.905 | OK | partA_controlled_noise_strong_normmatched.jsonl, rule mp_invvar |
| 33 | §4 table l.122, B, 'tuning-free GD/MP inverse variance λ ∝ σ' | **90.1** | 90.1 | 90.105 | OK | partB_lora_variants_v2.jsonl, rule mp_invvar |
| 34 | §4 table l.123, A, 'update shape at b = 4 (stable rank / noi', dsrank b=4 | 90.5 | 90.5 | 90.548 | OK | partA_controlled_noise_strong_v2.jsonl |
| 35 | §4 table l.123, A, 'update shape at b = 4 (stable rank / noi', dnoiseshare b=4 | 90.5 | 90.5 | 90.525 | OK | partA_controlled_noise_strong_v2.jsonl |
| 36 | §4 table l.123, A, 'update shape at b = 4 (stable rank / noi', dmedsv b=4 | 90.5 | 90.5 | 90.525 | OK | partA_controlled_noise_strong_v2.jsonl |
| 37 | §4 table l.123, A-nm, 'update shape at b = 4 (stable rank / noi', dsrank b=4 | **90.3–90.4** | 90.3/90.4 | 90.338 | OK | partA_controlled_noise_strong_normmatched.jsonl -- reported as a range; each of the three rules must round to one end |
| 38 | §4 table l.123, A-nm, 'update shape at b = 4 (stable rank / noi', dnoiseshare b=4 | **90.3–90.4** | 90.3/90.4 | 90.397 | OK | partA_controlled_noise_strong_normmatched.jsonl -- reported as a range; each of the three rules must round to one end |
| 39 | §4 table l.123, A-nm, 'update shape at b = 4 (stable rank / noi', dmedsv b=4 | **90.3–90.4** | 90.3/90.4 | 90.397 | OK | partA_controlled_noise_strong_normmatched.jsonl -- reported as a range; each of the three rules must round to one end |
| 40 | §4 table l.123, B, 'update shape at b = 4 (stable rank / noi', best b over the 3 shape rules | ≤ 91.1 (best b) | <= 91.1 | 90.813 | OK | partB_lora_variants_v2.jsonl -- the cell is an upper bound on the best b; at b=4 itself the three rules give: dsrank 88.72, dnoiseshare 88.77, dmedsv 87.60 |
| 41 | §4 table l.124, A, 'update norm, b = 4', dnorm b=4 | 90.5 | 90.5 | 90.543 | OK | partA_controlled_noise_strong_v2.jsonl |
| 42 | §4 table l.124, A-nm, 'update norm, b = 4', dnorm b=4 | 78.9 | 78.9 | 78.887 | OK | partA_controlled_noise_strong_normmatched.jsonl |
| 43 | §4 table l.124, B, 'update norm, b = 4', dnorm b=4 | ≈ 91.0 | 91.0 | 88.242 | MISMATCH | partB_lora_variants_v2.jsonl -- B cell is stated as '≈ 91.0' |
| 44 | §4 frozen-TEST table l.141, A, 'uniform' | 89.4 | 89.4 | 89.424 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 45 | §4 frozen-TEST table l.141, A-nm, 'uniform' | 80.6 | 80.6 | 80.555 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 46 | §4 frozen-TEST table l.141, B, 'uniform' | 90.6 | 90.6 | 90.571 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 47 | §4 frozen-TEST table l.142, A, 'literal κ, b = 1 / 4', kappa b=1 | 89.0 / 88.7 (0/8 tasks above uniform) | 89.0 | 89.028 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 48 | §4 frozen-TEST table l.142, A, 'literal κ, b = 1 / 4', kappa b=4 | 89.0 / 88.7 (0/8 tasks above uniform) | 88.7 | 88.677 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 49 | §4 frozen-TEST table l.142, A, 'literal κ, b = 1 / 4', kappa b=1, tasks above uniform | 89.0 / 88.7 (0/8 tasks above uniform) | 0/8 | 0/8 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 50 | §4 frozen-TEST table l.142, A, 'literal κ, b = 1 / 4', kappa b=4, tasks above uniform | 89.0 / 88.7 (0/8 tasks above uniform) | 0/8 | 0/8 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 51 | §4 frozen-TEST table l.142, A-nm, 'literal κ, b = 1 / 4', kappa b=1 | 80.5 / 80.5 | 80.5 | 80.504 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 52 | §4 frozen-TEST table l.142, A-nm, 'literal κ, b = 1 / 4', kappa b=4 | 80.5 / 80.5 | 80.5 | 80.487 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 53 | §4 frozen-TEST table l.142, B, 'literal κ, b = 1 / 4', kappa b=1 | 90.5 / 90.5 | 90.5 | 90.534 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 54 | §4 frozen-TEST table l.142, B, 'literal κ, b = 1 / 4', kappa b=4 | 90.5 / 90.5 | 90.5 | 90.519 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 55 | §4 frozen-TEST table l.143, A, 'update stable rank, b = 4', dsrank b=4 | 90.3 (7/8) | 90.3 | 90.252 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 56 | §4 frozen-TEST table l.143, A, 'update stable rank, b = 4', tasks above uniform | 90.3 (7/8) | 7/8 | 7/8 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 57 | §4 frozen-TEST table l.143, A-nm, 'update stable rank, b = 4', dsrank b=4 | 90.0 (8/8) | 90.0 | 89.982 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 58 | §4 frozen-TEST table l.143, A-nm, 'update stable rank, b = 4', tasks above uniform | 90.0 (8/8) | 8/8 | 8/8 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 59 | §4 frozen-TEST table l.143, B, 'update stable rank, b = 4', dsrank b=4 | 88.2 (0/8) | 88.2 | 88.233 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 60 | §4 frozen-TEST table l.143, B, 'update stable rank, b = 4', tasks above uniform | 88.2 (0/8) | 0/8 | 0/8 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 61 | §4 frozen-TEST table l.144, A, 'GD/MP inverse variance', mp_invvar | 90.3 | 90.3 | 90.272 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 62 | §4 frozen-TEST table l.144, A-nm, 'GD/MP inverse variance', mp_invvar | 83.3 | 83.3 | 83.317 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 63 | §4 frozen-TEST table l.144, B, 'GD/MP inverse variance', mp_invvar | 89.6 (0/8) | 89.6 | 89.569 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 64 | §4 frozen-TEST table l.144, B, 'GD/MP inverse variance', tasks above uniform | 89.6 (0/8) | 0/8 | 0/8 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 65 | §4 frozen-TEST table l.145, A, 'oracle', oracle_invvar | 90.3 | 90.3 | 90.280 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 66 | §4 frozen-TEST table l.145, A-nm, 'oracle', oracle_invvar | 89.7 | 89.7 | 89.726 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 67 | §4 frozen-TEST table l.145, B, 'oracle', oracle_best_single_by_dev | 89.9 (best single by dev) | 89.9 | 89.908 | OK | test_freeze.jsonl / test_freeze_summary.json |
| 68 | §5 table l.156, SVHN edited, b = 4, α = 1.6, lambda before | 0.13 → 0.45 (0.14 → 0.62) | 0.13 | 0.1338 | OK | tail_stress_*.json |
| 69 | §5 table l.156, SVHN edited, b = 4, α = 1.6, lambda after | 0.13 → 0.45 (0.14 → 0.62) | 0.45 | 0.4488 | OK | tail_stress_*.json |
| 70 | §5 table l.156, SVHN edited, b = 4, α = 1.6, attn before | 0.13 → 0.45 (0.14 → 0.62) | 0.14 | 0.1384 | OK | tail_stress_*.json |
| 71 | §5 table l.156, SVHN edited, b = 4, α = 1.6, attn after | 0.13 → 0.45 (0.14 → 0.62) | 0.62 | 0.6175 | OK | tail_stress_*.json |
| 72 | §5 table l.156, SVHN edited, b = 4, α = 1.6, acc before | 70.0 → 68.7 | 70.0 | 70.029 | OK | tail_stress_*.json |
| 73 | §5 table l.156, SVHN edited, b = 4, α = 1.6, acc after | 70.0 → 68.7 | 68.7 | 68.712 | OK | tail_stress_*.json |
| 74 | §5 table l.157, SVHN, b = 16, lambda before | 0.16 → 0.60 (0.84) | 0.16 | 0.1570 | OK | tail_stress_*.json |
| 75 | §5 table l.157, SVHN, b = 16, lambda after | 0.16 → 0.60 (0.84) | 0.60 | 0.5974 | OK | tail_stress_*.json |
| 76 | §5 table l.157, SVHN, b = 16, attn after | 0.16 → 0.60 (0.84) | 0.84 | 0.8436 | OK | tail_stress_*.json |
| 77 | §5 table l.157, SVHN, b = 16, acc before | 69.6 → 65.0 | 69.6 | 69.627 | OK | tail_stress_*.json |
| 78 | §5 table l.157, SVHN, b = 16, acc after | 69.6 → 65.0 | 65.0 | 64.954 | OK | tail_stress_*.json |
| 79 | §5 table l.158, SVHN, b = ∞, lambda before | 0.22 → 0.69 | 0.22 | 0.2192 | OK | tail_stress_*.json |
| 80 | §5 table l.158, SVHN, b = ∞, lambda after | 0.22 → 0.69 | 0.69 | 0.6849 | ROUNDING | tail_stress_*.json -- = 50/73 |
| 81 | §5 table l.158, SVHN, b = ∞, acc before | 50.6 → 42.2 | 50.6 | 50.556 | OK | tail_stress_*.json |
| 82 | §5 table l.158, SVHN, b = ∞, acc after | 50.6 → 42.2 | 42.2 | 42.245 | OK | tail_stress_*.json |
| 83 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, lambda before (b=4) | 0.11 → 0.40 / 0.53 / 0.52 | 0.11 | 0.1131 | OK | tail_stress_*.json |
| 84 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, lambda after b=4 | 0.11 → 0.40 / 0.53 / 0.52 | 0.40 | 0.3959 | OK | tail_stress_*.json |
| 85 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, lambda after b=16 | 0.11 → 0.40 / 0.53 / 0.52 | 0.53 | 0.5304 | OK | tail_stress_*.json |
| 86 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, lambda after b=inf | 0.11 → 0.40 / 0.53 / 0.52 | 0.52 | 0.5205 | OK | tail_stress_*.json |
| 87 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, acc before b=4 | 70.0 → 68.4; 69.6 → 66.4; 50.6 → 48.6 | 70.0 | 70.029 | OK | tail_stress_*.json |
| 88 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, acc after b=4 | 70.0 → 68.4; 69.6 → 66.4; 50.6 → 48.6 | 68.4 | 68.366 | OK | tail_stress_*.json |
| 89 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, acc before b=16 | 70.0 → 68.4; 69.6 → 66.4; 50.6 → 48.6 | 69.6 | 69.627 | OK | tail_stress_*.json |
| 90 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, acc after b=16 | 70.0 → 68.4; 69.6 → 66.4; 50.6 → 48.6 | 66.4 | 66.358 | OK | tail_stress_*.json |
| 91 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, acc before b=inf | 70.0 → 68.4; 69.6 → 66.4; 50.6 → 48.6 | 50.6 | 50.556 | OK | tail_stress_*.json |
| 92 | §5 table l.159, SUN397 edited, b = 4 / 16 / ∞, acc after b=inf | 70.0 → 68.4; 69.6 → 66.4; 50.6 → 48.6 | 48.6 | 48.633 | OK | tail_stress_*.json |
| 93 | §2 table l.47, 'κ₂(W) (literal)', ρ vs gradient pref a=1 | +0.045 | +0.045 | 0.04534 | OK | score_vs_targets.json gradpref_a1 |
| 94 | §2 table l.47, 'κ₂(W) (literal)', ρ vs oracle λ | −0.012 | -0.012 | -0.01174 | OK | score_vs_targets.json oracle_layerwise_alpha |
| 95 | §2 table l.47, 'κ₂(W) (literal)', within-expert vs gradient | +0.095 (p 0.03) vs gradient; ≈ 0 vs oracle | 0.095 | 0.09536 | OK | score_vs_targets.json gradpref_a1 |
| 96 | §2 table l.47, 'κ₂(W) (literal)', its p | +0.095 (p 0.03) vs gradient; ≈ 0 vs oracle | 0.03 | 0.0300 | OK | score_vs_targets.json (two-sided); one-sided p(null>=obs) in decompose_pref.json = 0.0166 |
| 97 | §2 table l.47, 'κ₂(W) (literal)', within-expert vs oracle | +0.095 (p 0.03) vs gradient; ≈ 0 vs oracle | ≈ 0 | +0.0096 (oracle alpha) / +0.0374 (oracle alpha=1) | OK | score_vs_targets.json oracle_layerwise_alpha / oracle_layerwise_a1 |
| 98 | §2 table l.48, 'κ(W C^{1/2}) (budget B)', ρ vs gradient pref a=1 | +0.035 | +0.035 | 0.03523 | OK | score_vs_targets.json |
| 99 | §2 table l.48, 'κ(W C^{1/2}) (budget B)', ρ vs oracle λ | −0.073 | -0.073 | -0.07273 | OK | score_vs_targets.json |
| 100 | §2 table l.48, 'κ(W C^{1/2}) (budget B)', within-expert p over the 4 targets ('p >= 0.14') | ≈ 0 (p ≥ 0.14) | p >= 0.14 | 0.1400 | OK | score_vs_targets.json (min over 4 targets) |
| 101 | §2 table l.49, 'update spread (stable / entrop', stable rank vs gradient pref a=1 | −0.43 / −0.48 | -0.43 | -0.4273 | OK | score_vs_targets.json |
| 102 | §2 table l.49, 'update spread (stable / entrop', entropy rank vs gradient pref a=1 | −0.43 / −0.48 | -0.48 | -0.4746 | ROUNDING | score_vs_targets.json (analysis_grad_vs_kappa.json: -0.4746) |
| 103 | §2 table l.49, 'update spread (stable / entrop', stable rank vs oracle λ | −0.10 | -0.10 | -0.0998 | OK | score_vs_targets.json oracle_layerwise_alpha |
| 104 | §2 table l.49, 'update spread (stable / entrop', entropy rank vs oracle λ | −0.10 | -0.10 | -0.1008 | OK | score_vs_targets.json oracle_layerwise_alpha |
| 105 | §2 table l.49, 'update spread (stable / entrop', between-expert | ≈ 0; all of it is between-expert (expert identity, ±0.8) | ±0.8 | abs(between-expert) from 0.54 to 0.85 over 4 targets x 2 scores | IMPRECISE | score_vs_targets.json -- per target (min..max over the two scores): gradpref_a1 0.85..0.85; gradpref_a2.4 0.62..0.68; oracle_layerwise_alpha 0.54..0.60; oracle_layerwise_a1 0.82..0.84 |
| 106 | §2 table l.49, 'update spread (stable / entrop', within-expert | ≈ 0; all of it is between-expert (expert identity, ±0.8) | ≈ 0 | max abs(within-expert) = 0.035 | OK | score_vs_targets.json |
| 107 | §2 table l.50, 'functional energy tr(ΔCΔᵀ) (bu', ρ vs gradient pref a=1 | +0.33 | +0.33 | 0.3301 | OK | score_vs_targets.json |
| 108 | §2 table l.50, 'functional energy tr(ΔCΔᵀ) (bu', ρ vs oracle λ | +0.07 | +0.07 | 0.0678 | OK | score_vs_targets.json oracle_layerwise_alpha |
| 109 | §2 table l.50, 'functional energy tr(ΔCΔᵀ) (bu', within-expert | +0.12 to +0.16 (p ≤ 0.007), the only per-layer predictor | +0.12 to +0.16 (p <= 0.007) | gradpref_a1: +0.126 (p 0.005); gradpref_a2.4: +0.119 (p 0.007); oracle_layerwise_alpha: +0.077 (p 0.084); oracle_layerwise_a1: +0.164 (p 0.000) | PARTIAL: holds for gradpref_a1, gradpref_a2.4, oracle_layerwise_a1; not for oracle_layerwise_alpha | score_vs_targets.json B:fnorm_D -- the stated range/p-bound is checked per target |
| 110 | §2 table l.50, 'functional energy tr(ΔCΔᵀ) (bu', 'the only per-layer predictor' | +0.12 to +0.16 (p ≤ 0.007), the only per-layer predictor | only functional energy | also p<0.05 vs gradient pref: gradpref_a1 A:fro_D +0.099 (p 0.024); gradpref_a1 A:smin_W -0.090 (p 0.041); gradpref_a2.4 A:fro_D +0.123 (p 0.003); gradpref_a2.4 A:smin_W -0.128 (p 0.004) \| significant within-expert vs oracle targets: oracle_layerwise_a1 B:fnorm_D, oracle_layerwise_a1 B:rel_fnorm_D | NOT UNIQUE (budget-A scores also significant) | score_vs_targets.json -- A:fro_D (update norm, weights only) and A:smin_W (negative sign) are significant within-expert for the gradient-preference targets |
| 111 | §2b table l.64, 'gradient pref, α = 1', within-map ρ | +0.076 (0.09) | 0.076 | 0.07599 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 112 | §2b table l.64, 'gradient pref, α = 1', its p | +0.076 (0.09) | 0.09 | 0.090 | OK (permutation noise, abs(diff) <= 0.03) | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 113 | §2b table l.64, 'gradient pref, α = 1', within-expert | +0.070 (0.11) | 0.070 | 0.06952 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 114 | §2b table l.64, 'gradient pref, α = 1', its p | +0.070 (0.11) | 0.11 | 0.125 | OK (permutation noise, abs(diff) <= 0.03) | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 115 | §2b table l.65, 'gradient pref, α = 2.4', within-map ρ | +0.076 (0.10); attention +0.18, MLP −0.12 | 0.076 | 0.07632 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 116 | §2b table l.65, 'gradient pref, α = 2.4', its p | +0.076 (0.10); attention +0.18, MLP −0.12 | 0.1 | 0.087 | OK (permutation noise, abs(diff) <= 0.03) | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 117 | §2b table l.65, 'gradient pref, α = 2.4', attention maps | +0.076 (0.10); attention +0.18, MLP −0.12 | 0.18 | 0.1776 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 118 | §2b table l.65, 'gradient pref, α = 2.4', MLP maps | +0.076 (0.10); attention +0.18, MLP −0.12 | -0.12 | -0.1181 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 119 | §2b table l.65, 'gradient pref, α = 2.4', within-expert | +0.059 (0.19) | 0.059 | 0.05891 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 120 | §2b table l.65, 'gradient pref, α = 2.4', its p | +0.059 (0.19) | 0.19 | 0.194 | OK (permutation noise, abs(diff) <= 0.03) | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 121 | §2b table l.66, 'oracle layerwise + α (2 seeds)', within-map ρ (mean of the 2 seeds) | +0.09 (0.046) | 0.09 | 0.0903 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 122 | §2b table l.66, 'oracle layerwise + α (2 seeds)', its p (per seed) | +0.09 (0.046) | 0.046 | 0.039, 0.046 | OK (permutation noise, abs(diff) <= 0.03) | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 123 | §2b table l.66, 'oracle layerwise + α (2 seeds)', within-expert (mean of the 2 seeds) | +0.075 (≈ 0.08) | 0.075 | 0.07549 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 124 | §2b table l.66, 'oracle layerwise + α (2 seeds)', its p (per seed) | +0.075 (≈ 0.08) | 0.08 | 0.084, 0.093 | OK (permutation noise, abs(diff) <= 0.03) | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 125 | §2b table l.67, 'oracle, α = 1', within-map ρ (seed 0) | +0.03 (0.48) | 0.03 | 0.0310 | OK | B4b (re-derived from results/spectra/stats.csv + in-scope targets) |
| 126 | §2b table l.67, 'oracle, α = 1', its p (seed 0) | +0.03 (0.48) | 0.48 | 0.488 | OK (permutation noise, abs(diff) <= 0.03) | B4b (re-derived from results/spectra/stats.csv + in-scope targets) -- seed 1: rho +0.055, p 0.211 |
| 127 | §1 l.29 | Within-map ρ(gradient preference, κ) = +0.045 | 0.045 | 0.04534 | OK | analysis_grad_vs_kappa.json (alpha 1) |
| 128 | §1 l.29 | oracle λ vs κ: lower end −0.02 | -0.02 | -0.0199 | OK | analysis_oracle_vs_kappa.json (4 layerwise fits) |
| 129 | §1 l.29 | oracle λ vs κ: upper end +0.02 | 0.02 | 0.0173 | OK | analysis_oracle_vs_kappa.json (4 layerwise fits) |
| 130 | §1 l.30 | Within-layer ρ(κ, σ_min) = −0.91 (full-FT) | -0.91 | -0.9061 | OK | analysis_grad_vs_kappa.json |
| 131 | §1 l.30 | Within-layer ρ(κ, σ_min) = −1.00 (LoRA) | -1.00 | -0.9995 | OK | results/aa/lora/lora_stats.csv (outside the scope list; mean over 48 maps of Spearman(kappa2_W, smin_W)) |
| 132 | §1 l.31 | λ_expert before (SVHN, b=4) | 0.13 | 0.1338 | OK | tail_stress_svhn.json |
| 133 | §1 l.31 | λ_expert after, lower end (SVHN b=4) | 0.45 | 0.4488 | OK | tail_stress_svhn.json |
| 134 | §1 l.31 | λ_expert after, upper end (SVHN b=inf) | 0.69 | 0.6849 | ROUNDING | tail_stress_svhn.json -- 0.6849 = 50/73 |
| 135 | §1 l.31 | costs up to 8 pt | 8 | 8.31 | OK | tail_stress_*.json (largest drop of the 8-task mean, alpha 1.6) |
| 136 | §1 l.32 | full-FT oracle gain +8 pt dev | 8 | 8.37 | OK | layerwise_alpha_seed0 vs uniform grid |
| 137 | §1 l.32 | full-FT oracle test 77.2 | 77.2 | 77.234 | OK | layerwise_alpha_seed0.json |
| 138 | §1 l.32 | LoRA oracle gain +12 pt (dev) | 12 | 12.54 | ROUNDING | layerwise_alpha_seed0 vs uniform grid -- test gain is +12.86 pt |
| 139 | §1 l.32 | LoRA oracle test 64.1 | 64.1 | 64.078 | OK | layerwise_alpha_seed0.json |
| 140 | §1 l.32 | global per-expert weights ≤ +0.4 pt (DEV gain, both banks, seed 0) | <= 0.4 | 0.392 | OK | global_alpha_seed0.json vs uniform grid -- dev gains full-FT +0.32, LoRA +0.39 |
| 141 | §1 l.32 | same claim on TEST (global_alpha, seeds, both banks) | <= 0.4 | 1.136 | MISMATCH | global_alpha_seed*.json vs frozen-test uniform -- gains on test: full-FT +0.48, +0.51; LoRA +1.14; the claim does not name a split: true on dev, false on test |
| 142 | §1 l.28 | master: dev best 77.74 (kappa\|tuned) | 77.74 | 77.7359 | OK | results/freeze/test.jsonl dev_metric (master's file, outside scope) |
| 143 | §1 l.28 | master: uniform dev 77.65 | 77.65 | 77.6514 | OK | results/freeze/test.jsonl dev_metric |
| 144 | §1 l.28 | master: frozen TEST kappa 76.61 (normalized) | 76.61 | 76.6055 | OK | results/freeze/test.jsonl mean_norm |
| 145 | §1 l.28 | master: frozen TEST uniform 76.68 (normalized) | 76.68 | 76.6841 | OK | results/freeze/test.jsonl mean_norm |
| 146 | §1 l.34 | update-spread rule reaches the oracle with norms matched (+9 pt over uniform) | 9 | 9.07 | OK | A-nm: dsrank b=4 minus uniform |
| 147 | §2 l.41 | log-λ Spearman between seeds (layerwise_alpha) | 0.98 | 0.9843 | OK | oracle_lambda/*_lambda.csv (= analysis_oracle_vs_kappa.json) |
| 148 | §2 l.41 | log-λ Spearman between seeds (layerwise_a1) | 0.98 | 0.9796 | OK | oracle_lambda/*_lambda.csv (= analysis_oracle_vs_kappa.json) |
| 149 | §3 l.105 | attention maps nearly one expert per map (n_eff 1–2.5) | 1–2.5 | per-map n_eff in attention maps: 1.02 to 3.11 (max per fit 2.78, 2.40, 3.01, 3.11); type means 1.34-1.84 | IMPRECISE | oracle_lambda/*_lambda.csv (B3e, B3f) -- the type means are within 1-2.5; the largest single-map n_eff is above 2.5 |
| 150 | §3 l.105 | 'with the same assignment across seeds' (attention) | same assignment | same top expert in 45/48 (layerwise_alpha) and 40/48 (layerwise_a1) attention maps; MLP 20/24 and 23/24 | OK (approximately: 83-94% of attention maps) | B3d |
| 151 | §3 l.108 | LoRA within-map ρ(kappa, oracle) layerwise_a1_seed0 | 0.03 | 0.0342 | OK | results/aa/lora/oracle_vs_scores.json |
| 152 | §3 l.108 | LoRA within-map ρ(kappa, oracle) layerwise_alpha_seed0 | 0.03 | 0.0248 | ROUNDING | results/aa/lora/oracle_vs_scores.json |
| 153 | §3 l.94 | global oracle test 69.8 | 69.8 | 69.772 | OK | global_alpha_seed0.json |
| 154 | §3 l.94 | tuned uniform test 69.3 | 69.3 | 69.294 | OK | results/freeze/test.jsonl (cfg.tag = uniform\|tuned) |
| 155 | §3 l.94 | no global rule can gain more than about +0.5 | 0.5 | 0.477 | OK | global_alpha_seed0.json vs frozen-test uniform |
| 156 | §4 l.128 | κ falls with noise in about 70% of maps for all experts | about 70% | mean 70.9%, range over experts 61.6-79.5% (A-strong); first run (mild noise) mean 73.5% | OK (mean 70.9%; individual experts 62-79%) | B5b |
| 157 | §4 l.128 | the literal rule's argmin selects the noisiest copy | the noisiest copy | A-strong, kappa b=inf: mean share of maps won by the noisiest copy 0.572 (per task 0.534-0.658), by the cleanest 0.128; chance 0.250 | OK (majority of maps: the noisiest copy wins 53-66% of the maps per task vs 25% by chance) | partA_controlled_noise_strong_v2.jsonl (B5f1) |
| 158 | §4 l.132 | tuning-free GD/MP rule is too flat when signal scales differ (A-nm) | too flat | A-nm weights on the 4 copies (cleanest to noisiest): mp_invvar 0.339, 0.223, 0.219, 0.219 vs oracle_invvar 0.924, 0.058, 0.014, 0.004 | OK | partA_controlled_noise_strong_normmatched.jsonl (B5f2) |
| 159 | §4 l.134 | quality correlates with update spread at +0.5 (mean over tasks) | 0.5 | 0.487 | OK | partB_meta_v2.json |
| 160 | §4 l.134 | quality vs mean log κ: lowest −0.31 | -0.31 | -0.3077 | OK | partB_meta_v2.json |
| 161 | §4 l.134 | quality vs mean log κ: highest +0.60 | 0.60 | 0.5967 | OK | partB_meta_v2.json |
| 162 | §4 l.136 | uniform soup is already near the D-oracle (+0.6) | 0.6 | 0.533 | ROUNDING | B-v2: oracle_quality_softmax100 minus uniform -- the report subtracts the rounded 91.6 - 91.0 |
| 163 | §4 l.133 | mp_invvar on B: 90.1 | 90.1 | 90.105 | OK | partB_lora_variants_v2.jsonl |
| 164 | §4 l.133 | mp_invvar on B: 0/8 tasks above uniform | 0/8 | 0/8 | OK | partB_lora_variants_v2.jsonl |
| 165 | §4 l.147 | Frozen TEST: 'This replicates dev exactly.' | exact replication | test minus dev mean accuracy for the 15 frozen configurations shown in the report: -0.72 to -0.26 pt; win counts that change: A kappa b=1: dev 3/8 -> test 0/8; A kappa b=4: dev 1/8 -> test 0/8; A dsrank b=4: dev 8/8 -> test 7/8; A mp_invvar: dev 8/8 -> test 7/8; A-nm kappa b=1: dev 3/8 -> test 4/8; A-nm kappa b=4: dev 3/8 -> test 2/8; B kappa b=1: dev 4/8 -> test 1/8; B kappa b=4: dev 2/8 -> test 1/8 | IMPRECISE (qualitative pattern replicates, numbers do not match exactly) | B5g (dev and test side by side) |
| 166 | §5 l.151 | edit is 0.47% of ‖Δ‖ (median) -- SVHN | 0.47 | 0.4684 | OK | tail_stress_svhn.json -- SUN397 edit: 0.35% |
| 167 | §5 l.151 | expert accuracy unchanged (≤ 0.08 pt on every task) | <= 0.08 | 0.0792 | OK | tail_stress_*.json expert_alone_before/after -- largest change 0.079 pt |
| 168 | math note l.76 | layer-specific association +0.095 | 0.095 | 0.09536 | OK | decompose_pref.json |
| 169 | math note l.76 | its p 0.017 (one-sided, null >= obs) | 0.017 | 0.01660 | OK | decompose_pref.json p_within_ge (the report/§2 quotes the two-sided 0.03) |
| 170 | math note l.18 | within-layer Spearman(κ_i, σ_q(W_i)) = −0.91 | -0.91 | -0.9061 | OK | analysis_grad_vs_kappa.json |
| 171 | math note l.52 | 100% of its first-order effect on dev loss comes from 5 maps | 100% | 99.54% (alpha 1), 99.16% (alpha 2.4) | ROUNDING | analysis_grad_vs_kappa.json frac_dL_from_top5 |
| 172 | math note l.95 | 128x128 matrix: κ 6.7e6 to 1.5e3 as noise goes 1e-6 to 1e-2 | 6.7e6 -> 1.5e3 | 6.71e+06 -> 1.53e+03 | OK | synthetic_checks.json C11 |
| 173 | math note l.97 | L0 k_proj 1.04e8 -> 9.9e3; L6 fc1 14.4 -> 7.5 | 1.04e8 -> 9.9e3; 14.4 -> 7.5 | 1.04e+08 -> 9.87e+03; 14.4 -> 7.55 | OK | synthetic_checks.json C11_real_base |
| 174 | math note l.99 | noisy copies: κ falls with the noise level in 75% of the maps | 75% | mean over experts 73.5% (mild noise run), 70.9% (strong run); max over experts 75.3% / 79.5% | IMPRECISE (75% is the maximum over experts of the mild-noise run) | partA_meta*.json (B5b) |
| 175 | math note l.111 | stable rank of the updates tracks the noise level with rank correlation +1.00 | +1.00 | +1.000 (mild-noise run), +0.975 (strong run, mean over experts) | OK (mild-noise run; 0.94-0.99 in the strong run) | partA_meta*.json (B5b) |
| 176 | math note l.100 | the literal rule's global variant puts 0.73-1.0 on the noisiest copy | 0.73-1.0 | global:logkappa weight on the noisiest copy (A-strong, mean over tasks): b=1: 0.62; b=2: 0.82; b=4: 0.92; b=8: 0.96; b=16: 0.99; b=128: 1.00 | DEPENDS ON b (not reproducible as one range) | partA_controlled_noise_strong_v2.jsonl (B5f1) |

**Legend.**
- status OK = the recomputed value rounds to the reported number at the reported precision. ROUNDING = differs by one unit of the last reported digit (the report rounds a rounded number, or uses another seed). MISMATCH = larger difference. IMPRECISE / NOT UNIQUE / DEPENDS ON b = the statement is qualitatively right but its stated range or exclusivity does not match the data.
- Where several candidates exist (seeds, rules at the same b) all are listed; the first (seed 0 / the named rule) is the primary comparison. Test accuracy of tuned uniform comes from the frozen-test files named in the source column.
- Status counts: OK: 159, ROUNDING: 8, IMPRECISE: 4, MISMATCH: 2, PARTIAL: 1, NOT UNIQUE: 1, DEPENDS ON b: 1

### X2. Discrepancies between the report and the files (all non-OK rows of X1, plus omissions found while building B1-B7)

**What varies / split / metric / units.** Collects the rows of X1 whose status is not OK and notes on results that exist in the files but are absent from or contradicted by the report; numbers as in X1 (accuracy in %, correlations dimensionless).

**Source.** `docs/aa/AA_SIDE_REPORT.md`; `docs/aa/MATH_kappa_of_finetuned_weights.md`; `tables B1-B7 of this document`

| # | kind | report / file location | claim or note | reported | recomputed / found | where |
| ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | MISMATCH | §4 table l.124, B, 'update norm, b = 4', dnorm b=4 | ≈ 91.0 | 91.0 | 88.242 | X1 #43; status text: MISMATCH; source: partB_lora_variants_v2.jsonl -- B cell is stated as '≈ 91.0' |
| 2 | MISMATCH | §1 l.32 | same claim on TEST (global_alpha, seeds, both banks) | <= 0.4 | 1.136 | X1 #141; status text: MISMATCH; source: global_alpha_seed*.json vs frozen-test uniform -- gains on test: full-FT +0.48, +0.51; LoRA +1.14; the claim does not name a split: true on dev, false on test |
| 3 | PARTIAL | §2 table l.50, 'functional energy tr(ΔCΔᵀ) (bu', within-expert | +0.12 to +0.16 (p ≤ 0.007), the only per-layer predictor | +0.12 to +0.16 (p <= 0.007) | gradpref_a1: +0.126 (p 0.005); gradpref_a2.4: +0.119 (p 0.007); oracle_layerwise_alpha: +0.077 (p 0.084); oracle_layerwise_a1: +0.164 (p 0.000) | X1 #109; status text: PARTIAL: holds for gradpref_a1, gradpref_a2.4, oracle_layerwise_a1; not for oracle_layerwise_alpha; source: score_vs_targets.json B:fnorm_D -- the stated range/p-bound is checked per target |
| 4 | NOT UNIQUE | §2 table l.50, 'functional energy tr(ΔCΔᵀ) (bu', 'the only per-layer predictor' | +0.12 to +0.16 (p ≤ 0.007), the only per-layer predictor | only functional energy | also p<0.05 vs gradient pref: gradpref_a1 A:fro_D +0.099 (p 0.024); gradpref_a1 A:smin_W -0.090 (p 0.041); gradpref_a2.4 A:fro_D +0.123 (p 0.003); gradpref_a2.4 A:smin_W -0.128 (p 0.004) \| significant within-expert vs oracle targets: oracle_layerwise_a1 B:fnorm_D, oracle_layerwise_a1 B:rel_fnorm_D | X1 #110; status text: NOT UNIQUE (budget-A scores also significant); source: score_vs_targets.json -- A:fro_D (update norm, weights only) and A:smin_W (negative sign) are significant within-expert for the gradient-preference targets |
| 5 | IMPRECISE | §2 table l.49, 'update spread (stable / entrop', between-expert | ≈ 0; all of it is between-expert (expert identity, ±0.8) | ±0.8 | abs(between-expert) from 0.54 to 0.85 over 4 targets x 2 scores | X1 #105; status text: IMPRECISE; source: score_vs_targets.json -- per target (min..max over the two scores): gradpref_a1 0.85..0.85; gradpref_a2.4 0.62..0.68; oracle_layerwise_alpha 0.54..0.60; oracle_layerwise_a1 0.82..0.84 |
| 6 | IMPRECISE | §3 l.105 | attention maps nearly one expert per map (n_eff 1–2.5) | 1–2.5 | per-map n_eff in attention maps: 1.02 to 3.11 (max per fit 2.78, 2.40, 3.01, 3.11); type means 1.34-1.84 | X1 #149; status text: IMPRECISE; source: oracle_lambda/*_lambda.csv (B3e, B3f) -- the type means are within 1-2.5; the largest single-map n_eff is above 2.5 |
| 7 | IMPRECISE | §4 l.147 | Frozen TEST: 'This replicates dev exactly.' | exact replication | test minus dev mean accuracy for the 15 frozen configurations shown in the report: -0.72 to -0.26 pt; win counts that change: A kappa b=1: dev 3/8 -> test 0/8; A kappa b=4: dev 1/8 -> test 0/8; A dsrank b=4: dev 8/8 -> test 7/8; A mp_invvar: dev 8/8 -> test 7/8; A-nm kappa b=1: dev 3/8 -> test 4/8; A-nm kappa b=4: dev 3/8 -> test 2/8; B kappa b=1: dev 4/8 -> test 1/8; B kappa b=4: dev 2/8 -> test 1/8 | X1 #165; status text: IMPRECISE (qualitative pattern replicates, numbers do not match exactly); source: B5g (dev and test side by side) |
| 8 | IMPRECISE | math note l.99 | noisy copies: κ falls with the noise level in 75% of the maps | 75% | mean over experts 73.5% (mild noise run), 70.9% (strong run); max over experts 75.3% / 79.5% | X1 #174; status text: IMPRECISE (75% is the maximum over experts of the mild-noise run); source: partA_meta*.json (B5b) |
| 9 | DEPENDS ON b | math note l.100 | the literal rule's global variant puts 0.73-1.0 on the noisiest copy | 0.73-1.0 | global:logkappa weight on the noisiest copy (A-strong, mean over tasks): b=1: 0.62; b=2: 0.82; b=4: 0.92; b=8: 0.96; b=16: 0.99; b=128: 1.00 | X1 #176; status text: DEPENDS ON b (not reproducible as one range); source: partA_controlled_noise_strong_v2.jsonl (B5f1) |
| 10 | ROUNDING | §3 table l.102, full-FT, layerwise oracle + alpha, dev | 78.7 / 77.2 | 78.7 | 78.646 | X1 #6; status text: ROUNDING (OK for seed 1, 2-seed mean); source: layerwise_alpha_seed0.json (dev) -- ; other candidates: seed 1 78.675, 2-seed mean 78.660 |
| 11 | ROUNDING | §5 table l.158, SVHN, b = ∞, lambda after | 0.22 → 0.69 | 0.69 | 0.6849 | X1 #80; status text: ROUNDING; source: tail_stress_*.json -- = 50/73 |
| 12 | ROUNDING | §2 table l.49, 'update spread (stable / entrop', entropy rank vs gradient pref a=1 | −0.43 / −0.48 | -0.48 | -0.4746 | X1 #102; status text: ROUNDING; source: score_vs_targets.json (analysis_grad_vs_kappa.json: -0.4746) |
| 13 | ROUNDING | §1 l.31 | λ_expert after, upper end (SVHN b=inf) | 0.69 | 0.6849 | X1 #134; status text: ROUNDING; source: tail_stress_svhn.json -- 0.6849 = 50/73 |
| 14 | ROUNDING | §1 l.32 | LoRA oracle gain +12 pt (dev) | 12 | 12.54 | X1 #138; status text: ROUNDING; source: layerwise_alpha_seed0 vs uniform grid -- test gain is +12.86 pt |
| 15 | ROUNDING | §3 l.108 | LoRA within-map ρ(kappa, oracle) layerwise_alpha_seed0 | 0.03 | 0.0248 | X1 #152; status text: ROUNDING; source: results/aa/lora/oracle_vs_scores.json |
| 16 | ROUNDING | §4 l.136 | uniform soup is already near the D-oracle (+0.6) | 0.6 | 0.533 | X1 #162; status text: ROUNDING; source: B-v2: oracle_quality_softmax100 minus uniform -- the report subtracts the rounded 91.6 - 91.0 |
| 17 | ROUNDING | math note l.52 | 100% of its first-order effect on dev loss comes from 5 maps | 100% | 99.54% (alpha 1), 99.16% (alpha 2.4) | X1 #171; status text: ROUNDING; source: analysis_grad_vs_kappa.json frac_dL_from_top5 |
| 18 | OMISSION / NOTE | §4 (A-nm column) | reversed kappa rule (kappa_rev, hard selection b=inf = highest-kappa copy) is not discussed | – | A-nm: kappa_rev b=inf 86.99 vs uniform 81.27 (8/8 tasks above uniform); A-strong: 89.82 vs 89.76 (6/8); B: 88.00 vs 91.04 (0/8) | B5d3, B5e3 |
| 19 | OMISSION / NOTE | §5 | tail-stress rows at alpha = 1.0 (b = 4) are in the JSON files but not in the report table | – | svhn: 67.33 -> 67.48 (+0.15 pt), lambda 0.134 -> 0.449; sun397: 67.33 -> 65.49 (-1.84 pt), lambda 0.113 -> 0.396 (the SVHN edit does not lower the 8-task accuracy at alpha = 1) | B6c, B6d |
| 20 | OMISSION / NOTE | §2 table row 'κ(W C^{1/2}) (budget B)' | the budget-B kappa statistic contains round-off values | – | 25 of 584 (map, expert) rows have kappa_WC > 1e12 (maps: L0.k_proj, L0.v_proj, L0.q_proj, L1.k_proj); the within-map ranks of those maps are determined by round-off | B7e |
| 21 | OMISSION / NOTE | results/aa/soup/partB_lora_variants.jsonl, summary.md | the first B run is incomplete and still listed in summary.md/json as '(2 tasks)' | – | 2 tasks present; 112 rows; superseded by partB_lora_variants_v2.jsonl (identical on the common rows) | B5a |
| 22 | OMISSION / NOTE | results/aa/oracle_lambda/analysis_oracle_vs_kappa.json | contains only the 4 layerwise fits and a seed_agreement block for the layerwise families | – | no entry for global_alpha_seed0/1 (their lambda CSVs and JSONs exist); the global fits are summarised from the CSVs in B3d/B3g | B3d, B3g, B3h |
| 23 | OMISSION / NOTE | §4 'update norm, b = 4' (A-nm: 78.9) | dnorm in the norm-matched pool is a round-off-driven pseudo-random choice, not a signal | – | max deviation of the mean lambda of dnorm from uniform: 0.124 in A-nm vs 0.750 in A-strong; accuracy at b=4 is 78.89 (< uniform 81.27) | B5a legend, B5d3, B5f2 |

**Legend.**
- kinds: MISMATCH = the reported number is not supported (beyond rounding); ROUNDING = off by one unit of the last digit (rounded-number arithmetic, another seed, or the report rounding up a 3-decimal value); IMPRECISE = the statement is qualitatively right but its range/wording is looser than the data; PARTIAL / NOT UNIQUE / DEPENDS ON b = holds only for some targets / not exclusively / only for some temperatures; OMISSION / NOTE = a result in the files that the report does not mention.
- rows 'DIFFERS' would mark a p-value of the section 2b re-derivation that deviates by more than 0.03 from the report (none unless listed).

### X3. Claims of the report that cannot be verified from the files in this scope

**What varies / split / metric / units.** Lists the report statements for which no result file or script exists in the scope (or which are outside it); no numbers are produced.

**Source.** `docs/aa/AA_SIDE_REPORT.md`

| location | statement | why not verified |
| :--- | :--- | :--- |
| §2b | log kappa_q90 rows (+0.076 (0.09), ...) | no producing script/result file in the aa track; re-derived in B4b from master's results/spectra/stats.csv (see the §2b rows of this table) |
| §1 | shuffled-score null 77.74 ± 0.19 | produced by master's shuffle sweeps (results/sweeps/*), outside the scope of this table set |
| §2 | Two independent κ implementations agree to 1e-13 median | no result file or script in the aa results; not checked |
| §6, §9 | LoRA selection/replication numbers and the RegMean results | outside this table set (results/aa/lora/*); only the LoRA oracle rows and the LoRA rho(kappa, sigma_min) were checked here |

## Notes on parsing and provenance

- Every file listed in the task scope was parsed (JSON, JSONL, CSV, NPZ, MD); no parsing failure. `results/aa/figures/*.png` are outside the scope and were not read.
- `synthetic_checks.json` stores numbers only for C1, C2, C3, C5, C6, C11, C11_real_base and `fails: []`; the PASS/FAIL line of each sub-check and the numbers of C4, C7-C10 come from re-running a copy of `scripts/aa_synthetic_checks.py` in the scratchpad (seed 0; its JSON equals the stored one, max relative deviation 0.0e+00). `C11_real_base` could not be re-run (needs a base-weight npz).
- `oracle_lambda/analysis_oracle_vs_kappa.json` has no entry for the global_alpha fits; those are summarised from the lambda CSVs (B3d, B3e, B3g). `oracle_lambda_lora/` has no analysis file: the LoRA oracle correlations are in `results/aa/lora/oracle_vs_scores.json` (B3i), outside the scope list but the only place they are recorded.
- `soup/partA_meta.json` (first A run) has no 'levels' key: the noise levels were recovered from the oracle_invvar weights and the script default (B5a). `soup/test_freeze_summary.json` is not written by any script of `scripts/`; it is reproduced from `test_freeze.jsonl` (all 25 configurations equal).
- `score_vs_targets.md` equals `score_vs_targets.json` rendered in the script's format: True.
- The cross-check (X1-X3) parses `docs/aa/AA_SIDE_REPORT.md` as it was at generation time (sha256 228884418c25056b..., modified 22:50) and `docs/aa/MATH_kappa_of_finetuned_weights.md` (sha256 4d84c114e35d0a91...); the report is being edited by another session (a paragraph was appended to section 9 while this was built; sections 0-5 and 7 are unchanged), so re-run the script to refresh X1-X3.
- Files read outside the scope list, only for cross-checks: `results/freeze/test.jsonl` (tuned-uniform test accuracy, master's numbers), `results/aa/lora/test_freeze.jsonl`, `results/aa/lora/lora_stats.csv`, `results/aa/lora/oracle_vs_scores.json`, `results/spectra/stats.csv` (kappa_q90 for B4b), `results/sanity/expert_matrix_dev.json` (dev sizes). Nothing in the worktree was modified.
