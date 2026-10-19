# Protocol 02: the professor's rule on LLM banks, reproduction audit, inverse-solve follow-ups

*Written 2026-10-19, before any LLM merge was evaluated. The only earlier LLM evaluations in this project were
plumbing checks of the new evaluation path (`experiments/llm/pilot_*.py`): Task Arithmetic at the source
project's own selected alpha on val, and one expert on its own task. Changes after this point are appended
as dated amendments at the end of this file; nothing above the amendments is edited.*

Labels used in reports: **[established]** = run completed, numbers from records; **[tentative]** = pattern
without its settling test; **[deduction]** = mathematical argument; **[assessment]** = judgement.

## 0. Ownership, locations, integrity

* One Claude session (azpitarte-c6) owns the repository `~/condmerge` (branch master). Remote:
  `github.com/GabrielAzpitarte/Model_Merging` (push authorised by the user on 2026-10-19). Commits are authored by
  Gabriel Azpitarte <gabriel.azpitarte@icloud.com>.
* New code lives in new files only:
  * `src/condmerge/llm/*`, `experiments/llm/*`, `experiments/repro/*`, `experiments/inverse/*` and
    `experiments/uncert/*`;
  * `vendor/llama31`, `vendor/llama32`: byte-identical snapshots of the two source evaluators, with their
    sha256 recorded in each bank manifest.
  * The Report 1/2 code is not edited. Any bug fix there is a new commit and is logged.
* Outputs:
  * LLM records go to `results/llm/<bank>_T<T>/<group>/<worker>.jsonl`, one append-only shard per worker
    process, so no file has two writers.
  * Predictions go to `artifacts/llm/preds/`, spectra to `artifacts/llm/spectra/`.
  * CLIP follow-ups go to `results/repro/`, `results/inverse/` and `results/uncert/`.
* Dispatcher integrity (the earlier report had a bug in which a method silently ran as uniform):
  * every LLM merge record stores the coefficient matrix actually requested (maps x experts), alpha, the
    rule's special-case log, per-map merged-update norms, and a fingerprint (fixed random probes u^T dW v
    summed over maps);
  * the fingerprint is computed both from the request and from the GPU buffers actually installed, and the
    two must agree or the job fails;
  * any non-uniform rule whose coefficients come out exactly uniform is flagged;
  * the CLIP reproduction records per-map coefficients and merged-parameter checksums in the same way.
* The frozen test configurations are written to `configs/llm/frozen_*.json` and committed **before** any test
  or fresh evaluation is enqueued.

## 1. LLM banks (verified from files, `results/llm/manifest_*.json`)

| setting | base (revision) | experts | answers / evaluator | role |
|---|---|---|---|---|
| S1 `l31_8b` T8 | meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f2 | banking77, clinc, ledgar, massive, mnli, anli, mrpc, qqp | multi-token, exact sum log-prob argmax with certificate | **primary** |
| S2 `l32_3b` T8 | unsloth/Llama-3.2-3B-Instruct @ 006f5dcd (weights identical to meta-llama per its project) | mnli, qnli, snli, rte, sick, scitail, mrpc, qqp | single-token next-token argmax | second independent bank: transfer AND own-dev replication |
| S3 `l32_3b` T20 | same as S2 | 20 tasks | same | more experts; own-dev replication |
| S4 `l31_8b` T20 (optional, lowest priority) | same as S1 | 20 tasks | same as S1 | transfer of S1's frozen configuration only |

* LoRA: PEFT r = 16, lora_alpha = 16, so the scale is 1 and is already absorbed into B. Targets are
  q/k/v/o_proj of every block. There is no bias, no modules_to_save, and nothing else is trained. The base is
  frozen in BF16 and the adapters are FP32.
* Effective expert matrix: W_i[p] = W0[p] + B_i[p] A_i[p], assembled in float64. Here W0 is the stored BF16
  tensor, promoted exactly.
* Merging is exact: W0 + alpha * sum_i lambda_i[p] B_i A_i via LoRA concatenation on an FP32 side path. This
  is the evaluators' own convention, and the factors are never averaged separately.
* Pilot check: the concatenation update equals the source CoreSpace update to 4e-8 relative.
  * The source path reproduces its own recorded val numbers exactly today.
  * The concatenation path differs from them by 0.2-0.35 pt.
  * This is evaluation numerics (BF16 rounding flips near-tie examples), so a **numerical-noise floor** is
    measured (sec. 4.6).

## 2. Splits and metrics

* **val**: the source projects' own selection split, 20 % of the evaluation source with seed-0 permutation,
  capped at 250 per task. All selection and all diagnostics use it.
* **test**: the source projects' reporting split, the next 80 % capped at 1000 per task.
  * Earlier projects used it to report other merging methods (TA, TIES, TSV, Iso-C, RegMean++, ...).
  * No condition-number rule has ever been evaluated on it.
  * It is "untouched by this study", not untouched in absolute terms.
* **fresh**: NEW. The examples after the test cap, capped at 1000 per task. No run of either project ever used
  them. They are available only where the evaluation source is large enough; per-task sizes are in the
  manifest. Fresh results are reported over the tasks that have them, as macro averages and per task.
* Disclosure: the alpha range below is informed by the source project's TA val curve (val only).
* **Primary metric:** mean normalised accuracy.
  * Per task: local accuracy (argmax over the task's own answers) divided by the task's own expert's local
    accuracy on the same split, then averaged over tasks.
  * It is the source projects' selection metric and the analogue of the CLIP full-FT metric.
* **Secondary metrics:** joint macro accuracy (argmax over the union of all tasks' answers, no task ID), local
  macro accuracy, and worst-task normalised accuracy.
* Units are always labelled. CLIP full-FT numbers are normalised; CLIP LoRA numbers are absolute accuracy.
  They are never compared without conversion.
* **Statistics:**
  * paired bootstrap over examples, stratified by task, 2000 resamples, from saved per-example predictions;
    95 % percentile CIs;
  * per-task differences and the worst-task change are reported;
  * shuffle positions are reported as ranks; no randomisation p-value is claimed below 19 permutations.

## 3. Methods (each separately named)

Definitions: `src/condmerge/llm/rules.py`. All rules use lambda[p] = softmax_i(-beta c_i[p]) with ONE beta
for all matrices, beta = b / scale, where scale = the median over matrices of the across-expert SD of c.

* scale is a single global rescaling: it only sets the units of b. It is not per-matrix normalisation.
* b = 0 takes an explicit uniform branch. b = inf is per-matrix argmin, with ties sharing the mass.
* A +inf score gets weight 0. An all-+inf matrix gets uniform weights (logged). NaN raises an error.
* The merge is W0 + alpha * sum_i lambda_i[p] Delta_i[p]. alpha = 1 is exactly the professor's convex
  combination of full expert matrices.

| name | score c_i[p] | status |
|---|---|---|
| `uniform` | (beta = 0) | control |
| `kappa` | sigma_max(W_i) / sigma_min(W_i) of the FULL expert matrix, exact fp64 | **the professor's hypothesis** |
| `kappa_rev` | -kappa | reversed control |
| `shuf<s>:kappa` | kappa permuted across experts within each matrix (seed s) | null with identical concentration per matrix |
| `global:kappa` | one score per expert: kappa_g,i = exp(mean_p log kappa_i[p]) (geometric mean over the adapted matrices), used at every matrix | the "whole-model" reading; a **convention**, not a canonical network condition number |
| `gshuf<s>:global:kappa` | global scores permuted across experts | its null |
| `logkappa` (R1) | log kappa (lambda ∝ kappa^-beta) | refinement 1, see sec. 6 |
| `kappa_res` (R2) | sigma_max / max(sigma_min, tau_bf16[p]) | refinement 2, see sec. 6 |
| `shuf<s>:logkappa`, `shuf<s>:kappa_res` | within-matrix shuffles | their nulls |
| `isoc` | alpha * mean(sigma) U V^T of sum_i Delta_i (Iso-C), alpha = the source project's val-selected value | strong weight-only context baseline, through the same exact path |
| experts, base | | references |

* RegMean++ (task IDs plus calibration activations) is quoted from the source project's records as context
  only. It uses a different information budget.
* Not substituted for kappa (these are different hypotheses): condition numbers of A or B, of the
  rank-deficient update BA, truncated or iterative estimates, robust or activation-weighted condition
  numbers.

## 4. Grids, selection, confirmation

1. **alpha grid** (every rule, uniform included): {0.8, 1.0, 1.3, 1.6, 2.0}. It includes alpha = 1.
   * The source's TA val curve peaks at alpha = 1.6 in this convention (TA@0.2 x T = 8) and declines
     beyond 2.4.
   * Concentrated rules need smaller alpha, because the merged-update norm grows with concentration.
2. **b grid** (fixed from weights only before any evaluation):
   * argmin, plus 4 finite b whose mean n_eff over matrices is closest to K x {0.85, 0.65, 0.45, 0.25} on a
     fine log grid;
   * each rule gets its own b values;
   * shuffles use the b values of their base rule, so their per-matrix concentration is identical;
   * `kappa_rev` uses kappa's b values (same units).
   * If a target is unreachable (plateaus), the nearest value is used and recorded.
   * Coefficient diagnostics per b go to `configs/llm/devgrid_*.json`: n_eff distribution, the fraction of
     matrices that are near-one-hot (max lambda > 0.99), and the fraction that are near-uniform.
   * If the grid gives only uniform-or-one-hot weights in almost every matrix, that is recorded as a property
     of the rule; it is not tuned away.
3. **Dev runs per setting:**
   * uniform x alpha;
   * kappa, kappa_rev, global, R1, R2: each its b grid x alpha;
   * shuffles: S1 has 5 kappa seeds and 2 seeds for each of global, R1 and R2; S2/S3 have 10 kappa seeds and
     3 for each of the others.
4. **Selection:** the best dev mean normalised accuracy, ties going to smaller b and then to alpha closer to
   1. Each shuffle seed is selected on its own (equal selection opportunity). Also frozen:
   * `uniform@alpha=1`;
   * `kappa@alpha=1` (best b at alpha 1);
   * the concentrated setting used by the tail edit (sec. 5).
5. **Confirmation:** every frozen configuration is evaluated on test and on fresh. Additional matched-config
   nulls: 15 more kappa shuffles (seeds 5-19), evaluated at kappa's frozen (b, alpha) on test.
6. **Numerical-noise floor:**
   * uniform at its frozen alpha with coefficient jitter of 1e-7 relative (3 seeds), on val and test;
   * one exact repeat (determinism check).
   * Differences below this floor are reported as indistinguishable.
7. **Second bank:**
   * (a) **transfer**: S1's frozen (b, alpha) values evaluated unchanged on S2 test and fresh;
   * (b) **replication**: S2's own dev grid and selection, then test and fresh.
   * These are separate experiments and are reported separately.
8. **Pre-registered support criterion for H-LLM1 (per setting):**
   * kappa-tuned minus uniform-tuned > 0 on test with the 95 % CI excluding 0;
   * kappa-tuned above all of its tuned shuffles on test;
   * a positive difference on fresh.
   * "Practically meaningful" means >= +0.5 normalised points.
   * Otherwise the verdict is NOT SUPPORTED. If additionally |difference| <= the noise floor or the CI
     contains 0, it is reported as null.

## 5. Mechanism tests on the LLM banks

* **Spectral diagnostics** (weights only):
  * per matrix: kappa0, the across-expert spread of kappa, and the share of log kappa variance from
    sigma_max vs sigma_min;
  * counts of singular values below the computational error bound and below the BF16 source floor tau_bf16;
  * an interlacing check (sigma_{j+r}(W0) <= sigma_j(W_i) <= sigma_{j-r}(W0));
  * concordance of expert orderings across matrices (Kendall's W);
  * within-matrix correlation of kappa with ||Delta||_F.
* **Validation of the spectral computation:** representative attention matrices (layers 0, 1, mid, last x
  q/k/v/o), base + 3 experts.
  * gesdd vs gesvd (fp64);
  * an independent sigma_min from inverse iteration on the QR factor R (triangular solves), with the residual
    ||W v - sigma u|| / sigma_1;
  * fp32 GPU singular values (to show what fp32 loses);
  * the Gram route eig(W^T W) (to show the squaring loss);
  * assembly in fp32 vs fp64.
* **Source-precision sensitivity:** 2 "re-rounding" replicates. Each draws W0 + E, with E uniform within +-ulp/2
  of each stored BF16 entry: an equally valid higher-precision preimage of the same checkpoint. Recompute
  every T8 expert's kappa for all matrices and report:
  * the Kendall tau of the expert ordering per matrix;
  * the change in lambda at the frozen b.
* **Tail-edit 2 x 2** (S1 and S2):
  * Edited expert: the expert with the smallest mean lambda under the frozen kappa configuration.
  * The edit: in every matrix where that expert's sigma_min(W_i) < tau_bf16[p], its singular values are lifted
    to max(sigma_j, tau_bf16[p]). This is a change within the checkpoint's own BF16 uncertainty.
  * Cells:

    | | original coefficients | coefficients recomputed from the edited matrices |
    |---|---|---|
    | original matrices | A | C |
    | edited matrices | B | D |

  * Plus uniform merging with original and with edited matrices.
  * Measurements:
    * the edited expert's own-task accuracy;
    * KL of its first-answer-token distribution over the bank's answers, and the relative change of
      last-prompt-token hidden states;
    * score and ranking changes, and coefficient changes;
    * merged accuracy in A-D.
  * Settings: the dev-selected (b, alpha) of `kappa`, and a concentrated setting (b with mean
    n_eff ~ 0.45 K at its dev-best alpha), the latter labelled a stress test.
* **Allocation usefulness** (budget D, labelled-data diagnostic):
  * g[p, i] = d(mean val loss)/d lambda_i[p] at the uniform-tuned merge (S1, S2);
  * the within-matrix Spearman between -g and each score's preference (-kappa, -log kappa, -kappa_res,
    -||Delta||).

## 6. Refinements (at most two; declared here, before results)

**R1 `logkappa`**
* Failure addressed: kappa spans orders of magnitude across matrices (pilot: k_proj ~1e3-1e4, layer-0 q_proj
  ~2e8). One beta then gives argmin selection in near-singular matrices and uniform weights elsewhere
  (MATH_NOTES N3).
* Why it should help: log puts the scores in relative units, so one beta gives comparable concentration
  everywhere.
* Assumption: relative kappa differences carry expert reliability.
* Falsified if it is not above tuned uniform and its own shuffles on test and fresh.
* Algebra: log kappa_i - log kappa0 (base) is the same softmax, since common shifts cancel. So R1 is also the
  "base-relative" rule. On CLIP, `logkappa` was null.

**R2 `kappa_res`**
* Failure addressed: sigma_min can lie below the BF16 resolution floor of the stored base (pilot: 3B layer-0
  k_proj sigma_min 4.6e-3 < tau 6.0e-3; 8B layer-0 q_proj 1.6e-7). There the literal rule's ranking reflects
  digits the checkpoint does not determine.
* Why it should help: flooring sigma_min at tau removes preferences that are not resolved by the stored
  information. Matrices with all experts below the floor become uniform.
* Assumption: the resolved part of the spectrum carries reliability information.
* Falsified as for R1.
* Not equivalent to kappa when some experts are above tau and others below.

Both use the same grid rules, shuffles, freeze and confirmation as kappa. A positive finding needs fresh
confirmation AND replication on S2. If neither beats uniform and its shuffles, this branch is closed (no
further search over spectral statistics).

## 7. CLIP reproduction audit (sec. 2 of the brief)

All on frozen configurations from the records, with per-map coefficients and merged-parameter checksums logged.

| item | records reproduced | tolerance |
|---|---|---|
| B/32 test: uniform-tuned (alpha 1.6), uniform alpha 1, kappa-tuned (b 0.25, alpha 1.6), kappa at alpha 1 (b 4), kappa_rev, 5 shuffles | results/freeze/test.jsonl | <= 0.05 pt (fp16 nondeterminism) |
| full-FT RegMean budget C, n = 16 (3 subsets) and 512: kappa-cap ridge vs tuned fixed ridge vs tuned shrinkage | results/regmean/strength_sweep_C_test_*.jsonl | <= 0.05 pt |
| LoRA RegMean frozen test, kappa-cap 1e4 vs shrink 0.95 (KnOTS, lora_c5a, vit20) | results/aa/lora/regmean_test.json | <= 0.05 pt |

## 8. Inverse-solve lead (separate extension)

**H-INV0 (audit of the ~4.1 % collapses, LoRA RegMean):** for the collapsed configurations (KnOTS N 512
alpha 1.3, plain; N 16 plain), check:
* finiteness of the merged parameters and of the activations / logits in fp16 and fp32 evaluation;
* the prediction-class distribution;
* Gram symmetry and positive semi-definiteness;
* solver residuals ||D G - rhs|| / ||rhs||;
* fp32 vs fp64 solves, Cholesky vs LU vs explicit inverse vs pseudo-inverse.

Verdict categories:
* numerical failure: non-finite values, overflow, or a solver that fails its residual;
* statistical overfitting: finite, accurate solve whose correction is dominated by noise directions.

**H-INV1 (LoRA, equal budgets):**
* Methods: kappa-cap ridge (kappa_t in {1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5}); a fixed relative ridge
  rho = c tr(G)/d (c in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}); and RegMean shrinkage a in
  {0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1}.
* Each method gets the same 7 strengths x alpha {0.8, 1.0, 1.3}, the same calibration (budget C: N 512, and
  N 16 with 3 subsets), the same solve (full input space) and ridge target zero.
* Banks: KnOTS, lora_c5a, vit20.
* Dev selection, then frozen test with per-image predictions and a paired bootstrap.
* A rank-16 truncation of every method's update is reported separately.
* Criterion: kappa-cap minus tuned fixed ridge > 0 with the CI excluding 0 in >= 2 of 3 banks. Otherwise
  "no practical value beyond tuned ridge".

## 9. Uncertainty diagnostic (sec. 9 of the brief), CLIP full FT B/32, budgets C and B

The correction is F = B G^-1, with B = sum_i (D_i - R) G_i, G = sum_i G_i, R = the uniform average, for each of
the 73 linear maps. Data:
* a reference set of 1024 train images per task, for the large-calibration reference F_ref;
* disjoint small subsets of n in {16, 64} images per task, with 10 subsets each;
* a disjoint set of 256 images per task, whose activations measure error.

**Error measure:** e = ||(F_hat - F_ref) X_ind||_F^2 / ||X_ind||_F^2, at the map's own expert-specific inputs.

**Predictors, all computed from the small subset only (no reference leakage):**
* u_boot: a bootstrap over whole images (B = 32), recomputing B and G jointly, of
  E||F_b - F_hat||^2 measured in the same norm;
* kappa(G_hat);
* 1 / lambda_min(G_hat);
* tr(G_hat^-1);
* ||F_hat||.

**Success:** u_boot's Spearman with e across maps and subsets is >= 0.5, and exceeds every other predictor's by
>= 0.1, in both budgets. Its calibration (e / u_boot) is reported.

**Only if this succeeds:**
1. A scalar gate per map, D = R + gamma_p F_hat with gamma_p = 1 / (1 + c u_boot / ||F_hat||^2), and one global c
   tuned on dev. It is compared with tuned ridge, the kappa-cap and plain RegMean downstream (dev selection,
   frozen test), for each budget separately.
2. Directional shrinkage only if the scalar gate helps.

Budget B with identical Grams for all experts makes B = 0, so the correction vanishes. This is stated as a
deduction.

## 10. Stopping rules

* The H-LLM1 verdict per setting follows sec. 4.8.
* Refinements follow sec. 6. There is no further spectral-statistic search.
* The inverse-solve lead is closed if H-INV1 fails.
* The uncertainty method is not built if the diagnostic in sec. 9 fails. In that case the recommendation
  cannot be "develop uncertainty gating".
* The professor update is drafted only after the S1/S2 test and fresh results and the tail edit are complete.

## Amendments

(none yet)

### Amendment 1 (2026-10-19 06:05, before H-INV1 ran)

The H-INV0 audit (`experiments/inverse/collapse_audit.py`, results/inverse/collapse_audit.json) found the ~4.1 %
LoRA RegMean "collapses" are an **fp16 evaluation overflow**:
* the unregularised solve has weights up to 6e5 (base max 0.27) in directions the calibration Grams barely span;
* fp16 autocast produces non-finite features;
* the same weights evaluated in fp32 reach about the kappa-capped solve's accuracy (KnOTS N = 16: fp32 62.29 vs
  kcap 62.30; lora_c5a N = 512: fp32 65.40 vs kcap 65.62).

Changes:
* H-INV1 (`lora_ridge_equal.py`) therefore selects and reports in **fp32**, and also reports the fp16 accuracy of
  every selected configuration. Otherwise plain RegMean would be eliminated by a numerical artefact rather than by
  its statistics.
* The same fp16/fp32 audit is added for full fine-tuning (`collapse_audit_fullft.py`).

### Amendment 2 (2026-10-19 06:35, before the diagnostic or any gate ran)

The gate stage (`experiments/uncert/gate.py`, run only if the sec. 9 criterion passes) gains a **shuffled-uncertainty
null** (`gate_shuf`): the same gate with u_rel permuted across maps. The prior-work check (docs/lit/PRIOR_WORK_02.md)
shows uncertainty-proportional shrinkage toward an anchor is a textbook mechanism. So a positive gate result must beat:
* the tuned fixed ridge and the kappa cap;
* its own shuffled-uncertainty null.

The prior-work check also notes that IterIS's relative ridge alpha ||G||_F I bounds kappa(G + rho I) <= (1 + alpha) / alpha.
In the near-singular regime the kappa cap is therefore a re-parametrisation of a known ridge; it is not reported as a
new method.
