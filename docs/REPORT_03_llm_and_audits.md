# Report 3: the condition-number rule on LLM banks, reproduction audit, inverse-solve and uncertainty follow-ups

*Status: PARTIAL. Sections marked **[pending]** wait for running experiments (docs/HANDOFF_20261019.md).*
*Protocol: docs/PROTOCOL_02_llm_followups.md (pre-registered 2026-10-19 before any LLM merge was evaluated; amendments
1-2 dated). Tables generated from the records: docs/REPORT_03_tables.md.*

Labels:
* **[established]**: completed run, numbers taken from the records;
* **[tentative]**: a pattern without its settling test;
* **[deduction]**: a mathematical argument;
* **[assessment]**: a judgement.

Units: **norm** = mean normalised accuracy in % (each task's accuracy divided by its own expert's accuracy on the same
split, averaged over tasks); **acc** = mean absolute accuracy in %; **joint** = LLM accuracy with every task's answers
competing (no task ID).

## 1. What was tested and on what [established]

The professor's rule, implemented exactly. For every adapted matrix p of every expert i:
* W_i[p] = W0[p] + B_i[p] A_i[p], assembled in float64 from the stored BF16 base and the FP32 LoRA factors;
* kappa_i[p] = sigma_max / sigma_min of that FULL matrix, from an exact float64 SVD;
* lambda_i[p] = softmax_i(-beta kappa_i[p]), with one beta for all matrices;
* W_merge = W0 + alpha sum_i lambda_i[p] Delta_i[p]. alpha = 1 is the professor's convex combination of the full
  expert matrices.

Merges are exact: the LoRA factors are concatenated on the evaluators' own FP32 side path, never averaged separately.

LLM banks were verified from files (manifests in results/llm/manifest_*.json):

| setting | base | experts | role |
|---|---|---|---|
| S1 | Llama-3.1-8B-Instruct @ 0e9e39f2 (BF16) | 8: banking77, clinc, ledgar, massive, mnli, anli, mrpc, qqp | primary |
| S2 | Llama-3.2-3B-Instruct @ 006f5dcd | 8: mnli, qnli, snli, rte, sick, scitail, mrpc, qqp | independent bank: transfer + replication |
| S3 | same 3B base | 20 tasks | more experts: replication |
| S4 | same 8B base | 20 tasks | optional transfer |

Expert recipe, confirmed from the training scripts and json files:
* PEFT LoRA r = 16, alpha = 16, so the scale is 1 and is already inside B;
* targets are q/k/v/o_proj; there are no biases and no other trained parameters;
* the base is frozen in BF16.

Evaluation is the source projects' own code (vendored, sha256 in the manifests):
* S1: exact multi-token sum-log-probability argmax, with a certificate;
* S2/S3: single-token argmax.

Pipeline checks:
* the concatenation update equals the source CoreSpace update to 4e-8 relative;
* the experts' val/test accuracies reproduce the source records to within one example per task (8B: |diff| <= 0.4 pt);
* the source CoreSpace path reproduces its own recorded TA@0.2 exactly today.

Evaluation noise floor [established]:
* evaluation is deterministic: an exact repeat reproduces every record bit for bit;
* perturbing every coefficient by 1e-7 (relative) moves val mean normalised accuracy by up to 0.19 pt on S1 and up to
  0.33-0.35 pt on S2/S3 (3 jitter seeds; BF16 rounding flips near-tie examples);
* method differences of that size are not distinguishable from rounding (records: results/llm/*/noise/).

Splits:
* val: the source projects' selection split;
* test: their reporting split, which earlier projects used for OTHER methods but never for any kappa rule;
* fresh: NEW, the examples after the test cap, never used by anyone. Available for 7/8 S1 tasks and 4/8 S2 tasks
  (+43 SciTail).

## 2. Reproduction audit of the CLIP results [established]

| item | result |
|---|---|
| B/32 frozen test: uniform (tuned, alpha 1), literal kappa (tuned, alpha 1), kappa_rev, 5 shuffles | every test number reproduces exactly (max abs diff 0.000 pt), with 100 % per-image agreement against the original run's saved predictions |
| dispatcher check | the literal rule's coefficients depart from uniform by up to 0.875 in some maps, and its merged-update fingerprint differs from uniform's, so the method really ran. The original records also agree with each other only below 100 % across methods |
| full-FT RegMean, budget C (n = 16 x 3 subsets, n = 512): kappa cap vs tuned fixed ridge vs tuned shrinkage | all 12 frozen test numbers reproduce exactly; solve residuals <= 1e-14 |
| LoRA RegMean frozen test (3 banks; uniform, shrink 0.95, kappa cap 1e4) | reproduces within 0.02 pt (fp16 nondeterminism) |

Records: results/repro/clip_repro_{rules,regmean,lora}.json.

## 3. Numerical reliability of kappa on the LLM banks [established for S1; S2 pending]

**Computation is exact.** On 16 representative matrices (layers 0, 1, 16, 31 x q/k/v/o; base + 3 experts):
* LAPACK gesdd vs gesvd agree on sigma_min to a median 3e-14 (max 1.5e-10);
* an independent QR + inverse-iteration sigma_min agrees to a median 2e-13 (max 1e-10);
* smallest-triplet residuals are <= 3e-15 sigma_1.

Two alternatives are inadequate and were not used:
* fp32 SVD: sigma_min errors up to 3.5 %;
* the Gram route sqrt(eig(W^T W)), which squares kappa: errors up to 0.2 %.

Assembling W_i in fp32 instead of fp64 changes kappa by up to 6e-4. Interlacing (sigma_{j+16}(W0) <= sigma_j(W_i) <=
sigma_{j-16}(W0)) has 0 violations in all 128 x 8 matrices.

**But in q_proj and o_proj kappa is not determined by the checkpoint.** The base is stored in BF16. An equally valid
higher-precision preimage differs from it by a perturbation of spectral norm ~tau_bf16, the BF16 resolution floor,
estimated per matrix from the exact ulp of every entry. Results (S1, all 128 matrices, 8 experts):

| projection | kappa0 (median) | sigma_min / tau_bf16 (median) | across-expert spread of kappa / kappa0 | share of log-kappa variance from sigma_min |
|---|---|---|---|---|
| q_proj | 2.0e5 (2.8e4 - 4.5e8) | 0.014 | 477 % | 100 % |
| o_proj | 5.9e4 (2.0e4 - 5.1e6) | 0.027 | 619 % | 100 % |
| k_proj | 47 (33 - 7.1e3) | 41 | 0.8 % | 99.9 % |
| v_proj | 8.2 (5.5 - 31) | 100 | 0.2 % | 95 % |

Re-rounding test: two replicates of W0 + E, with E uniform within +-ulp/2 of each stored entry, so each rounds back to
the same checkpoint.

| projection | Kendall tau of expert ordering | same argmin expert | median max kappa change | matrices whose lambda changes > 0.5 at the n_eff ~ 0.45 K temperature |
|---|---|---|---|---|
| q_proj | 0.71 | 72 % | 176 % | 28 % |
| o_proj | 0.79 | 70 % | 88 % | 30 % |
| k_proj | 1.00 | 95 % | 0.02 % | 3 % |
| v_proj | 1.00 | 100 % | 0.01 % | 0 % |

Two further facts:
* expert orderings are not consistent across matrices (Kendall's W = 0.011; per projection 0.015-0.11), so no expert
  is "the ill-conditioned one";
* within a matrix, kappa is unrelated to update size in q/v/o (Spearman with ||Delta||_F: +0.01 / -0.02 / -0.13) and
  negatively related in k_proj (-0.44).

**Reading [deduction + established].**
* In the resolved matrices (v_proj, most k_proj), kappa barely differs across experts (0.2-0.8 %). There the rule is
  close to uniform at any moderate temperature.
* In the matrices where kappa differs a lot (q/o_proj), all of the difference sits in sigma_min, 30-1500x below
  what the stored base resolves. One third of the rule's choices there flip under an equally valid rounding of the
  same checkpoint.
* The rule's discriminating signal is therefore concentrated exactly where it is least reliable.

Records: results/llm/spectral/l31_8b_T8_{validation,diagnostics,rerounding}.{json,csv}. Figure:
results/llm/summary/figures/fig_llm_mechanism.png.

**Per-matrix kappa is not a property of the network function [deduction, checked numerically].** Llama attention
admits exact function-preserving reparametrisations that change per-matrix condition numbers:
* for each KV head, W_V -> M W_V, with every query head's o_proj block multiplied by M^-1 (attention output is
  linear in V);
* for each KV group and RoPE frequency pair, W_Q rows scaled by c and W_K rows by 1/c (a scaled identity on a 2-D
  pair commutes with the RoPE rotation).

On the real base weights of layer 5 (experiments/llm/gauge_check.py, results/llm/spectral/gauge_check.json):

| reparametrisation | function change | kappa of each matrix |
|---|---|---|
| v/o | outputs change by 7e-14 relative | kappa(W_V) 8.3 -> 1881; kappa(W_O) 3.1e4 -> 2.4e6 |
| q/k | attention logits change by 3e-15 | kappa(W_Q) 5.2e5 -> 1.7e6; kappa(W_K) 78 -> 455 |

Experts that share one base are compared in a single gauge, so this does not by itself make across-expert
comparisons meaningless. It does remove any reading of kappa(W) as a functional "singularity" or reliability property
of an attention matrix.

## 4. Does the rule help on the LLM banks? [pending]

(S1 dev grid running. Frozen configurations will be in configs/llm/frozen_S1.json; test/fresh in
results/llm/summary.)

## 5. Mechanism tests on the LLM banks [pending]

(Tail edit within the BF16 floor: cells A-D plus uniform; edited-expert accuracy, KL and hidden-state change.
Gradient-preference allocation test.)

## 6. Refinements R1 (log kappa) and R2 (kappa floored at the BF16 resolution) [pending]

## 7. Inverse-solve conditioning

**7.1 The "4.1 %" collapses are numerical [established].** The ~4.1 % LoRA RegMean collapses of Report 2 are fp16
evaluation overflow:
* the unregularised solve puts weights up to 6e5 (base max 0.27) in directions the calibration Grams barely span;
* fp16 autocast produces non-finite features, and every image is sent to one class;
* in fp32 the same weights score about as well as the kappa-capped solve.

| fp32 evaluation, acc % | plain RegMean | kappa cap 1e4 |
|---|---|---|
| KnOTS N = 512, alpha 1.3 (the reported collapse) | 50.83 | 50.92 |
| KnOTS N = 16, alpha 0.8 / 1.0 / 1.3 | 62.29 / 57.79 / 47.77 | 62.30 / 58.00 / 47.97 |
| lora_c5a N = 512, alpha 0.8 / 1.0 / 1.3 | 65.40 / 63.69 / 61.05 | 65.62 / 63.65 / 60.87 |

**The unregularised solve is numerically ill-posed rather than inaccurate.**
* The Grams are exactly symmetric with no PSD violations, but some are singular (kappa = inf). The median kappa is
  6e4 at N = 512 and 1.8e5 at N = 16.
* Float64 LU has residuals <= 1e-12. LU, Cholesky and explicit inversion agree to ~1e-7 relative.
* A pseudo-inverse, which drops the near-null directions, gives a solution that differs by 100 %. A float32 solve
  differs by 130-430 %.
* So the huge components of the plain solution are fixed by float64 arithmetic, not by the data.
* With the kappa cap, every solver agrees to 1e-12 and float32 agrees to 2e-4.

Records: results/inverse/collapse_audit.json.

**Full fine-tuning is different.** The n = 16 degradation is statistical, the same in fp16 and fp32, with no
non-finite values:

| subset | plain RegMean, fp16 / fp32 (norm %) | kappa cap 1e5 | shrinkage 0.95 |
|---|---|---|---|
| 0 | 65.24 / 65.30 | 88.73 | 88.63 |
| 1 | 65.87 / 65.90 | 87.88 | 87.56 |
| 2 | 63.75 / 63.96 | 88.57 | 88.27 |

**7.2 Equal-budget LoRA comparison (fp32) [pending].** results/inverse/lora_ridge_equal_test.json.

## 8. Uncertainty of calibration-derived corrections [pending]

(Diagnostic: does bootstrap uncertainty of F = B G^-1 predict F's error against a 1024-image reference better than
kappa(G), 1/lambda_min, tr(G^-1) and ||F||? Pre-registered criterion in protocol sec. 9. The gate stage runs only if
the diagnostic passes.)

## 9. Prior work [established; details docs/lit/PRIOR_WORK_02.md]

* **The professor's exp(-beta kappa(W)) weighting:** no instance found in 58 queries and 34 full texts. In merging
  papers, condition numbers appear only as diagnostics: SWUDI, TARA, HARC.
* **A theoretical caveat** (the reviewer's inference from SLOA Example 1): per-matrix kappa(W_Q) is not invariant
  under the function-preserving reparametrisation (W_Q R, W_K R^-T). The same holds for LoRA factors. So a
  per-matrix kappa is not a property of the network function.
* **Kappa-capped ridge:** prior art as a family. IterIS's relative ridge alpha ||G||_F I bounds
  kappa <= (1 + alpha)/alpha. Condition-number-constrained covariance estimation is established statistics (Won et
  al. 2013). It is a re-parametrisation of a ridge, not a new method.
* **Bootstrap-gated corrections:** not found in merging. ReForge has the anchor-ridge algebra and an analytic
  posterior but tunes lambda on validation, with no resampling and no gate. The mechanism class (Stein-rule and
  pretest shrinkage, bagging) is textbook statistics.

## 10. Corrections to the earlier reports [established]

These were applied in place in docs/FULL_REPORT.md (errata block and "[corrected 2026-10-19]" markers), with notes
in REPORT_02 and SUMMARY:
* the LoRA 4.1 % collapses are numerical (sec. 7.1);
* a failed transfer does not prove that a gain was a selection artifact;
* a fitted "oracle" is an achieved value, not an upper bound;
* an accuracy-preserving edit is not shown to be function-preserving;
* interlacing bounds a LoRA expert's kappa from below but does not pin it to the base. On the Llama banks q/o kappa
  varies by up to ~10^3 x kappa0 across experts;
* the worst-case sensitivity bound does not by itself determine coefficients;
* units are made explicit.

## 11. Decision [pending]

## 12. Message to the professor [pending; drafted only after the S1/S2 test, fresh and tail results]
