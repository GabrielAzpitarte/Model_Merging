# Conditioning-guided model merging: summary of findings (2026-10-18)

**Proposal under test** (the professor): "When you merge multiple W, you have weighting
parameters lambda. We can use an exponential function of W's condition number to determine that
weight ... larger condition number should have lower lambda."

**Short answer.**
1. As an expert-weighting rule, the proposal does not work on shared-base experts, in any form we
   could make precise.
2. Mechanistically, kappa2 of a fine-tuned weight is a probe of the *base's* weakest singular
   directions, not of the expert's reliability.
3. Conditioning does matter in the place the mathematics says it should: when a merger inverts
   a matrix (RegMean). There the solve becomes unstable with noisy statistics. A regulariser set by
   the condition number fixes it, and has a small but consistent edge over RegMean's built-in shrinkage
   (+0.2 to +0.8 on frozen test). Most of that edge comes from using an isotropic ridge; setting its
   strength per map from kappa(G) adds about 0.1-0.2 pt over the best fixed ridge.

Detailed records:
* REPORT_01_literal_rule.md: the exact rule, its controls, transfer, and the mechanism.
* REPORT_02_inverse_problem.md: conditioning of the merge solve.
* aa/AA_SIDE_REPORT.md: synthetic checks, per-layer oracle, noisy copies, tail stress, LoRA.
* RESEARCH_LOG.md: chronology, including all pre-registrations.
* PROTOCOL.md: definitions and information budgets.
* MATH_NOTES.md: derivations.

## 1. The literal rule (weights-only)

`lambda_i = softmax_i(-beta * kappa2(W_i))` per linear map, `W = W0 + alpha * sum lambda_i (W_i - W0)`.

| bench | literal rule vs uniform (test, mean normalised acc) | matched shuffled-score null |
|---|---|---|
| ViT-B/32 full FT, 8 tasks (dev-tuned) | -0.08 [-0.20, +0.04] | -0.13 ... +0.15 |
| ViT-B/16 full FT (frozen B/32 config, pre-registered) | -0.11 | -0.76 ... +0.24 |
| 7 pre-registered expert subsets (dev) | -0.05 on average (3/7 positive) | -0.29 |
| KnOTS LoRA, 4 banks (peer) | no replicable effect (2/4 positive) | — |

About 45 nearby variants were each tested at the same budget:
* log, quantile-robust, standardised, strict-rank, base-relative and data-supported scores;
* global vs layerwise weights;
* task-update spectra;
* a tuning-free inverse-variance rule;
* merged-kappa gradients;
* both signs throughout.

None survives held-out testing. The two that looked best on ViT-B/32 test failed the pre-registered
ViT-B/16 transfer.

**Why** [measured + derived]:
* In well-conditioned maps (MLPs), experts' kappa differ by about 0.1 %, and the score is a rank-2
  linear probe of the update. To first order, the rule minimises the merged layer's condition number.
* In near-singular maps (attention; 27 of 73 are singular at fp32 tolerance), kappa is about
  1/sigma_min, which is set by function-irrelevant spectral tails. Flooring one expert's tiny
  singular values changes its accuracy by at most 0.08 pt, yet swings the rule's weight from 0.13
  to 0.69 and costs the merge up to 8 pt (peer).
* For noisy estimates of one target, the case where the review's math favours down-weighting,
  noise *lowers* kappa (smoothed analysis). The rule then up-weights the noisiest copy, so it is
  anti-inverse-variance.

## 2. Where the headroom actually is

* A labelled oracle in the rule's own family (per-map softmax plus alpha) gains +8 pt (full FT) and
  +12 pt (LoRA), almost all from giving each attention map to one expert.
* Expert-level (global) weights are worth at most +0.3 to +0.5 pt.
* No spectral statistic predicts the oracle's per-map choices.

## 3. Conditioning of the inverse problem

RegMean with task-separated unlabelled statistics reaches 91.4 (normalised; 82.8 % accuracy) on
ViT-B/32 test, against 76.7 for uniform/TA. With pooled statistics and no task ids it is about uniform.
* With few calibration images, the inverted Gram matrices become near-singular (kappa up to 1e13).
  Plain RegMean then collapses (64 vs 91 on full FT; 4 % on LoRA). Any regulariser fixes it.
* The **kappa-capped ridge** (smallest ridge with kappa2(G + rho I) <= kappa_t) beats RegMean's
  off-diagonal shrinkage by a small, consistent margin:
  * full FT +0.2 to +0.4, test 3/3;
  * LoRA +0.2 to +0.8, test CIs exclude 0 on 3/3 banks.
* Where the edge comes from (pre-declared 2x2, then a post-hoc sweep with equal tuning grids,
  frozen test; Report 2 §3-4):
  * Setting RegMean's own diagonal shrinkage by a kappa cap never helps (-0.04 to -0.48).
  * LoRA: the edge is isotropy (+0.5 to +1.0); kappa-adaptive strength adds at most +0.2.
  * Full FT, 16 images per task: kappa-cap ridge 87.69 vs tuned fixed ridge 87.51 vs tuned
    shrinkage 87.39. That is +0.12 from isotropy and +0.18 from kappa-adaptive strength, with
    3/3 calibration subsets positive.
  * Full FT, 512 images per task: all regularisers are within 0.07 of plain RegMean (91.39).
* Verdict: conditioning explains why the solve must be regularised. It adds at most about 0.2 pt
  as a per-map ridge rule. That is too small for a method claim, and it is close to SWUDI's
  spectral filtering of merge solves.
