# Report 2: conditioning where the merge solves an inverse problem (review hypothesis row 3)

Date: 2026-10-18. Bench: 8 FusionBench CLIP ViT-B/32 full fine-tunes (master) and the KnOTS,
lora_c5a and vit20 LoRA banks (side branch, docs/aa/AA_SIDE_REPORT.md §9). Labels as in Report 1.
Full fine-tune scores are mean normalised accuracy (%); LoRA scores are mean absolute accuracy (%).

Sections 1-2 were written by the master session, which ended at 16:41 before its 2x2 runs
finished. Sections 3-6 were written the same evening from the recorded runs. The post-hoc
strength sweep in §4 and its test-selection rule were entered in docs/RESEARCH_LOG.md before
it ran.

## 1. Question

Report 1 rejected conditioning of the *forward weights* as a merge-weighting signal. The review
identifies one place where a condition number provably governs a merge's error: when the merger
inverts a matrix. For response matching (RegMean, review eq. 30-35),

    D (G + rho I) = sum_i D_i G~_i + rho R,      G = sum_i G~_i,   G_i = E_task i[x x^T],
    ||dD|| <= ||(G + rho I)^-1|| (||dB|| + ||D|| ||dG||).

Hypothesis: weak eigenvalues of G amplify noise in the estimated statistics; a regulariser set
by the condition number of G fixes the solve. We test whether that is true, and whether a
**conditioning-guided** regulariser beats RegMean's fixed off-diagonal shrinkage
`G~ = a G + (1 - a) diag G`. The regulariser is a kappa cap: the smallest rho with
`kappa2(G + rho I) <= kappa_t`, ridge toward the uniform average of the updates.

Code: src/condmerge/regmean.py (tested). The peer's independent implementation agrees to 5e-8.

## 2. Results

**RegMean is a strong baseline when statistics are task-separated (budget C).** Each expert's own
activations come from 512 unlabelled train images of its task.

| | dev norm | test norm (acc) |
|---|---|---|
| uniform average, alpha tuned | 77.65 | 76.68 (69.29) |
| RegMean plain (a = 1, alpha = 1) | 91.68 | 91.39 (82.77) |
| RegMean, kappa cap 1e5 (dev-best, alpha 1.2) | 92.07 | 91.34 |
| RegMean, shrinkage a = 0.99 (dev-best, alpha 1.2) | 91.93 | 91.24 |

This is consistent with published RegMean on this bench, about 80.5 % accuracy, since we use more
calibration images and no shrinkage. With 512 images per task, regularisation does not matter.

LoRA (peer, frozen test, N = 512, absolute accuracy):

| bank | uniform | RegMean, shrinkage 0.95 |
|---|---|---|
| KnOTS | 51.22 | 63.47 |
| lora_c5a | 53.04 | 64.45 |
| vit20 | 66.68 | 81.89 |

Truncating every merged map to rank 16, the size of one adapter, keeps most of the gain
(KnOTS dev: 61.2-61.7 vs 51.3 for uniform). Extra rank accounts for about 3.8 pt of it.

**Without task identities (budget B: each expert on the same pooled images) the regression has
nothing to separate the experts by.** RegMean-B gives 75.65 test at alpha = 1, the
kappa-capped version 75.9, and the best (a diagonal solve, alpha 1.6) 77.35, against 76.68 for
uniform. On LoRA (peer, dev), pooled Grams give -0.4 on KnOTS but +1.0 on lora_c5a and +2.6 on vit20
over uniform. These are small next to budget C (+13 to +15).

**The solve does become unstable with noisy statistics, and regularisation is essential**
[established, 3 calibration subsets]. Figure: results/figures/regmean_instability.png.

| alpha = 1, budget C | n = 16 / task (dev, 3 seeds) | n = 64 | n = 512 | n = 16, test |
|---|---|---|---|---|
| plain | 64.95 (63.7-65.9) | 89.99 | 91.68 | 64.20 |
| shrinkage a = 0.95 | 88.14 | 90.42 | 91.31 | 87.39 |
| kappa cap 1e5 | 88.37 | 90.73 | 91.58 | 87.69 |

kappa(G) has a median of 2.9e5 and a maximum of 1.8e13 at n = 16 (1.6e13-1.8e13 across subsets). The collapse comes from the near-singular
3072-dimensional inputs (fc2, patches). LoRA (peer): plain RegMean collapses to 4 % at every alpha in [corrected 2026-10-19: the LoRA 4 % collapses are fp16 evaluation overflow; in fp32 plain RegMean is within ~0.3 pt of the kappa-capped solve (results/inverse/collapse_audit.json). The full-FT n = 16 failure is genuine and statistical.]
every N = 16 setting.

**The conditioning-guided cap has a small, consistent edge over RegMean's shrinkage.**
* Full FT: +0.2 to +0.3 at every n on dev; +0.19, +0.43, +0.27 on test (3/3).
* LoRA (peer): +0.2 to +0.9, 7/7 positive, mean +0.66. The banks overlap, so these are not
  independent draws.
* LoRA frozen test (kappa cap 1e4 vs shrinkage 0.95, dev-best alpha; paired bootstrap over test
  images):
  * KnOTS +0.83 [+0.71, +0.95];
  * lora_c5a +0.51 [+0.40, +0.62];
  * vit20 +0.20 [+0.10, +0.30].
* The edge does not grow as n shrinks.

## 3. 2x2 mechanism check (pre-declared): conditioning or isotropy?

Declared in docs/RESEARCH_LOG.md (16:35) before the runs. Budget C, alpha = 1, n = 16
(3 calibration subsets) and n = 512. The regulariser is split into geometry x strength:

| geometry \ strength | kappa-adaptive (per map, from a kappa cap) | fixed |
|---|---|---|
| isotropic ridge to the mean | cell 1: smallest rho with kappa2(G + rho I) <= kappa_t | cell 4: rho = c tr(G)/d, c = median_maps(lambda_max / (tr(G)/d)) / kappa_t |
| diagonal shrinkage | cell 2: largest a with kappa2(a G + (1 - a) diag G) <= kappa_t | cell 3: a = 0.95 (RegMean default) |

Primary comparison: kappa_t = 1e5 for cells 1, 2 and 4, against cell 3. Declared rule:
* if cell 2 keeps cell 1's edge over cell 3 and cell 4 loses it, the edge is about conditioning;
* if the reverse, it is about isotropy (rotation invariance).

Code: experiments/regmean_2x2.py. Records: results/regmean/twobytwo_C_{dev,test}.jsonl.
Tables: experiments/analyze_regmean_2x2.py, which writes results/regmean/twobytwo_summary.json.

**Full FT [established].** Test; n = 16 is the mean of the 3 subsets, n = 512 in brackets:

| | kappa-adaptive | fixed | adaptivity effect |
|---|---|---|---|
| isotropic ridge | 87.69 (91.40) | 86.90 (91.38) | +0.78 (+0.03) |
| diagonal shrinkage | 87.28 (90.59) | 87.39 (91.07) | -0.11 (-0.48) |
| isotropy effect | +0.41 (+0.81) | -0.49 (+0.30) | |

Edge over cell 3 on test, per subset at n = 16, then at n = 512:

| cell | n = 16 (subsets 0 / 1 / 2) | n = 512 |
|---|---|---|
| 1, kappa-capped ridge | +0.19 / +0.43 / +0.27 | +0.33 |
| 2, kappa-capped shrinkage | -0.18 / -0.04 / -0.12 | -0.48 |
| 4, fixed ridge | -0.45 / -0.49 / -0.52 | +0.30 |

Dev shows the same pattern:

| cell | n = 16 (subsets 0 / 1 / 2) | n = 512 |
|---|---|---|
| 1 | +0.11 / +0.32 / +0.26 | +0.27 |
| 2 | -0.29 / +0.07 / -0.14 | -0.22 |
| 4 | -0.56 / -0.66 / -0.49 | +0.12 |

Outcome under the declared rule:
* **Cell 2 loses the edge in every setting.** Setting RegMean's own shrinkage by a kappa cap does
  not help. At kappa_t = 1e4 it is clearly harmful (-1.00 test at n = 16, -1.92 at n = 512). In some
  maps even a purely diagonal Gram (a = 0) misses the target, so the cap forces the strongest
  possible shrinkage there.
* **n = 512:** cell 4 keeps the edge (+0.30 test). This is the "isotropy" outcome.
* **n = 16:** cell 4 loses it too (-0.45 to -0.52 on test, in every subset), so neither declared
  outcome holds. The kappa-capped ridge beats both its diagonal counterpart and the
  strength-matched fixed ridge.
  * Cell 4 was matched to cell 1 only through the median map.
  * The cap's actual median strength is about half of that: 8.4e-4 vs 1.6e-3 of tr(G)/d.
  * The cap's strength also varies by orders of magnitude across maps.
  * A fixed ridge at its own best strength had not been tried. §4 does that.

**LoRA (peer) [established, dev only].** KnOTS bank, alpha = 0.8; strength matched at
kappa_t = 1e4 / c = 0.0136. n = 16 is the mean of 3 subsets, N = 512 in brackets:

| | kappa-adaptive | fixed | adaptivity effect |
|---|---|---|---|
| isotropic ridge | 63.60 (65.52) | 63.47 (65.29) | +0.13 (+0.23) |
| diagonal shrinkage | 62.58 (64.53) | 62.86 (64.76) | -0.28 (-0.23) |
| isotropy effect | +1.02 (+0.99) | +0.61 (+0.53) | |

On LoRA the declared outcome is clean: the edge is isotropy.

## 4. Post-hoc: kappa cap vs a fixed ridge tuned with the same budget [established]

Declared in docs/RESEARCH_LOG.md (22:40) before running, together with the test-selection rule.
Budget C, alpha = 1, all grid values on dev. Grids:

| arm | grid |
|---|---|
| kappa-cap ridge | kappa_t in {1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6} |
| fixed relative ridge | c = r_med / kappa_t over the same 7 values (c from 0.16 down to 1.6e-4) |
| fixed diagonal shrinkage | a in {0.5, 0.8, 0.9, 0.95, 0.99} |

Here r_med, 158-161, is the median over maps of lambda_max / (tr(G)/d). For each arm and each n,
the dev-best value was frozen, and only that value was evaluated on test. Records:
results/regmean/strength_sweep_C_{dev,test}_*.jsonl. The 34 configurations that repeat the
pre-declared run reproduce it exactly.

Dev, ridge arms (n = 16 is the mean of 3 subsets; selected values in bold):

| kappa_t (fixed ridge: c = r_med / kappa_t) | 1e3 | 3e3 | 1e4 | 3e4 | 1e5 | 3e5 | 1e6 |
|---|---|---|---|---|---|---|---|
| kappa-cap ridge, n = 16 | 86.75 | 87.77 | 88.11 | 88.32 | **88.37** | 88.08 | 87.57 |
| its median rho / (tr G/d) | 0.11 | 0.036 | 0.011 | 3.5e-3 | 8.4e-4 | 0 | 0 |
| fixed ridge, n = 16 | 87.08 | 87.75 | **88.02** | 87.96 | 87.58 | 86.84 | 85.47 |
| its c | 0.16 | 0.054 | 0.016 | 5.4e-3 | 1.6e-3 | 5.4e-4 | 1.6e-4 |
| kappa-cap ridge, n = 512 | 90.07 | 91.09 | 91.32 | 91.58 | **91.58** | 91.51 | 91.56 |
| fixed ridge, n = 512 | 90.38 | 91.06 | 91.33 | 91.44 | 91.43 | 91.51 | **91.52** |

The kappa cap's strengths are recorded as rho / (tr(G)/d + rho). For small values this equals
rho / (tr(G)/d); the 0.11 entry corresponds to 0.12.

Dev, fixed shrinkage (selected values in bold):

| a | 0.5 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| n = 16 | 85.58 | 87.16 | 87.76 | **88.14** | 88.14 |
| n = 512 | 88.00 | 90.43 | 91.10 | 91.31 | **91.55** |

Plain RegMean on dev: 64.95 at n = 16, 91.68 at n = 512. Two selections were ties at two
decimals, broken by the exact value as the rule specifies:
* n = 512 kappa cap: 3e4 = 91.5752 vs 1e5 = 91.5758;
* n = 16 shrinkage: 0.95 = 88.144 vs 0.99 = 88.137.

Frozen test of the selected configurations:
* n = 16: kappa_t = 1e5, c about 0.016, a = 0.95;
* n = 512: kappa_t = 1e5, c about 1.6e-4, a = 0.99.

| | kappa-cap ridge | fixed ridge, tuned | shrinkage, tuned |
|---|---|---|---|
| n = 16, subsets 0 / 1 / 2 | 88.04 / 87.20 / 87.83 | 87.93 / 86.91 / 87.68 | 87.85 / 86.77 / 87.56 |
| n = 16, mean | **87.69** | 87.51 | 87.39 |
| n = 512 | **91.40** | 91.37 | 91.33 |

Differences on test:

| | kappa cap - fixed ridge | kappa cap - shrinkage | fixed ridge - shrinkage |
|---|---|---|---|
| n = 16, subsets 0 / 1 / 2 | +0.10 / +0.29 / +0.15 | +0.19 / +0.43 / +0.27 | +0.09 / +0.14 / +0.12 |
| n = 16, mean | +0.18 | +0.30 | +0.11 |
| n = 512 | +0.03 | +0.07 | +0.04 |

Plain RegMean on test: 64.20 at n = 16 (mean of subsets), 91.39 at n = 512.

Reading:
* **At n = 512 the regulariser does not matter.** All three are within 0.07 of each other and of
  plain RegMean.
* **At n = 16 the kappa-cap ridge is best in every subset.** Its +0.30 over RegMean's shrinkage
  splits into two parts:
  * +0.11 from isotropy (tuned fixed ridge vs tuned shrinkage);
  * +0.18 from kappa-adaptive strength (kappa cap vs tuned fixed ridge).
* **The kappa cap wins with a far smaller typical ridge than the best fixed one.** Its median is
  8.4e-4 of tr(G)/d, against 0.016. It regularises the ill-conditioned maps strongly and leaves
  the rest almost untouched. That per-map targeting is the conditioning-specific part of the gain.
* **Size and uncertainty.**
  * +0.18 is small, and it rests on 3 calibration subsets, positive in all 3 on test and on dev.
  * Per-image predictions were not saved for these runs, so there is no image-level bootstrap.
  * For comparison, image-level 95 % intervals on the LoRA frozen test were about +-0.1 for
    differences of this size.

## 5. Verdict on hypothesis row 3

1. **Mechanism supported [established].** The merge solve is badly conditioned: median kappa(G)
   is 2.9e5 at n = 16, with a maximum of 1.8e13. With few calibration images, plain RegMean
   collapses (64 vs 91 on full FT; 4 % on LoRA), and any regulariser restores it.
2. **A conditioning-guided regulariser is the best of those tested, by a small margin
   [established].** Kappa-capped ridge vs RegMean's shrinkage on frozen test:
   * full FT: +0.19 to +0.43 at n = 16 (3/3 subsets);
   * LoRA: +0.20 to +0.83 (3/3 banks, CIs exclude 0).
3. **Where the margin comes from [established; small effects].**
   * LoRA: mostly isotropy (+0.5 to +1.0). Kappa-adaptive strength adds at most +0.2 (one matched
     strength, dev only).
   * Full FT, n = 16: +0.11 from isotropy and +0.18 from kappa-adaptive strength, against an
     equally tuned fixed ridge.
   * Full FT, n = 512: nothing (<= 0.07).
   * So conditioning matters in two ways:
     * it explains why the solve needs regularising;
     * a per-map ridge set by kappa(G) adds about 0.1-0.2 pt over the best single ridge strength.
4. **Relation to the professor's hypothesis [deduction].** The condition number that matters is
   that of the merge system's Gram matrix G = sum_i G_i, a property of the inputs and the solve.
   It is not kappa2(W_i) of any expert. It sets the regularisation, not the expert weights.
5. **Novelty [assessment].**
   * Capping a linear system's condition number with a ridge is a standard numerical device.
   * Spectral filtering of merge solves is close prior work (SWUDI).
   * A gain of at most 0.2 pt over a tuned fixed ridge does not support a method claim.
   * The large gain here, +14.7 normalised over uniform on full FT test, belongs to RegMean with
     task-separated statistics, an established method.

## 6. Limitations and open points

* **Full FT:** one bench (ViT-B/32, 8 tasks), alpha fixed at 1 in §3-4, 3 calibration subsets at
  n = 16 and one at n = 512.
* **LoRA 2x2:** dev only, with one matched strength. The §4 tuned comparison was not run on LoRA.
* **Uncertainty:** no image-level confidence intervals for §3-4, because per-image predictions
  were not stored.
* **Information budget:** every gain here needs task-separated unlabelled data (budget C). Pooled
  data without task ids (budget B) gives about uniform on full FT, and at most +2.6 on LoRA (dev).
* **Not tested:** other inverse-problem mergers (Fisher merging / MaTS, RegMean++, WUDI / OptMerge
  solvers) and larger backbones.
