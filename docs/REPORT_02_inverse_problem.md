# Report 2: conditioning where the merge solves an inverse problem (review hypothesis row 3)

Date: 2026-10-18. Bench: 8 FusionBench CLIP ViT-B/32 full fine-tunes (this session) and the KnOTS,
lora_c5a and vit20 LoRA banks (peer session, docs/aa/AA_SIDE_REPORT.md). Labels as in Report 1.

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
| RegMean, kappa cap 1e5 (dev-best, alpha 1.2) | 92.08 | 91.34 |
| RegMean, shrinkage a = 0.99 (dev-best, alpha 1.2) | 91.93 | 91.24 |

This is consistent with published RegMean on this bench, about 80.5 % accuracy, since we use more
calibration images and no shrinkage. With 512 images per task, regularisation does not matter.

**Without task identities (budget B: each expert on the same pooled images) the regression has
nothing to separate the experts by.** RegMean-B gives 75.65 test at alpha = 1, the
kappa-capped version 75.9, and the best (a diagonal solve, alpha 1.6) 77.35, against 76.68 for
uniform. LoRA behaves the same way (peer).

**The solve does become unstable with noisy statistics, and regularisation is essential**
[established, 3 calibration subsets]. Figure: results/figures/regmean_instability.png.

| alpha = 1, budget C | n = 16 / task (dev, 3 seeds) | n = 64 | n = 512 | n = 16, test |
|---|---|---|---|---|
| plain | 64.95 (63.7-65.9) | 89.99 | 91.68 | 64.20 |
| shrinkage a = 0.95 | 88.14 | 90.42 | 91.31 | 87.39 |
| kappa cap 1e5 | 88.37 | 90.73 | 91.58 | 87.69 |

kappa(G) has a median of 2.9e5 and a maximum of 1.6e13 at n = 16. The collapse comes from the near-singular
3072-dimensional inputs (fc2, patches). LoRA (peer): plain RegMean collapses to 4 % at every alpha in
every N = 16 setting.

**The conditioning-guided cap has a small, consistent edge over RegMean's shrinkage.**
* Full FT: +0.2 to +0.3 at every n on dev; +0.19, +0.43, +0.27 on test (3/3).
* LoRA (peer): +0.2 to +0.9, 7/7 positive, mean +0.6. The banks overlap, so these are not
  independent draws.
* The edge does not grow as n shrinks.

## 3. 2x2 mechanism check (pre-declared): conditioning or isotropy?

PENDING. Cells: {isotropic ridge, diagonal shrinkage} x {kappa-adaptive, fixed (strength-matched)}.
