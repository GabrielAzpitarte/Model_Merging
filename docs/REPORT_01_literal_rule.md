# Report 1: the literal condition-number rule on FusionBench ViT-B/32 (full fine-tuning)

Date: 2026-10-18. Sessions: master (azpitarte-5f) and branch `aa/side` (azpitarte-aa).
Status labels: **[established]** = completed runs; **[tentative]**; **[deduction]**.
Definitions are in docs/PROTOCOL.md and the mathematics in docs/MATH_NOTES.md.

## 1. What was tested

**Full-weight, layerwise exponential condition-number rule** (`kappa`), for each of the 73 linear maps:
`lambda_i = softmax_i(-beta * kappa2(W_i))`, merged as `W0 + alpha * sum_i lambda_i (W_i - W0)`.
`alpha = 1` is the professor's normalised average `sum_i lambda_i W_i`, and `beta = 0` is uniform averaging.
`kappa2 = s_1 / s_q` is computed exactly in float64 from the stored float32 matrix. Biases inherit
their weight's lambda; LayerNorm and embedding tables are averaged uniformly.
Temperatures: `beta = b / scale` with b in {0.25 ... 128, inf}, where scale is the median within-map
score s.d.; alpha is in {1.0, 1.6, 2.0, 2.4, 2.8, 3.2}. Every rule gets the same 66-config grid.
All rules are weights-only (A); selecting (b, alpha) on dev is shared by every method.

Bench: the 8 FusionBench full-fine-tuned CLIP ViT-B/32 experts. The pipeline reproduces the published numbers:
zero-shot 48.0 and experts 90.3/90.5, matching the published values.

## 2. Results on dev [established]

Mean normalised accuracy (%), 8 experts:

| rule | alpha = 1, best b | best over shared grid (b, alpha) |
|---|---|---|
| uniform (beta = 0) | 74.80 | 77.65 (alpha 1.6), acc 70.28 |
| **kappa (literal)** | 74.80 (b=2) | **77.74** (0.25, 1.6) |
| shuffled kappa, 5 seeds | 74.65-75.03 | 77.74 +- 0.19 (77.54-78.10) |
| kappa_rev (reversed) | 74.81 | 77.35 |
| logkappa (power law) | 74.88 | 77.71 |
| kappa_strict (numerical-rank rule) | 74.88 | 77.52 |
| kappa_z (per-map standardised) | 74.40 | 76.73 |
| global:kappa_z (one lambda per expert) | 74.60 | 76.65 |
| MetaGPT (lambda prop. to \|\|D\|\|^2) | 74.55 | 76.98 |
| dnorm (smaller update first, per map) | 74.00 | 77.26 |
| best single expert (Cars) | 54.82 | n/a |

* The literal rule matches uniform and lies exactly at the mean of the shuffled-score null. Its
  shuffled versions use the same per-map scores in random expert order, so they have the same
  concentration profile. There is no evidence that kappa2(W_i) carries information about which
  expert deserves weight.
* At alpha = 1 the rule is flat at 74.6-74.8 from b = 0.25 to b = 128, while concentration drops
  from n_eff 6.9 to 3.6 of 8. The argmin limit, one expert per map, collapses to 65.0. So the
  "more concentration raises update scale" confound produces no gain here (H3).
* Rules that concentrate *every* map, MLP maps included, degrade monotonically: kappa_z at b = 4
  gives 66.1, and per-map largest-update selection gives 36. Rules that only concentrate attention
  maps (literal kappa, by accident of its scale) are harmless.

Figure: results/figures/pilot_rule_vs_b.png. Raw records: results/sweeps/rule_sweep_dev*.jsonl.

## 3. Why: what the literal score measures on this bank [established + deduction]

* kappa0 spans 14 to 1.0e8 across the 73 maps, and the maps fall into two regimes (MATH_NOTES N1).
  * **MLP maps (24):** kappa0 is 15-60, and experts differ from the base by about 0.1 % relative.
    The first-order expansion `kappa(W0 + D) ~= kappa0 (1 + <D, u1 v1^T/s1 - uq vq^T/sq>)`
    explains the differences (median r = 0.94-0.98). The score is therefore a rank-2 linear probe
    of the update that ignores all but 2 of m*n directions. Being linear, it makes the rule
    first-order entropy-regularised minimisation of the merged layer's condition number (N2).
  * **Attention maps (48):** kappa0 is 1e3-1e8, fine-tuning moves kappa by about 100 %, and the
    first-order fit fails. 27 of 73 base maps are numerically rank-deficient at fp32 tolerance. In
    L0/L1 k_proj and L0 q_proj, fp32 rounding-level noise is 50-90 % of the across-expert spread.
* Within maps, kappa2(W_i) is mostly sigma_min(W_i): within-layer rho = -0.91 (peer). It is
  unrelated to update magnitude (median Spearman 0.05). Its argmin winners are spread over all 8
  experts, so no expert is consistently "ill-conditioned" (pairs: fraction of maps with
  kappa_i > kappa_j is 0.4-0.6 for every pair).
* **Pairs [established]:** in 28 pairs x 11 lambdas, the favourable lambda* follows a per-expert
  worth model (R^2 = 0.91; DTD and Cars deserve more weight, MNIST and EuroSAT less). Worth is
  unrelated to kappa (rho = -0.02, p = 0.96; sign test 10/19). Larger updates deserve *less*
  weight (rho = -0.49, p = 0.008).
* **Gradient preference and oracle (peer, budget D) [established]:** within-map rho(kappa,
  preference) is about +0.05 (n.s.). The layer-specific part has a small *reverse* sign
  (+0.095, p about 0.02). This fits the worst-case sensitivity argument N5, which says
  higher-kappa maps are more fragile and want more weight. An oracle layerwise softmax fitted with
  labels gains about 8 points of absolute accuracy, all from per-map expert selection in attention
  maps; a global oracle lambda gains only 0.3. kappa does not predict the oracle's choices
  (within-map rho about 0). The data-supported score kappa(W_i C^1/2) is equally blind.
* **Shared-target noise (peer) [established]:** with noisy copies of one expert, the regime where
  the review's inverse-variance math favours decreasing weights, noise *lowers* kappa in about 73 % of maps
  (smoothed analysis). The literal rule therefore up-weights the noisiest copy and is
  anti-inverse-variance. The update's stable rank recovers the oracle.

## 4. Confounders addressed

| confounder | control | outcome |
|---|---|---|
| chance / selection on dev | shuffled scores (identical concentration), 5 seeds | kappa = null mean |
| update scale | matched alpha grid, alpha = 1 vs tuned alpha | no gain at either |
| expert selection | argmin limit, best single expert | both far below uniform |
| magnitude | dnorm, MetaGPT, sigma decomposition | magnitude rules are no better |
| sign | kappa_rev, logkappa_rev | inside the null too |
| layer temperature | logkappa, kappa_z, kappa_rel | no gain; kappa_z harmful |
| precision | fp64 SVD; fp32 noise check; kappa_strict | rounding-dominated maps identified; strict rule = null |

## 5. Frozen test, subsets and transfer [established]

**ViT-B/32 test.** Configurations were frozen on dev (commits edbdcf5, ee218d2) before any test
evaluation. Differences are against uniform-tuned (76.68 normalised), with 95 % CIs from a paired,
task-stratified bootstrap over test images.

| rule (dev-tuned) | test norm | diff [CI] |
|---|---|---|
| uniform, alpha 1.6 | 76.68 | 0 |
| **kappa (literal)** | 76.61 | -0.08 [-0.20, +0.04] |
| shuffled kappa (5 seeds) | 76.55-76.84 | -0.13 ... +0.15 |
| kappa_rev / logkappa / kappa_strict | 76.46 / 76.66 / 76.38 | -0.23 / -0.02 / -0.30 |
| kappa_z / global:kappa_z / MetaGPT | 76.00 / 75.75 / 75.71 | -0.69 / -0.93 / -0.97 |
| literal kappa at alpha = 1 vs uniform at alpha = 1 | 73.29 vs 73.39 | -0.10 |

Test results for all 49 dev-selected rules are in results/freeze/test_comparison.json; most are
below uniform. Only `logkappa_q90_rev` (reversed tail-robust bulk ratio, +0.47) and
`global:logkappa` (+0.27) were above the B/32 shuffle range. Both were selected from about 45 rules.

**Pre-registered subsets** (configs/subsets.json; 4 representative/random 4-expert subsets and
2 random 6-expert subsets; dev): kappa minus uniform averages -0.05 pt (positive in 3/7), against
-0.29 for the shuffled mean and -0.82 for kappa_rev.

**Pre-registered ViT-B/16 transfer** (configs/b16_transfer_configs.json, commit 75b4237). The B/32-frozen rules
and their (b, alpha) were evaluated once on the B/16 test split, with matched-config shuffles.

| rule | B/16 test norm | diff vs uniform (82.01) |
|---|---|---|
| kappa (literal) | 81.90 | -0.11 (its shuffles: -0.76 ... +0.24) |
| logkappa_q90_rev | 81.00 | **-1.01** (its shuffles: -0.95 ... +0.32) |
| global:logkappa | 82.03 | +0.02 (gshuf: -1.04 ... -0.08) |
| kappa_rev | 79.87 | -2.15 |

Both B/32 candidates fail to transfer, so they were selection artifacts. `logkappa_q90_rev` is
also strongly harmful on four LoRA banks (peer: -4 to -17 pt, worse than its own shuffles).

**LoRA (peer).** A hard-kappa-selection effect on the KnOTS bank did not replicate across 4
banks (2 positive, 2 at chance), and no mechanism survived. Dead end.

## 6. Verdict on the professor's hypothesis

* **Rejected for multi-task merging of shared-base experts, in every form tested.** These are
  full fine-tuning (ViT-B/32, ViT-B/16) and LoRA, with layerwise or global weights, raw/log/robust/
  standardised/strict/base-relative/data-supported scores, and both signs. At matched budgets no
  conditioning rule beats uniform averaging or a shuffled-score null on held-out test data.
* **Mechanism.** In a shared-base bank, kappa2(W_i) is a property of the base perturbed by the update.
  * In well-conditioned (MLP) maps it is a rank-2 linear probe of the update, about 0.1 % spread.
  * In near-singular (attention) maps it is essentially 1/sigma_min, which is set by function-
    irrelevant tails: a 0.5 % edit swings the weights and costs up to 8 pt.
  * Under estimation noise it points the wrong way: noise *improves* conditioning (smoothed
    analysis), so the rule up-weights the noisiest copy.
* **Where a decreasing rule is justified mathematically** (noisy estimates of one target), the
  usable signal is the *update's* spectral shape, with the reverse sign: flatter means noisier. That
  signal works under isotropic noise only and fails on real recipe variation (peer).
* **Where the headroom is.** A labelled oracle in the same per-map softmax family gains +8 pt
  (full FT) and +12 pt (LoRA), almost all from routing attention maps to single experts. No
  spectral statistic predicts that routing.

## 7. What deserves further investigation

1. **The inverse-problem route (review hypothesis "the instability lies in the solve").** This is the
   one branch of the review's hypothesis table not yet tested, and the only place where a condition
   number provably governs a merge's error (eq. 33-35). Test: RegMean-type solves with task-separated
   (budget C) and pooled-unlabelled (budget B) Gram statistics, and a conditioning-guided
   regularisation (per-layer shrinkage or ridge set by kappa(G)) against RegMean's fixed shrinkage.
   Nearest priors: RegMean, MaTS, RegMean++, ACE, SWUDI.
2. **A mechanism-level negative result** as a contribution in itself: the regime analysis, the
   first-order "merged-conditioning" equivalence, tail manipulability and the anti-inverse-variance
   effect. Novelty must be checked against SADA/LARV/CORAM and Ludwig (2025).
3. Outside the conditioning brief (needs a decision): data-driven per-map routing of attention maps,
   where the oracle headroom lives.

## Errata (2026-10-18 evening, script-based cross-check of this report against the raw files)

* §2:
  * The literal rule's best at alpha = 1 is at b = 4 (74.80), not b = 2 (74.67).
  * kappa_z gives 66.5 at b = 4 (66.1 is b = 8).
  * "Per-map largest-update selection gives 36" is not reproduced. dnorm_rev at b = inf gives 58.2 at
    alpha 1, 45.2 at 1.6 and 33.0 at 2.0.
* §1:
  * The absolute-beta Gavish-Donoho rules use 18 configs, not the 66-config grid.
  * The 10 data-supported `s*` rules use pooled calibration inputs (budget B), not weights only.
* §3:
  * MLP kappa0 is 14-60 in 18 of 24 maps (L0 fc1 509, L1 fc1 80, L11 fc2 63).
  * The rounding-dominated maps also include L2 k_proj (53 %).
  * The pairwise fraction of maps with kappa_i > kappa_j is 0.36-0.71, not 0.4-0.6 (5 of 28 pairs
    fall outside).
* §5:
  * A third rule is above the kappa shuffle range on B/32 test: wc:kappa_rev|tuned +0.21
    [+0.06, +0.38]. It was not carried to the B/16 transfer.
  * The global nulls are shuffles of global:kappa_z.
  * The subsets are five 4-expert and two 6-expert sets.
  * "About 45 rules" is 41 distinct non-null rules.
* The corrected numbers are in docs/FULL_REPORT.md, and the full tables in
  docs/appendix/A_fullft_tables.md.
