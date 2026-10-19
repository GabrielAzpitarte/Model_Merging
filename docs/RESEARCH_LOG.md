# Research log: conditioning-guided merging (condmerge)

Entries are dated, append-only. Labels: **[established]** = measured, run completed;
**[tentative]** = pattern seen but not yet controlled / replicated; **[deduction]** =
mathematical argument made here; **[assumption]** = not yet verified.

## 2026-10-18: setup

* Two Claude sessions were started with the same brief. Split: master and the full-FT
  pilot (session azpitarte-5f, GPUs 0-2); synthetic checks, KnOTS LoRA bank and the
  gradient-preference / oracle-lambda diagnostics on branch `aa/side`
  (session azpitarte-aa, GPUs 3-4). See docs/PROTOCOL.md for all definitions.
* **[established]** Evaluation pipeline reproduces published numbers: base zero-shot
  test mean 48.0 (published 48.0); FusionBench experts test mean 90.3 / dev 90.5
  (published 90.5); fp16 vs fp32 differences <= 0.19 pt per task.
* **[established]** Dev baselines (8 experts, mean normalised / mean absolute accuracy):
  uniform average (alpha = 1) 74.8 / 67.3; Task Arithmetic best alpha = 1.6 (s = 0.2)
  77.7 / 70.3; MetaGPT (lambda prop. to ||D||^2) alpha = 1: 74.6 / 67.2, best alpha = 1.6: 77.0 / 69.8.

## Hypotheses under test (from the review, section 4.1, made concrete)

| id | hypothesis | discriminating measurement |
|---|---|---|
| H1 | Literal rule: lower kappa2(W_i) per map deserves more weight | `kappa` vs `kappa_rev` vs `shuf*:kappa` at matched (b, alpha) grid; pair lam* vs kappa ratio |
| H2 | kappa only proxies update magnitude (peer C11: in near-singular maps, larger updates lift sigma_min) | within-map corr(kappa2(W_i), \|\|D_i\|\|); `kappa` vs `dnorm_rev` |
| H3 | Apparent gains at alpha = 1 come from concentration raising the update scale | `kappa` vs shuffled at identical concentration; fixed-alpha vs tuned-alpha |
| H4 | Variance route (lambda prop. to kappa^-2) | `logkappa` at its variance-implied temperature (beta = 2 on log kappa) |

## 2026-10-18 13:57-15:30: pilot results

### Wave 1: literal rule and controls (8 experts, dev, shared 66-config grid)  [established]
* The literal `kappa` rule is indistinguishable from uniform and from the shuffled-score null.
  * dev-best: 77.74, against 77.74 +- 0.19 for the shuffled null (5 seeds, range 77.54-78.10) and 77.65 for uniform-tuned.
  * At alpha = 1 it is flat at 74.6-74.8 for every finite b; the argmin limit collapses to 65.0.
* `kappa_rev`, `logkappa`, `logkappa_rev` and `kappa_strict` all fall inside the null.
* Concentrating every map destroys coherence: `kappa_z` at b = 4 gives 66; per-map largest-update selection gives 36.
* MetaGPT gives 76.98 and the best single expert 54.8. The best single expert is below zero-shot (55.0).

### Frozen TEST (configs frozen in edbdcf5 / ee218d2 before evaluation; paired task-stratified bootstrap vs uniform-tuned 76.68)  [established]
* The literal rule is `kappa|tuned` 76.61, -0.08 [-0.20, +0.04]; at alpha = 1 it gives 73.29 vs 73.39.
* The 5 shuffled-kappa nulls range from -0.13 to +0.15.
* Reversed and variant rules, all at or below uniform:

  | rule | diff vs uniform |
  |---|---|
  | `kappa_rev` | -0.23 |
  | `logkappa` | -0.02 |
  | `kappa_strict` | -0.30 |
  | `kappa_z` | -0.69 |
  | `global:kappa_z` | -0.93 |
  | MetaGPT | -0.97 |
  | `dnorm` | -0.21 |
  | best single | -22.7 |

* Two rules sit above the null. They are candidates only, because each was selected from about 45 rules on dev:
  * `logkappa_q90_rev`, a reversed tail-robust bulk ratio s1/s_{0.9q}: +0.47 [+0.24, +0.70].
  * `global:logkappa`: +0.27, with its reverse at -1.35. The gshuf global nulls give -0.76 / -0.68 / +0.01.
  * Both go to a pre-registered ViT-B/16 transfer test (configs/b16_transfer_configs.json, 75b4237).

### Spectra, pairs, subsets  [established]
* Spectra fall into two regimes (MATH_NOTES N1-N2):
  * MLP maps: kappa varies by about 0.1 % across experts, and the first-order rank-2 probe holds (r about 0.95).
  * Attention maps: near-singular, with about 100 % spread. 27 of 73 base maps are fp32-singular; L0/L1 q/k are rounding-dominated.
* Pairs: worth model R^2 0.91. kappa does not predict it (rho -0.02). The larger update deserves less weight (rho -0.49, p 0.008).
* The 7 pre-registered subsets on dev show nothing either:

  | rule | mean diff vs uniform | subsets positive |
  |---|---|---|
  | `kappa` | -0.05 | 3/7 |
  | shuffled mean | -0.29 | 0/7 |
  | `kappa_rev` | -0.82 | 1/7 |

### Wave 2/3: nearby interpretations, bounded  [established on dev; test numbers in results/freeze/test.jsonl]
* Robust and base-relative full-weight scores (`logkappa_q90`, `kappa_rel`): within the null, except the candidate above.
* Data-supported scores (B), using kappa(W_i C^{1/2}) and kappa_+: within the null. The peer finds them blind to per-map targets.
* Update-domain scores (entropy/stable rank, kappa_q90 of D, data-supported norms): none beat uniform. The spread statistics encode expert identity (peer).
* Inverse-variance random-matrix rule (Gavish-Donoho sigma_hat, derived beta = 1): hurts on the distinct-task bank (75.5 dev), as predicted.
* Interaction-aware rules:
  * Merged-kappa gradient (`mkgrad`, both signs): about 1 pt below uniform.
  * Within-expert-centred rules (`wc:*`): harmless at small b, collapse when they concentrate.

### Peer track (docs/aa/AA_SIDE_REPORT.md, merged 4099e6d)  [established unless marked]
* kappa is blind to the per-layer gradient preference and to the oracle allocation.
* The oracle headroom is +8 pt (full FT) / +12 pt (LoRA), all in per-map attention routing; global weights give at most +0.4.
* Tail stress: a function-null edit (0.5 % of ||D||) swings lambda_expert from 0.13 to 0.69 and costs up to 8 pt.
* Shared-target noise: the literal rule is anti-inverse-variance, because noise lowers kappa. The update spectrum shape works under isotropic noise only and fails on real recipe variants.
* LoRA: the hard-kappa-selection effect did not replicate across 4 banks (2 positive, 2 at chance). Dead end.

## 2026-10-18 16:15: inverse-problem route (review hypothesis row 3), test confirmation plan

Primary test comparisons were fixed from dev before any test evaluation. All other test rows are descriptive.

* Budget C (task-separated calibration), n = 512:
  * plain RegMean (a = 1, alpha = 1), the untuned standard;
  * dev-best kappa cap: ridge to mean, kappa_t = 1e5, alpha = 1.2;
  * dev-best fixed shrinkage: a = 0.99, alpha = 1.2.
* Budget B (pooled, no task ids): dev-best is the diagonal solve (a = 0, alpha = 1.6), compared with uniform-tuned (76.68 test).
* Instability: n = 16 per task, seeds 0/1/2, alpha = 1. Plain vs shrink 0.95 vs kappa cap 1e5.

## 2026-10-18 16:35: 2x2 mechanism check (pre-declared)

* RegMean test results:

  | budget | method | test |
  |---|---|---|
  | C, n = 512 | plain | 91.39 |
  | C, n = 512 | kappa cap 1e5 | 91.34 |
  | C, n = 512 | shrinkage 0.99 | 91.24 |
  | B | diagonal solve | 77.35 (uniform 76.68) |

* At n = 16 on test, kappa cap minus shrinkage 0.95 is +0.19 / +0.43 / +0.27 (3/3 positive). The peer finds 7/7 positive on LoRA (+0.6 mean).
* 2x2 cells:
  1. kappa-capped isotropic ridge;
  2. kappa-capped diagonal shrinkage;
  3. fixed shrinkage;
  4. fixed relative isotropic ridge, with strength matched to cell 1 via c = median lambda_max/(tr/d) / kappa_t.
* Primary matched comparison: kappa_t = 1e5 (cells 1, 2, 4) vs a = 0.95 (cell 3), alpha = 1, at n = 16 (3 subsets) and n = 512, on dev AND test.
* Verdict rule:
  * if cell 2 keeps the edge and cell 4 loses it, the edge is due to conditioning;
  * if the reverse, it is due to isotropy.

## 2026-10-18 22:40: 2x2 outcome; post-hoc strength sweep (declared before running)

* The master session ended at 16:41, before its 2x2 jobs finished (dev 16:43, test 16:48).
  A later session recovered the results (results/regmean/twobytwo_C_{dev,test}.jsonl, commit 2e09047)
  and writes them up in Report 2 §3.
* **[established]** Pre-declared verdict rule, primary cells (kappa_t = 1e5 vs a = 0.95, alpha = 1):
  * cell 2 (kappa-capped diagonal shrinkage) LOSES the edge everywhere: -0.11 / -0.48 test (n = 16 / 512);
  * cell 4 (strength-matched fixed relative ridge) keeps it at n = 512 (+0.31 test) and LOSES it at n = 16 (-0.49 test).
  * So at n = 512 the edge is isotropy; at n = 16 neither pre-declared outcome holds.
* Open point: cell 4's strength was matched on the median map only. The two fixed strengths tried
  differ by 0.4-0.6 pt at n = 16, so a properly tuned fixed ridge has not been compared.
* **Post-hoc check, equal tuning grids** (budget C, alpha = 1, n = 16 subsets 0/1/2 and n = 512, dev;
  results/regmean/strength_sweep_C_dev_*.jsonl):
  * kappa-cap ridge, kappa_t in {1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6};
  * fixed relative ridge, c = r_med / kappa_t over the same 7 values (c from about 0.16 down to 1.6e-4);
  * fixed diagonal shrinkage, a in {0.5, 0.8, 0.9, 0.95, 0.99}.
* Selection rule, fixed now, before any test evaluation:
  * per arm and per n, take the grid value with the highest dev mean normalised accuracy
    (for n = 16, the mean over the 3 subsets);
  * evaluate only those configs on test.
  * Primary comparison: kappa-cap ridge vs fixed relative ridge. Both are isotropic, so this
    isolates kappa-adaptive strength. Secondary: each against the tuned fixed shrinkage.

## 2026-10-18 22:45: strength sweep on dev; test selection (recorded before any test run)

* **[established, dev]** Mean normalised %, alpha = 1, budget C. n = 16 is the mean of 3 subsets; n = 512 is one set.
  * kappa-cap ridge peaks at kappa_t = 3e4-1e5: 88.32-88.37 at n = 16, 91.58 at n = 512.
    Its median per-map strength there is only 8e-4 to 3.5e-3 of tr(G)/d, because the ridge goes to the ill-conditioned maps.
  * The fixed relative ridge peaks at c of about 0.016: 88.02 at n = 16, and is flat at 91.4-91.5 for c <= 5e-3 at n = 512.
  * Fixed shrinkage peaks at a = 0.95-0.99: 88.14 at n = 16, 91.55 at n = 512.
  * At n = 512 every regularised solve is within 0.2 of plain RegMean (91.68), so regularisation is not needed there.
  * The 24 configs repeated from the pre-declared run reproduce exactly (max |diff| 0.000).
* Selected by the declared rule; only these are evaluated on test (results/regmean/strength_sweep_C_test_*.jsonl):

  | n | kappa-cap ridge | fixed relative ridge | fixed shrinkage |
  |---|---|---|---|
  | 16 | kappa_t = 1e5 | c = r_med / 1e4 (about 0.016) | a = 0.95 |
  | 512 | kappa_t = 1e5 | c = r_med / 1e6 (about 1.6e-4) | a = 0.99 |

## 2026-10-18 22:50: strength sweep frozen test  [established]

* Test, mean normalised % (n = 16: subsets 0/1/2 and their mean; then n = 512):

  | arm | n = 16 subsets | n = 16 mean | n = 512 |
  |---|---|---|---|
  | kappa-cap ridge (kappa_t 1e5) | 88.04 / 87.20 / 87.83 | 87.69 | 91.40 |
  | fixed ridge (c about 0.016 at n = 16, 1.6e-4 at n = 512) | 87.93 / 86.91 / 87.68 | 87.51 | 91.37 |
  | shrinkage (a 0.95 at n = 16, 0.99 at n = 512) | 87.85 / 86.77 / 87.56 | 87.39 | 91.33 |

* n = 16 differences:
  * kappa cap - fixed ridge: +0.10 / +0.29 / +0.15 (mean +0.18);
  * kappa cap - shrinkage: +0.30 mean;
  * fixed ridge - shrinkage: +0.12 mean.
* n = 512: all within 0.07 of each other and of plain RegMean (91.39).
* All test records carry commit 6ae7a69 (clean tree).
* Reading: on full FT at small n, the kappa-cap ridge's edge over RegMean's shrinkage is about
  40 % isotropy and 60 % kappa-adaptive per-map strength. Both parts are small, and the n = 16
  evidence is 3 calibration subsets. Write-up: Report 2 §3-6; summary updated.
