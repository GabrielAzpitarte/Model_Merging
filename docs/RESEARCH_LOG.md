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
