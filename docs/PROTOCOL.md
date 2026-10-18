# Protocol and definitions (condmerge)

This file pins down every choice the professor's one-sentence proposal leaves open,
so that results are comparable across rules and sessions.

## The proposal and its literal interpretation

> "When you merge multiple W, you have weighting parameters lambda. We can use an
> exponential function of W's condition number to determine that weight ... larger
> condition number should have lower lambda."

**Full-weight, layerwise exponential condition-number rule (`kappa`).** For every
linear map `p` and expert `i` (full fine-tuned weight `W_i[p] = W0[p] + D_i[p]`):

    lambda_i[p] = exp(-beta * kappa2(W_i[p])) / sum_j exp(-beta * kappa2(W_j[p]))
    W_merge[p]  = W0[p] + alpha * sum_i lambda_i[p] * D_i[p]

`alpha = 1` is exactly the normalised full-weight average `sum_i lambda_i W_i`.
`beta = 0` is uniform averaging. Task Arithmetic with per-task scale `s` is uniform
lambda with `alpha = K s`.

* `kappa2 = sigma_1 / sigma_q`, `q = min(m, n)`, from a float64 SVD of the stored
  float32 matrix (exact for the stored matrix; `kappa2_raw` in `stats.csv`).
* `kappa_strict`: `+inf` when the numerical rank at tolerance `max(m,n) * eps32 * sigma_1`
  is below `q`; such experts get lambda 0 (all `+inf` in a layer -> uniform).
* Softmax via log-sum-exp; NaN scores raise.

## Parameters

* **Linear maps (73):** 12 x {q, k, v, out}_proj + 12 x {fc1, fc2} + the patch-embedding
  convolution reshaped to (768, 3*32*32).
* **Biases** of linear maps inherit their weight's lambda (`bias_policy = tied`).
* **Everything else** (LayerNorm, class embedding, position-embedding table) gets the prior
  (uniform lambda). The position embedding is a lookup table, not a linear map.
* Keys are transformers-5 unprefixed names (`encoder.layers.0.self_attn.k_proj.weight`).

## Temperature grid

`beta = b / scale(rule)`, `scale` = median over linear maps of the across-expert standard
deviation of the rule's scores, so `b` is the logit spread at the median layer.
`b` in {0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, inf}; `inf` = argmin selection per map.
`alpha` in {1.0, 1.6, 2.0, 2.4, 2.8, 3.2}. Every beta-rule gets this same grid
(66 configs); `uniform` and `metagpt` only tune alpha (6 configs). Coefficient
concentration is logged per config as `n_eff = exp(H(lambda))` averaged over maps.

## Named variants (each is a different method)

| name | score `c_i[p]` (lower -> more weight) |
|---|---|
| `kappa` | kappa2(W_i) (literal) |
| `kappa_rev` | -kappa2(W_i) (reversed preference) |
| `shuf<s>:kappa` | kappa2 scores permuted across experts within each map (seed s) |
| `logkappa`, `logkappa_rev` | +/- log kappa2(W_i) (power law in kappa) |
| `kappa_z`, `kappa_z_rev` | within-map z-score of kappa2 (layer-standardised temperature) |
| `kappa_strict` | kappa2 with the numerical-rank rule |
| `global:<rule>` | one score per expert = parameter-weighted mean of the per-map scores |
| `logsmax`, `neglogsmin` | log sigma_1(W_i), -log sigma_q(W_i) (the two halves of log kappa) |
| `dnorm`, `dnorm_rev` | +/- \|\|D_i\|\|_F per map (update-magnitude control) |
| `metagpt` | lambda_i proportional to \|\|D_i\|\|^2 over all parameters (no beta) |
| `d*` | statistics of the task update D_i (refinements, later) |

## Data, splits, information budgets

* Experts: FusionBench `tanganke/clip-vit-base-patch32_<task>` (full fine-tuning of the
  vision tower; frozen zero-shot heads from the base text tower). 8 tasks: SUN397, Cars,
  RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD.
* **dev** = 20 % of each official test split (permutation seed 0), used for *all*
  selection and diagnostics. **test** = remaining 80 %, touched only by
  `experiments/freeze_eval.py` on configurations frozen beforehand.
* **calib** = 512 train images per task (unlabelled use only).
* Information budgets: **(A)** weights only; **(B)** pooled unlabelled calibration
  inputs without task IDs; **(C)** task-separated unlabelled inputs; **(D)** labelled dev
  data. Every rule above is (A). Selecting (beta, alpha) on dev is a (D) step and is
  shared identically by every method. Oracle diagnostics are (D) and labelled as such.
* Metric: normalised accuracy = merged accuracy / own-expert accuracy on the same split,
  averaged over tasks; also mean absolute accuracy and the minimum normalised accuracy.
* Precision: fp16 autocast for all forwards (max |fp16 - fp32| over tasks <= 0.19 pt).
