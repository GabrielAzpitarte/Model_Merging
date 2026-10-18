# Condition-number merging: side-track report (aa/side)

*Session azpitarte-aa, 2026-10-18. Companion to master's main pilot (session azpitarte-5f), which owns the
full-FT rule sweeps, the pairs and subsets, and the test freeze. This track ran the Stage-1 synthetic checks,
the per-layer diagnostics (gradient preference and the oracle allocation), the shared-target regime, and the
KnOTS LoRA bank. Information budgets: **A** weights only; **B** pooled unlabeled calibration inputs, no task
IDs; **C** task-separated unlabeled; **D** labeled dev (oracle or diagnostic only).*

Status legend: **ESTABLISHED** (run completed, numbers below), **TENTATIVE** (pattern seen, settling test
named), **DEDUCED** (mathematics; see [MATH note](MATH_kappa_of_finetuned_weights.md)), **PENDING**.

## 0. The hypothesis under test

The professor's rule: λ_i^ℓ = softmax_i(−β κ₂(W_i^ℓ)), and W_merge = Σ_i λ_i W_i, equivalently
W0 + α Σ λ_i Δ_i with α = 1. "Larger condition number → lower weight."

Conventions shared with master:
- 73 linear maps: the 72 attention/MLP weights plus the patch embedding.
- Biases are tied to their weight; LayerNorm and position/class embeddings get the prior.
- κ₂ = σ1/σq is computed in fp64 from the stored fp32 weights.

## 1. Synthetic checks: ESTABLISHED (`scripts/aa_synthetic_checks.py`, all pass)

C1–C10 numerically verify the review's eqs. 8–37:
- inverse variance, and the controlled least-squares designs (κ-decreasing weights help there);
- correlation and duplication effects; orientation effects;
- distinct tasks, where the κ rule loses;
- heterogeneous quadratics, where no scalar mixture reaches the optimum;
- entropy-regularised softmax; the regression merge with its ridge and perturbation bound;
- the LoRA gauge and the exactness of the QR-core spectrum;
- coordinate dependence of κ.

New check C11: Gaussian weight noise LOWERS κ₂ of near-singular matrices (smoothed analysis). It does so on
every real CLIP base matrix tested: L0 k_proj goes from 1.04e8 to 9.9e3 at noise std 1e-2.

## 2. The literal κ carries no per-layer information on the full-FT bank: ESTABLISHED

Diagnostic targets (budget D):
- (i) d(mean dev CE)/dλ_i^p at the uniform merge, at α = 1 and α = 2.4 (`scripts/aa_grad_preference.py`);
- (ii) the oracle allocation: layerwise softmax per linear map plus global α, fitted on dev labels
  (`scripts/aa_oracle_lambda.py`).

| | value |
|---|---|
| within-layer Spearman(preference, κ), α=1 | +0.045 (wrong sign for the rule; perm p 0.31 two-sided) |
| first-order effect of the literal rule on dev CE | at the permutation-null median (p 0.48); 100 % carried by 5 huge-κ maps |
| layer-specific (within-expert) association, gradient α=1 | +0.095 (p 0.03): small, REVERSE direction |
| within-map Spearman(oracle λ, κ), all 4 fits | −0.02 … +0.02 |
| within-expert association with the oracle | ≈ 0 (p 0.4–0.8) |
| data-weighted κ(W C^{1/2}) (budget B) vs all targets | within-map −0.07 … +0.04, within-expert ≈ 0 (p ≥ 0.14) |

Supporting facts:
- κ₂(W_i) ≈ 1/σ_min(W_i) up to a shared constant (within-layer ρ −0.91).
- Two independent implementations agree: this track's NumPy/LAPACK and master's torch, κ median relative
  difference 1e-13.
- Master's rule sweep (normalized accuracy, dev): the literal rule's best is 77.74, against the
  shuffled-score null 77.74 ± 0.19 (5 seeds) and uniform 77.65.

## 3. Where the headroom actually is: ESTABLISHED (oracle, budget D)

| oracle (fit on dev labels) | α | dev acc | test acc |
|---|---|---|---|
| uniform λ, best α on dev | 1.6 | 70.28 | (master freeze) |
| global: one λ per expert + α (2 seeds) | 1.39 / 1.34 | 70.60 / 70.46 | 69.77 / 69.80 |
| layerwise, α = 1 (2 seeds) | 1 | 75.04 / 75.02 | 73.91 / 73.82 |
| layerwise + α (2 seeds) | 1.98 / 2.00 | 78.65 / 78.67 | 77.23 / 77.27 |

Absolute accuracy; the test split is untouched by the fit.

- **Reproducible:** log-λ Spearman across seeds is 0.98.
- **All the headroom is per-layer.** Expert-level weights are worth at most +0.3 pt.
- **Attention maps become nearly one expert per map** (n_eff 1.0–2.5), with an identical assignment across
  seeds. The MLP maps stay mixed (n_eff 2–6). This is consistent with master's finding that z-scored κ,
  which concentrates the MLP maps, is harmful.
- **What correlates with the oracle:**
  - Update-spread statistics (stable / entropy rank, budgets A and B): this is expert identity (between-expert
    ±0.8, within ≈ 0).
  - The only per-layer predictor is the functional update energy tr(ΔCΔᵀ) (budget B): within-expert +0.12 to
    +0.16, p ≤ 0.007. The effect is small. It is a magnitude statistic, not conditioning.

## 4. The shared-target regime (the review's favourable case): ESTABLISHED for controlled noise

Setup: same-task soups of 4 noisy copies of each FusionBench expert, with noise r × rms(Δ) per map. The
optimum is inverse variance (review eq. 13). Evaluation is on the own-task dev split.

- κ FALLS with noise in about 73 % of maps, for all 8 experts (C11 on real experts).
- Strong noise, r ∈ {1, 4, 8, 16}:

| rule | SUN397 | Cars |
|---|---|---|
| uniform | 71.8 | 76.1 |
| inverse-variance oracle | 74.5 | 78.2 |
| literal κ, b = 1 / 16 / ∞ | 69.9 / 67.9 / **2.7** | 76.9 / 74.7 / **11.4** |
| log κ, b = 1 | 57.7 | 65.4 |
| update stable rank (sharper → more weight), b ≥ 1 | 74.5 | 78.5 |
| update norm (smaller → more weight), b ≥ 4 | 74.5 | 78.4 |

The literal rule is anti-inverse-variance in exactly the regime the review identified as favourable. The
**update spectrum** is the right object there, and in the reversed direction: a sharper, more ill-conditioned
update is more reliable.

PENDING:
- the norm-matched noise variant (norm rules blind by construction), which tests spectral information beyond
  magnitude;
- real same-task LoRA soups (13 variants per task).

## 5. LoRA (KnOTS r16): checks ESTABLISHED, sweep PENDING

- The QR-core spectrum is exact (9e-10 vs dense SVD), and rank-r interlacing holds (0 violations), so
  κ₂(W0 + BA) ≥ σ_{r+1}(W0)/σ_{q−r}(W0).
- Factor κ is gauge dependent: ×5468 (A) and ×2444 (B) under a gauge with cond(R) = 1e4, while the product
  is unchanged to 2e-13.
- Merge-correctness tests: `tests/test_aa_lora.py`.
- Bank caveat: the KnOTS SUN397 adapter is weak (own task 65.2 vs zero-shot 64.4) and damages other tasks
  when applied alone, as recorded in the ICA project. The best single expert (EuroSAT, α = 0.6) is 54.9,
  above uniform at 52.3. Any LoRA "gain" must therefore be read against selection controls.
- Sweep in progress: literal κ best so far 54.3 at b = 128, n_eff 1.35 (per-map selection). The shuffled
  rows at high b are pending.

## 6. Variant ledger (this track)

| variant | observed problem it addresses | proposed explanation | assumptions | smallest discriminating experiment | closest prior | status |
|---|---|---|---|---|---|---|
| literal κ₂(W_i), layerwise softmax | the professor's rule | ill-conditioned experts are unreliable | κ tracks reliability | within-map Spearman vs gradient preference and oracle λ; shared-target noisy soups | (none exact); SADA, CT, AMM (exp weights) | **DEAD END**: blind per layer; anti-inverse-variance under noise |
| log κ (power law κ^-β) | raw κ spans 1e1–1e8 across maps | variance route λ ∝ κ^-2 (review eq. 22) | design-matrix model applies to W | same as above | review §3.3 | dead with κ: same ranks; collapses faster under noise |
| data-weighted κ(W C^{1/2}) (budget B) | κ is dominated by base directions the data may not excite | condition only on data-supported directions | pooled C is representative | score-vs-targets (§2) | PAVE, ESM, NSC | **FAILED**: equally blind (within ≈ 0) |
| update spread (stable / entropy rank, κ₊ of Δ) | κ(W) is a σ_min probe, not a property of the update | concentration = signal | the spike/bulk model | expert-identity decomposition | STAR, SADA, TSV, LARV | multi-task: expert identity only, sign flips with α; **shared target: SUPPORTED** (norm-matched pending) |
| MP / Gavish–Donoho inverse variance λ ∝ σ̂^-2 (tuning-free) | free temperature; κ moves the wrong way under noise | bulk of the update spectrum = noise level | copies = common signal + independent bulk noise | controlled noise (strong and norm-matched) and real variants | Model Stock (ECCV'24, geometric SNR across checkpoints), MonoSoup (2026, single-update spectral reweighting), Fisher merging (inverse variance via Fisher) | PENDING (A3, B-v2, test freeze) |
| functional update energy tr(ΔCΔᵀ), positive sign | where should an expert get weight? | experts matter where they act on data | pooled C | within-expert decomposition | ESM (activation importance), MetaGPT (norms) | weak per-layer signal (+0.12..0.16), full-FT only; LoRA sign differs; not conditioning |
| factor κ(A), κ(B) (LoRA) | LoRA κ₂(Δ) = ∞ | — | a canonical gauge exists | gauge test | — | **INVALID**: changes ×10³ under exact reparameterisation |
| κ₊(Δ; τ) (LoRA) | finite substitute for κ₂(Δ) | — | threshold choice | threshold sweep; rule sweep | — | τ = 0.1 caps κ₊ at 10 (80 % truncated); τ ≤ 1e-3 gives σ1/σ16; sweep PENDING |

## 7. Verdict so far (per the kill-only-proven-dead-ends standard)

The literal full-weight rule meets every DEAD END criterion:
1. **Accuracy-level evidence:** the master sweep on the full dev set (test freeze pending).
2. **A clear miss:** it equals the shuffled null; this is not a decimal-point miss.
3. **Registered variant plus repairs all fail:** log, z-score, strict, reversed, global and data-weighted
   variants.
4. **No bug:** two implementations agree to 1e-13; β = 0 reproduces uniform.
5. **Evidenced mechanism:** κ₂ is a σ_min probe; noise improves conditioning (smoothed analysis), so the rule
   is anti-inverse-variance in the shared-target regime.

What survives as a conditioning-connected direction is the update-spectrum noise proxy for same-target
pools: an inverse-variance power law, with ŝ estimated from the spectrum's bulk. The norm-matched test and
the real-variant soups are its settling tests.
