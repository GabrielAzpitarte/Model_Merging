# Condition-number merging: side-track report (aa/side)

*Session azpitarte-aa, 2026-10-18. Companion to master's main pilot (session azpitarte-5f), which owns the
full-FT rule sweeps, the pairs and subsets, and the full-FT test freeze. This track ran the Stage-1 synthetic
checks, the per-layer diagnostics (gradient preference and the oracle allocation), the shared-target regime,
the KnOTS LoRA bank, and the mechanism stress tests.*

Information budgets: **A** weights only; **B** pooled unlabeled calibration inputs without task IDs;
**D** labeled dev data (oracle or diagnostic only). The test split is used only for frozen configurations.

Labels: **ESTABLISHED** (run completed); **TENTATIVE** (pattern plus its settling test);
**DEDUCED** (see the [math note](MATH_kappa_of_finetuned_weights.md)); **PENDING**.

## 0. The hypothesis and its literal interpretation

λ_i^ℓ = softmax_i(−β κ₂(W_i^ℓ)), and W = W0 + α Σ λ_i Δ_i; α = 1 is the professor's normalized average.

Conventions, shared with master:
- 73 linear maps: 72 attention/MLP weights plus the patch embedding.
- Biases are tied to their weight; LayerNorm and the position/class embeddings get the prior.
- κ₂ = σ1/σq in fp64.
- β = b / (median over maps of the across-expert score std), b ∈ {0.25, …, 128, ∞}.

## 1. Answers in one place

| question | answer | evidence |
|---|---|---|
| Does the exact rule help on full-FT CLIP ViT-B/32 (8 tasks)? | **No.** It equals uniform and the shuffled-score null. | Master: dev best 77.74 = shuffled 77.74 ± 0.19, uniform 77.65. Frozen TEST: κ 76.61 vs uniform 76.68 (normalized). |
| Does κ predict which expert deserves weight in a layer? | **No.** | Within-map ρ(gradient preference, κ) = +0.045; oracle λ vs κ −0.02…+0.02; layer-specific component ≈ 0 or slightly reversed (§2). |
| What does κ₂(W_i) measure? | 1/σ_min of a near-singular base map: one coordinate (u_qᵀΔv_q) of the update. | Within-layer ρ(κ, σ_min) = −0.91 (full-FT), −1.00 (LoRA); math note §1. |
| Can a functionally null edit swing the rule? | **Yes.** | Tail stress (§5): an edit of 0.5% of ‖Δ‖ (no accuracy change) moves λ_expert from 0.13 to 0.45–0.69 and costs up to 8 pt. |
| Is there headroom in the rule's own family? | **Large, all per-layer.** | Oracle (D): full-FT +8 pt dev (test 77.2); LoRA +12 pt (test 64.1); global per-expert weights ≤ +0.4 pt (§3). |
| Where should a κ-decreasing rule work in principle? | Noisy estimates of one target (review §3.3). | The literal rule is **anti-inverse-variance** there: noise improves conditioning (§4). |
| Is there a conditioning-connected signal that works? | The update spectrum's **shape**, under isotropic noise only. | It reaches the inverse-variance oracle even with norms matched (+9 pt), but fails on real same-task variants (§4). |
| LoRA? | Small, tentative positive for hard κ selection, below the best single expert. | §6. |

## 2. Per-layer diagnostics on full-FT: ESTABLISHED (budget D targets)

Targets:
- d(mean dev CE)/dλ_i^p at the uniform merge, at α = 1 and α = 2.4;
- the oracle λ (layerwise softmax plus α, fitted on dev labels, 2 seeds; log-λ Spearman 0.98 between seeds).

Tables: `results/aa/score_vs_targets.md`, `results/aa/grad_pref/`, `results/aa/oracle_lambda/`.

| score | within-map ρ vs gradient pref, α=1 | within-map ρ vs oracle λ | within-expert (layer-specific) |
|---|---|---|---|
| κ₂(W) (literal) | +0.045 | −0.012 | +0.095 (p 0.03) vs gradient; ≈ 0 vs oracle |
| κ(W C^{1/2}) (budget B) | +0.035 | −0.073 | ≈ 0 (p ≥ 0.14) |
| update spread (stable / entropy rank) | −0.43 / −0.48 | −0.10 | ≈ 0; all of it is between-expert (expert identity, ±0.8) |
| functional energy tr(ΔCΔᵀ) (budget B) | +0.33 | +0.07 | +0.12 to +0.16 (p ≤ 0.007), the only per-layer predictor |

Two independent κ implementations (this track's NumPy and master's torch) agree to 1e-13 median.

## 3. The oracle allocation: ESTABLISHED (budget D, absolute accuracy)

| bank | uniform (best α) | global oracle (λ per expert + α) | layerwise oracle, α = 1 | layerwise oracle + α |
|---|---|---|---|---|
| full-FT, dev / test | 70.3 / – | 70.6 / 69.8 | 75.0 / 73.9 | 78.7 / 77.2 |
| LoRA r16, dev / test | 52.3 / 51.2 | 52.7 / 52.4 | 62.3 / 61.7 | 64.8 / 64.1 |

- In full-FT the oracle makes attention maps nearly one expert per map (n_eff 1–2.5) with the same assignment
  across seeds, while the MLP maps stay mixed. This is consistent with master's finding that concentrating
  the MLP maps (κ_z) is harmful.
- κ is uncorrelated with the oracle in both banks (LoRA within-map ρ +0.03).

## 4. The shared-target regime: ESTABLISHED (own-task dev; same-task soups)

Pools:
- **A**: 4 noisy copies of each FusionBench expert, noise 1/4/8/16 × rms(Δ) per map;
- **A-nm**: the same copies, each update rescaled to the clean norm per map (norm rules blind);
- **B**: 13 real LoRA variants per task (Alexandre's banks: seeds, logit scales, label smoothing, step budgets).

| rule | A | A-nm | B (real variants) |
|---|---|---|---|
| uniform | 89.8 | 81.3 | 91.0 |
| oracle (inverse variance; quality softmax for B) | 90.5 | 90.1 | 91.6 |
| literal κ, b = 1 / 16 / ∞ | 89.4 / 88.8 / **58.8** | 81.2 / 81.2 / 74.0 | 91.0 / 90.4 / 88.3 |
| tuning-free GD/MP inverse variance λ ∝ σ̂^-2 | 90.5 | 83.9 | **90.1** |
| update shape at b = 4 (stable rank / noise share / bulk median) | 90.5 | **90.3–90.4** | ≤ 91.1 (best b) |
| update norm, b = 4 | 90.5 | 78.9 | ≈ 91.0 |

Findings:
- **The literal rule is anti-inverse-variance under estimation noise**, as smoothed analysis predicts (math
  note §3). κ falls with noise in about 70% of maps for all experts, and its argmin selects the noisiest copy.
- **The spectral shape of the update** carries noise information beyond magnitude: it recovers the oracle
  with norms matched, +9 pt over uniform.
- The tuning-free GD rule (the peer's `gd_invvar`, identical to my `mp_invvar` per map) is right when the
  copies share one signal scale (A). It is too flat when scales differ (A-nm).
- **On real recipe variation everything is ≈ uniform**, and the noise rule hurts (90.1, 0/8 tasks above
  uniform). Variant quality correlates with update spread at +0.5, so diffuse means better, and with mean
  log κ inconsistently (−0.31 … +0.60). The isotropic-noise model does not describe real checkpoint
  variation, and the uniform soup is already near the D-oracle (+0.6).
- PENDING: frozen-rule TEST evaluation (`scripts/aa_soup_test_freeze.py`).

## 5. Weak-tail stress test: ESTABLISHED (`results/aa/tail_stress_*.json`)

Edit: in the 47 maps with κ0 > 1e3, floor one expert's singular values at 10 σ_min. The edit is 0.47% of
‖Δ‖ (median), and the expert's accuracy is unchanged (≤ 0.08 pt on every task).

| literal rule | λ_expert before → after (attention maps) | merged dev accuracy before → after |
|---|---|---|
| SVHN edited, b = 4, α = 1.6 | 0.13 → 0.45 (0.14 → 0.62) | 70.0 → 68.7 |
| SVHN, b = 16 | 0.16 → 0.60 (0.84) | 69.6 → 65.0 |
| SVHN, b = ∞ | 0.22 → 0.69 | 50.6 → 42.2 |
| SUN397 edited, b = 4 / 16 / ∞ | 0.11 → 0.40 / 0.53 / 0.52 | 70.0 → 68.4; 69.6 → 66.4; 50.6 → 48.6 |

The rule's coefficients are controlled by function-irrelevant spectral tails.

## 6. LoRA (KnOTS r16): checks ESTABLISHED, κ-selection effect TENTATIVE

Checks:
- The QR-core spectrum is exact (9e-10).
- Rank-r interlacing holds, which gives the floor κ₂(W0 + BA) ≥ σ_{r+1}/σ_{q−r} of W0.
- Factor κ changes by about 5000× under an exact gauge change.
- 5/5 merge-correctness tests pass (`tests/test_aa_lora.py`).
- κ₊(Δ; τ = 0.1) is capped at 10 (80% of updates truncated), so it is a threshold artifact.
- Bank caveat: the KnOTS SUN397 adapter is weak.

**Dev sweep:** κ best 54.3 (b = 128, n_eff 1.35, near per-map selection). Free shuffles best 52.5–53.2,
uniform 52.3, best single expert 54.9.

**20-seed null (dev):** argmin-κ selection sits at the 95th percentile of count-preserving and of free random
per-map selections at α 0.8 and 1.0 (p ≈ 0.05). It is also robust to α, where random selections are not.

**Frozen TEST:**

| config | test |
|---|---|
| best single expert | 54.4 |
| κ b128 (rank-16 compressed: 53.2) | 53.3 |
| κ argmin | 52.9 |
| MetaGPT | 52.2 |
| uniform (rank-16 compressed: 50.2) | 51.2 |
| shuffles at κ's configuration | 46.8 / 51.3 / 50.2 |

**Mechanisms rejected:**
- expert frequency (global-frequency rule 52.2);
- functional gentleness (argmin tr(ΔCΔᵀ) gives 52.1–52.5, 19% agreement);
- bulk base-subspace overlap (ρ ≈ 0);
- null-space occupancy for k = 4–32 (ρ ≈ 0, agreement at chance; the inputs visit V0_low at 80–90% of the
  isotropic rate).

In LoRA, κ = σ_min only (argmin-κ = argmax-σ_min in 100% of maps).

PENDING (settling test): replication on 3 independent 8-task LoRA banks (`scripts/aa_lora_replicate_banks.py`).

## 7. Variant ledger

| variant | problem addressed | assumption | discriminating experiment | closest prior | status |
|---|---|---|---|---|---|
| literal κ₂(W), layerwise | the professor's rule | κ tracks reliability | per-layer targets; shuffles; noisy soups; tail stress | SADA, CT, AMM (exp weights) | **DEAD END** (full-FT); LoRA hard selection TENTATIVE |
| log κ (power law) | κ spans 1e1–1e8 | variance route | same | review eq. 22 | dead with κ; collapses faster under noise |
| κ(W C^{1/2}) (budget B) | κ dominated by directions the data may not excite | pooled C is representative | per-layer targets | PAVE, ESM, NSC | **FAILED** (blind) |
| update spread or noise share | κ(W) is a σ_min probe | spike/bulk model | noisy, norm-matched and real soups | STAR, TSV, MonoSoup | SUPPORTED under isotropic noise; **FAILS on real variants** |
| GD/MP inverse variance (tuning-free) | free temperature | common signal plus independent bulk noise | same | Model Stock, Fisher merging | same as above; too flat when signal scales differ |
| functional energy, + sign | where experts matter | pooled C | within-expert split | ESM, MetaGPT | weak, full-FT only (LoRA sign reversed) |
| factor κ (LoRA) | κ₂(BA) = ∞ | a canonical gauge | gauge test | — | **INVALID** |

## 8. What contribution remains possible

1. **A mechanism-level negative result** for weight-conditioning merge rules:
   - κ₂ of a fine-tuned weight is a probe of the base's weakest singular direction;
   - it is manipulable by function-null edits;
   - it is anti-inverse-variance under estimation noise (smoothed analysis);
   - for LoRA it is pinned by interlacing and blind to the oracle.
2. **A regime map:**

   | regime | exact rule | update-spectrum shape | real checkpoints |
   |---|---|---|---|
   | shared target | — | works under isotropic noise | does not transfer |
   | distinct tasks | uninformative | expert identity | — |

3. **Where the merging headroom actually lies:** per-layer allocation (+8 / +12 pt), concentrated in
   attention-map routing, which no spectral statistic recovers. This points to data-dependent routing /
   selection research, outside conditioning.
4. Possibly a narrow LoRA hard-selection effect of the literal rule, if the cross-bank replication holds.

Novelty check (2026-10-18): no prior paper uses Marchenko–Pastur / Gavish–Donoho noise estimates to weight
checkpoints. The closest are Model Stock (ECCV'24), MonoSoup (arXiv 2602.09689), Fisher merging, and
TSV / STAR. The negative real-variant result limits the practical value of that direction anyway.
