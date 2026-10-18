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
| LoRA? | **No reliable effect.** Hard κ selection beat random per-map selection on 2 of 4 independent 8-task banks and was at or below chance on the other 2. It never beat the best single expert. | §6 |

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

### 2b. Check of master's dev-best rule `logkappa_q90_rev` (requested by the peer)

The rule weights maps by the tail-robust bulk ratio σ1/σ_{⌈0.9q⌉} of the full weight, giving more weight to
higher values. Master's numbers: dev-best of about 45 rules, frozen full-FT TEST +0.47 [+0.24, +0.70] over
tuned uniform.

Per-map association with log κ_q90 (positive = supports the rule):

| target | within-map ρ (p) | within-expert (p) |
|---|---|---|
| gradient pref, α = 1 | +0.076 (0.09) | +0.070 (0.11) |
| gradient pref, α = 2.4 | +0.076 (0.10); attention +0.18, MLP −0.12 | +0.059 (0.19) |
| oracle layerwise + α (2 seeds) | +0.09 (0.046) | +0.075 (≈ 0.08) |
| oracle, α = 1 | +0.03 (0.48) | — |

The median ratio κ_q50 carries nothing.

Reading: a small, consistently positive, attention-concentrated signal, in the REVERSE direction of the
professor's hypothesis. It is TENTATIVE because it was selected among about 45 rules; master's pre-registered
ViT-B/16 transfer is the settling test.

**LoRA transfer (this track; configs fixed a priori; `results/aa/lora/q90_transfer.jsonl`):** strongly
HARMFUL on every LoRA bank tested.
- KnOTS release: −4.1 to −17.6 pt vs uniform.
- lora_c5a: −9.3 to −16.7.
- lora_c5a_seed1: −8.7 to −12.5.
- vit20: +0.3 / −0.3 at b = 1, −1.7 / −2.9 at b = 4 (inside its shuffle range).
- It is worse than its own matched shuffles on every KnOTS-recipe bank.

Mechanism: for low-rank updates, κ_q90(W0 + BA) varies through the bulk value σ_{0.9q} (within-map ρ −0.85),
not through σ1 (+0.22). The rule therefore rewards adapters that shrink the base's bulk spectrum, which here
are destructive adapters.

**Settled by master's pre-registered ViT-B/16 transfer** (B/32-frozen configs, test, paired bootstrap vs
tuned uniform at 82.01):
- `logkappa_q90_rev` −1.01, below all 5 matched-config shuffles (−0.95 … +0.32): **FAILS**. The B/32 +0.47
  was a selection artifact.
- Literal κ −0.11, inside its shuffles (−0.76 … +0.24).
- `global:logkappa` +0.02; reversed κ −2.15; MetaGPT −0.83.

Bound on global rules: the global oracle (one weight per expert) reaches 69.8 absolute on test vs 69.3 for
tuned uniform. So no global rule can gain more than about +0.5 there; `global:logkappa`'s +0.27 lies inside its
shuffled-null range.

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
- **Frozen TEST** (rules fixed before any test look; `results/aa/soup/test_freeze_summary.json`):

  | rule | A | A-nm | B |
  |---|---|---|---|
  | uniform | 89.4 | 80.6 | 90.6 |
  | literal κ, b = 1 / 4 | 89.0 / 88.7 (0/8 tasks above uniform) | 80.5 / 80.5 | 90.5 / 90.5 |
  | update stable rank, b = 4 | 90.3 (7/8) | 90.0 (8/8) | 88.2 (0/8) |
  | GD/MP inverse variance | 90.3 | 83.3 | 89.6 (0/8) |
  | oracle | 90.3 | 89.7 | 89.9 (best single by dev) |

  This replicates dev exactly.

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

## 6. LoRA: checks ESTABLISHED; the κ-selection effect did NOT replicate

Checks:
- The QR-core spectrum is exact (9e-10).
- Rank-r interlacing holds, which gives the floor κ₂(W0 + BA) ≥ σ_{r+1}/σ_{q−r} of W0.
- Factor κ changes by about 5000× under an exact gauge change.
- 5/5 merge-correctness tests pass (`tests/test_aa_lora.py`).
- κ₊(Δ; τ = 0.1) is capped at 10 (80% of updates truncated), so it is a threshold artifact.
- In LoRA, κ is σ_min only (argmin-κ = argmax-σ_min in 100% of maps).

KnOTS release:
- Dev: κ best 54.3 (b = 128, near per-map selection); uniform 52.3; best single expert 54.9.
- Argmin-κ is at the 95th percentile of a 20-seed random-selection null (count-preserving and free; p ≈ 0.05).
- Frozen TEST: κ 53.3, uniform 51.2 (rank-16 compressed: κ 53.2, uniform 50.2), shuffles 46.8 / 51.3 / 50.2,
  best single expert 54.4.

**Cross-bank replication** (`results/aa/lora/replicate_banks.jsonl`; 20 free random per-map selections per
α; dev):

| bank | argmin-κ percentile, α 0.8 / 1.0 | κ b128 vs best uniform |
|---|---|---|
| KnOTS release | 95 / 95 | 54.3 vs 52.3 |
| KnOTS recipe, seed 0 (lora_c5a) | 95 / 100 | 55.3 vs 53.5 |
| KnOTS recipe, seed 1 (lora_c5a_seed1) | 55 / 45 | 51.8 vs 54.1 |
| vit20, 2000 steps (lora) | 20 / 15 | 58.0 vs 64.6 |

**Mechanisms tested and rejected:**
- expert frequency;
- functional gentleness;
- bulk base-subspace overlap;
- null-space occupancy (k = 4–32; the inputs visit V0_low at 80–90% of the isotropic rate);
- singularity avoidance: across 40 random selections, accuracy does not track the number of σ_min-decreasing
  picks (ρ +0.14, p 0.37).

Verdict: **NOT REPLICATED.** The KnOTS-bank advantage is a favourable draw from a high-variance selection
lottery. On the well-behaved vit20 bank, any hard per-map selection loses badly to averaging.

Per-map marginal values (`results/aa/lora/single_map_swap_summary.json`; uniform base, α = 1; each of the
48 × 8 single-map hand-overs evaluated on dev):
- The argmin-κ choice ranks 4.27 of 8 on average (random 4.5). It is the best choice in 10% of maps and the
  worst in 10%.
- Within-map ρ(−κ, gain) = −0.08.
- κ does not identify good per-map owners even one map at a time.

## 7. Variant ledger

| variant | problem addressed | assumption | discriminating experiment | closest prior | status |
|---|---|---|---|---|---|
| literal κ₂(W), layerwise | the professor's rule | κ tracks reliability | per-layer targets; shuffles; noisy soups; tail stress; cross-bank LoRA replication | SADA, CT, AMM (exp weights) | **DEAD END** (full-FT and LoRA; the LoRA selection effect did not replicate, 2/4 banks) |
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
   attention-map routing. No spectral statistic recovers it in a way that transfers: the best per-map signal,
   master's reversed bulk ratio, failed the pre-registered ViT-B/16 transfer and the LoRA transfer. This points
   to data-dependent routing / selection research, outside conditioning.
4. (Closed) The apparent LoRA hard-selection effect did not replicate across banks (2/4).
5. (Closed) `logkappa_q90_rev`: B/32 dev-best of about 45 rules with a small per-map signal. It failed both
   the ViT-B/16 transfer (−1.01, below all matched shuffles) and the LoRA transfer (−4 to −18 pt).

**Final status (2026-10-18):** no conditioning-based weighting survives on full-FT (ViT-B/32, ViT-B/16) or on
LoRA (4 banks), against matched shuffled-score nulls and frozen test evaluation.

Novelty check (2026-10-18): no prior paper uses Marchenko–Pastur / Gavish–Donoho noise estimates to weight
checkpoints. The closest are Model Stock (ECCV'24), MonoSoup (arXiv 2602.09689), Fisher merging, and
TSV / STAR. The negative real-variant result limits the practical value of that direction anyway.

## 9. Phase 2: "the instability lies in the solve" (review hypothesis row 3), RegMean on LoRA, IN PROGRESS

Conditioning enters a merge directly when the merger inverts a Gram matrix (review §3.6). Per adapted map:
D = (Σ D_i G̃_i)(Σ G̃_i + ρI)^{-1}, with RegMean's off-diagonal shrinkage G̃ = aG + (1−a)diag(G). The
κ-capped ridge picks the smallest ρ with κ(ΣG̃ + ρI) ≤ κ_target; the ridge targets zero or the uniform
average, as in master's `regmean_solve`. My independent implementation matches master's to 5e-8.

Gram budgets:
- **C**: task-separated; expert i on task i's calibration split.
- **Bp**: the pooled calibration set through each expert; label-free and task-ID-free.
- **B0**: pooled through the base; RegMean then must equal averaging.

Solves:
- full input space (dense output);
- exact adapter subspace, D = M Pᵀ with P spanning all Aᵢᵀ (output rank ≤ 128).

KnOTS bank, dev (`results/aa/lora/regmean_summary.md`):

| budget, solve | plain | shrink a = 0.95 / 0.9 | κ-cap 1e4 / 1e3 / 1e2 (ridge → zero) |
|---|---|---|---|
| C, full | 65.3, **4.1 at α = 1.3** | 64.8 / 64.3 | **65.5** / 64.6 / 61.6 (α 1.3: 50.9) |
| C, subspace | 57.9 | 57.8 / 57.4 | 57.9 / 57.9 / 57.7 |
| Bp, full | 51.8 | 51.7 / 51.7 | 51.8 / 51.6 / 51.6 |
| B0 | 52.3 = averaging (sanity) | | |

Uniform is 52.3 (α 0.8); the per-layer λ oracle was 64.8.

Reading:
- The plain solve is unstable: median κ(G) is 1e5, some maps have singular Grams, and it collapses when the
  update is scaled.
- Capping κ at 1e4 removes the collapse and slightly beats fixed shrinkage. This is a conditioning mechanism
  of the merge system (review §3.6), not of any expert's weight.
- It needs budget C. The label-free pooled Grams carry no task information for rank-16 adapters.
- Part of the gain is extra rank (the subspace solve gives 57.9).

**Calibration-size test (C budget, full solve):**

| images per task | plain | shrink a = 0.95 | κ-cap 1e4 | median κ(G) |
|---|---|---|---|---|
| 512 | 65.3 (collapses at α = 1.3) | 64.8 | **65.5** | 1.05e5 |
| 16 | **4.1 at every α** | 61.7 | **62.2** | 2.2e5 |

Noisier Gram estimates are worse conditioned, and the plain solve then fails completely. Conditioning-based
regularization rescues it and stays slightly ahead of RegMean's fixed shrinkage (+0.5 to +0.7). This
SUPPORTS review hypothesis row 3: weak Gram eigen-directions amplify estimation noise, and stable
regularization fixes it.

The margin over existing shrinkage is small, and spectral filtering of the merge solve is close prior work
(SWUDI [03]). So any novelty claim would be narrow.

**All LoRA settings, matched single configs** (κ-cap 1e4 → zero vs shrink 0.95; C budget, full solve, best
α):

| setting | plain | shrink 0.95 | κ-cap 1e4 | difference |
|---|---|---|---|---|
| KnOTS, N = 512 | 65.3 (4.1 at α 1.3) | 64.8 | 65.5 | +0.7 |
| KnOTS, N = 64 | 4.1 | 63.9 | 64.7 | +0.8 |
| KnOTS, N = 16 (3 calibration subsets) | 4.1 | 61.7 / 63.2 / 63.8 | 62.2 / 64.1 / 64.5 | +0.5 / +0.9 / +0.7 |
| lora_c5a, N = 512 | 4.1 | 65.1 | 65.6 | +0.5 |
| vit20, N = 512 (uniform 67.5) | 4.1 | 82.4 | 82.6 | +0.2 |

- Positive in 7/7 settings, mean +0.6; the settings share banks, so they are not independent.
- The edge does not grow as N shrinks.
- Master's full-FT result has the same sign but is smaller: +0.2 to +0.3 at N = 16 / 64 / 512 over 3 seeds.
- Candidate mechanism: RegMean's off-diagonal shrinkage depends on the coordinate basis, whereas the
  κ-capped isotropic ridge is rotation-invariant and acts only on the weak eigen-directions.
- RegMean with C-budget Grams is by far the strongest merge in this study: +12 to +15 pt over uniform on
  LoRA, above the per-layer λ oracle. That is an established method's strength, not the hypothesis's.

PENDING:
- transfer to lora_c5a and vit20;
- rank-16 compressed comparison;
- frozen test.
