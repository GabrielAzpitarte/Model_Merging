# What κ₂ of a fine-tuned weight measures, and when a κ-decreasing weight can be justified

*aa/side track, 2026-10-18. Derivations are this note's own unless cited. Each claim is tagged
[proved] (exact under its stated assumptions), [checked] (verified numerically in
`scripts/aa_synthetic_checks.py`, `scripts/aa_lora_spectra.py` or the real-checkpoint scripts), or [observed]
(a measurement on the FusionBench / KnOTS banks).*

Notation: expert *i* shares the base layer `W0 ∈ R^{m×n}` and has update `Δ_i`, so `W_i = W0 + Δ_i`.
Write `q = min(m, n)` and `σ_1 ≥ … ≥ σ_q`, with singular pairs `(u_j, v_j)` of `W0`. The literal score is
`κ_i = σ_1(W_i)/σ_q(W_i)`, and the literal rule is `λ_i ∝ exp(−β κ_i)`.

## 1. The literal score is a probe of the base's weakest direction

**Weyl bounds [proved]:** `|σ_j(W0+Δ) − σ_j(W0)| ≤ ‖Δ‖₂` for every `j`.

In all 73 CLIP linear maps, `‖Δ_i‖₂ ≪ σ_1(W0)` [observed], so `σ_1(W_i) ≈ σ_1(W0)` for every expert. The
numerator of κ is essentially shared, so the across-expert variation of κ comes from the denominator
`σ_q(W_i)`. Measured within-layer Spearman(κ_i, σ_q(W_i)) = −0.91 [observed].

**Perturbative regime [proved].** If `‖Δ‖₂` is small relative to the gaps around `σ_1` and `σ_q`, standard
first-order perturbation of simple singular values gives `σ_j(W0+Δ) = σ_j + u_jᵀΔv_j + O(‖Δ‖₂²/gap_j)`.
Hence

    log κ_i − log κ_0  ≈  u_1ᵀΔ_i v_1 / σ_1  −  u_qᵀΔ_i v_q / σ_q .

The literal score is therefore an affine function of two scalar coordinates of the update, its (1,1) and
(q,q) entries in the base's singular basis. Since `σ_q ≪ σ_1`, the second term dominates.

So, within a layer, ranking experts by κ ranks them by how much their update lifts the base's smallest
singular direction. For an update that is not specially aligned with `(u_q, v_q)`, that coordinate carries
a fraction of order `1/(mn)` of `‖Δ_i‖_F²`.

This regime holds in the MLP maps. There κ0 is about 15–60, experts' κ differs from κ0 by about 0.1%, and
the rank-2 probe explains the shifts (median r 0.94–0.98) [observed, master spectra].

**Non-perturbative regime [proved].** If `σ_q(W0) ≲ ‖Δ_i‖₂`, the base is near-singular relative to the
update. Let `N_ε` be the span of the right singular vectors of `W0` with `σ ≤ ε`. By the min-max
characterisation,

    σ_q(W0+Δ) = min_{‖x‖=1} ‖(W0+Δ)x‖  ≤  min_{x∈N_ε,‖x‖=1} ‖Δx‖ + ε .

So κ_i is set by how Δ_i acts on the base's near-null space. That is again a small, specific corner of
the update. This regime holds in the attention maps: κ0 ranges from 1e3 to 1.04e8, fine-tuning moves κ by
about 100%, and the first-order fit fails [observed].

In 27 of the 73 base maps, κ0 is near the float32 resolution. In L0/L1 k_proj and L0 q_proj, fp32-SVD
noise is 50–90% of the across-expert κ spread [observed, master], so there the score is partly rounding.

**Consequence for one global β [proved + observed].** The spread of κ within a layer is about 0.1% of κ0 in
the MLP maps and order κ0 (up to 1e8) in the attention maps. So a single β leaves the MLP maps at the prior
while concentrating the attention maps. The literal rule's behaviour is therefore set entirely by a few
q/k maps: 100% of its first-order effect on dev loss comes from 5 maps [observed].

## 2. LoRA: the base pins the full-weight condition number

**Rank-r interlacing [proved; checked: 0 violations over 384 (map, expert) pairs].** If rank(Δ) ≤ r, then
`σ_{j+r}(W0) ≤ σ_j(W0+Δ) ≤ σ_{j−r}(W0)`. Therefore

    κ₂(W0 + BA)  ≥  σ_{r+1}(W0) / σ_{q−r}(W0) ,

which is a floor set by the base alone.

A rank-16 adapter cannot lift σ_q above the base's (q−16)-th singular value.

**The ordinary κ₂ of the update is infinite** for r < q. The QR-core spectrum is exact [checked, 9e-10].

**Factor condition numbers are gauge dependent [proved; checked].** `(BR)(R⁻¹A) = BA` for every invertible R.
A gauge with cond(R) = 1e4 changes κ(A) by a median factor of 5468× and κ(B) by 2444×, while the product and
its spectrum are unchanged to 4e-13.

## 3. When can a κ-decreasing weight be right? The two regimes of the review

**(a) Distinct tasks (the multi-task merge).** No general monotone relation exists (review Fig. 1-left).

[observed] On the 8 FusionBench experts, d(mean dev CE)/dλ_i^p at the uniform merge shows no within-layer
association with κ (mean Spearman +0.045). After removing expert identity, the layer-specific association is
+0.095 (p 0.017), a small effect in the reverse direction of the rule.

The pairs sweep (master) finds expert worth explained by an expert-level model (R² 0.91), uncorrelated with
mean log κ (ρ −0.02).

**(b) Noisy estimates of one target (soups).** Inverse-variance weights are optimal [proved, review eq. 13].
A decreasing rule helps iff the score increases with the estimation error.

*Smoothed analysis [proved; Sankar–Spielman–Teng 2006]:* for Gaussian `G` with entry std `s`,
`P(σ_min(A + sG) ≤ ε) ≤ C ε √n / s`. Hence `σ_min(A + sG) ≳ s/√n` with high probability, and

    κ(A + sG) ≲ √n (‖A‖₂ + 2s√n) / s ,

which decreases in s while `σ_min(A) ≪ s/√n ≪ ‖A‖₂/√n`.

So for near-singular layers, **more estimation noise gives a better-conditioned weight**. The literal rule
then produces anti-inverse-variance weights.

- [checked] A synthetic 128×128 matrix with σ_min/σ_max = 1e-6 goes from κ 6.7e6 to 1.5e3 as the noise
  goes from 1e-6 to 1e-2.
- [observed] Real CLIP base matrices: L0 k_proj goes from 1.04e8 to 9.9e3 at noise std 1e-2. Even the
  tall, well-conditioned L6 fc1 goes from 14.4 to 7.5.
- [observed] Noisy copies of real FusionBench experts: κ falls with the noise level in 75% of the maps.
  The literal rule puts its largest weight on the noisiest copy, and its global variant puts 0.73–1.0 there.

**What does track noise: the update's spectrum [proved for the spiked model].** Take
`Δ_j = S + s_j N` with low-rank signal S and `N` having iid unit-variance entries. The bulk of the spectrum
sits in `[s_j(√m−√n), s_j(√m+√n)]` (Bai–Yin / Marchenko–Pastur), so bulk-sensitive statistics are monotone
in s_j. These include the stable rank, the entropy rank and the median singular value.

A plug-in estimate `ŝ_j = median σ(Δ_j) / μ_MP` (cf. Gavish–Donoho 2014) gives tuning-free inverse-variance
weights `λ_j ∝ ŝ_j⁻²`. This is a power law in a spectral statistic of Δ, not `exp(−β κ(W))`. Note the
direction: a sharper, more ill-conditioned update spectrum gets more weight.

[observed] The stable rank of the noisy copies' updates tracks the noise level with rank correlation +1.00.

## 4. What this leaves for "conditioning-guided merging"

1. The literal statistic (κ₂ of the full weight) reads one or two coordinates of the update in the base's
   singular frame (perturbative layers), or its action on the base's near-null space (near-singular layers).
   Neither is a summary of the update's function. Under estimation noise it moves in the wrong direction.
2. A conditioning-type statistic of the **update spectrum's tail/bulk** is the principled uncertainty proxy
   (inverse variance, power law), in the shared-target regime only.
3. In the multi-task regime the dominant structure is an **expert-level worth** that depends on the global
   scale α (a saturation effect: a task's accuracy is a concave function of its effective weight αλ_i). No
   per-layer conditioning score recovers it on these banks. The oracle shows large headroom inside the rule's
   own family: +8 pt absolute on dev.
4. Conditioning enters directly only where the merge **solves an inverse problem** (RegMean/MaTS/WUDI/SWUDI;
   review §3.6). There, the conditioning of the aggregate Gram, not of any expert's weight, controls error
   amplification.
