# Mathematical notes on the literal rule (deductions made here; numbers filled in from measurements)

Setting: experts share a base, `W_i = W0 + D_i`; SVD `W0 = U S V^T`, singular values
`s_1 >= ... >= s_q`, `kappa0 = s_1/s_q`. Labels follow docs/RESEARCH_LOG.md.

## N1. Two regimes for kappa2(W_i)  [deduction]

Weyl: `|s_j(W_i) - s_j(W0)| <= ||D_i||_2`.

* **(a) Perturbative regime**: `||D_i||_2` much smaller than `s_q(W0)` and than the end gaps
  `s_1 - s_2`, `s_{q-1} - s_q`. Then the expansion in N2 holds.
* **(b) Near-singular regime**: `s_q(W0) <~ ||D_i||_2`. Then `s_q(W_i)` is governed by the
  update's compression onto the base's weak singular subspaces, and generic (random-like)
  updates *raise* `s_q`: the smoothed-analysis effect (Sankar-Spielman-Teng), checked on CLIP
  base matrices by the peer session (C11). In this regime a lower `kappa2(W_i)` means the expert
  writes more energy into directions the base nearly annihilates, so the literal rule rewards
  larger, more diffuse updates.

## N2. In regime (a), the literal score is a rank-2 linear probe; the rule minimises merged conditioning  [deduction]

For simple `s_1`, `s_q`: `d s_1 = u_1^T D v_1`, `d s_q = u_q^T D v_q`, hence

    kappa2(W0 + D) = kappa0 + kappa0 <D, G0>_F + O(||D||^2),   G0 = u_1 v_1^T / s_1 - u_q v_q^T / s_q.

1. The score depends on `D_i` only through its two components along `u_1 v_1^T` and `u_q v_q^T`.
   Every other update direction, which is all but 2 of `m*n` directions, is invisible to it. It is
   sign-sensitive and is not a magnitude measure.
2. Because it is linear, for any simplex weights `kappa2(W0 + sum_i lambda_i D_i) ~= sum_i lambda_i kappa2(W_i)`.
   The softmax rule is the exact minimiser of `sum_i lambda_i c_i + tau KL(lambda || uniform)`
   (review eq. 23-24). So, to first order, **the literal rule is entropy-regularised minimisation
   of the merged layer's condition number** at alpha = 1 (and the ordering is the same for any
   alpha > 0). This gives the proposal a precise, testable meaning: "make each merged layer as
   well-conditioned as possible".
3. The bottom term dominates when `s_q << s_1`: `kappa0 * u_q^T D v_q / s_q = s_1 (u_q^T D v_q) / s_q^2`.

## N3. One beta cannot be moderate in every layer  [deduction + measurement]

The within-map score spread is about `kappa0 |<D_i - D_j, G0>|`, so it scales with `kappa0` and,
through the bottom term, with `1/s_q^2`. Across the 73 maps `kappa0` spans many orders of
magnitude (measured below), so with one global beta the literal rule is argmin-selection in
ill-conditioned maps and uniform in well-conditioned ones. Making temperatures comparable across
layers (`kappa_z`, `logkappa`) changes the method, so these get separate names.

## N4. The variance route yields an almost-uniform rule on a shared-base bank  [deduction + measurement]

If `lambda_i` is proportional to `1/v_i` and `v_i` is proportional to `kappa_i^2` (review eq. 22, large-kappa
limit), then `lambda_i` is proportional to `exp(-2 log kappa_i)`, the `logkappa` rule at beta = 2.
The relative weights are `lambda_i / lambda_j = (kappa_j / kappa_i)^2`, so a relative kappa spread `eps`
moves the weights by about `2 eps` from uniform.

## N5. The worst-case sensitivity argument points the other way  [deduction]

Merging perturbs expert i's map by `E_i = W_merge - W_i = sum_{j != i} lambda_j (D_j - D_i)` (alpha = 1),
and for full column rank

    ||E_i x|| / ||W_i x|| <= ||E_i||_2 / s_min(W_i) = kappa2(W_i) ||E_i||_2 / ||W_i||_2.

Higher-kappa experts are more fragile in the worst case, and `||E_i||` shrinks as `lambda_i` grows.
Equalising worst-case distortion therefore suggests *increasing* lambda with kappa, which is the
reversed rule. Two caveats apply. With a shared base, `kappa2(W_i) ~= kappa0` for every expert, so
this fragility belongs to the base rather than the expert. And the worst case is attained only along
`v_q`, so it is vacuous if the inputs carry little energy there (review eq. 10).

## N6. Concentration changes update scale at alpha = 1  [deduction + measurement]

For nearly orthogonal updates, `||sum lambda_i D_i||_F^2 ~= sum lambda_i^2 ||D_i||^2`, which grows from
`||D||^2/K` (uniform) to `||D||^2` (one-hot), a factor K in squared norm. The uniform average at
alpha = 1 is under-scaled: Task Arithmetic's dev-best alpha is 1.6. So any concentrating rule
raises the effective update scale and can beat uniform at alpha = 1 for reasons unrelated to
conditioning. Shuffled scores keep each map's concentration profile exactly, and so isolate the
effect. Tuning alpha with a matched budget removes it.
