"""Stage 1: numerically verify the review's conditional mathematics on small linear systems.

Every check prints PASS/FAIL against an analytic value and records numbers in a JSON.
Regimes where kappa-decreasing weights help AND regimes where they fail are both shown.
Pure numpy/scipy, CPU, seconds.

Checks (review section numbers in brackets):
 C1 [3.3 eq.12-13]  shared target, independent noise: optimal lambda = inverse variance (Monte Carlo + QP)
 C2 [3.3 eq.17-22]  controlled LS designs Z_i = diag(1,..,1,1/k_i): v_i = nu^2 (d-1+k_i^2); exp(-beta*kappa)
                    vs power law vs uniform vs oracle; eq. 21 bounds; scale and spectrum counterexamples
 C3 [3.3 eq.14-16]  correlated / duplicated experts: GLS weights can be negative; (1+rho)/2; softmax duplicate bias
 C4 [3.2 eq.8]      complementary vs cancelling orientations: equal kappa, opposite merge outcome
 C5 [3.8 Fig.1L]    distinct tasks: penalising the high-kappa expert is strictly worse
 C6 [3.5 eq.26-27]  heterogeneous quadratics: no scalar mixture reaches the curvature-weighted optimum
 C7 [3.4 eq.23-24]  softmax = unique minimiser of the entropy-regularised linear cost
 C8 [3.6 eq.30-35]  regression merge B G^-1, ridge, first-order perturbation bound, kappa(Z^T Z)=kappa(Z)^2
 C9 [3.7 eq.36-37]  LoRA gauge: factor kappas change, product spectrum invariant; QR-core spectrum exact
 C10 [3.2 eq.10]    kappa is coordinate dependent while responses/functional error are invariant
 C11 [new bridge]   weight-domain noise: kappa(W* + E) vs noise level s for near-singular W* -- noise can
                    IMPROVE conditioning (smoothed-analysis regime), reversing the rule's intent
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

RESULTS: dict = {}
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def kappa(m: np.ndarray) -> float:
    s = np.linalg.svd(m, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else math.inf


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def simplex_qp(C: np.ndarray, nonneg: bool = True) -> np.ndarray:
    """argmin lambda^T C lambda over the simplex (or the affine hyperplane if nonneg=False)."""
    k = C.shape[0]
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones_like(x)}]
    bounds = [(0, None)] * k if nonneg else None
    r = minimize(lambda x: x @ C @ x, np.full(k, 1 / k), jac=lambda x: 2 * C @ x, constraints=cons,
                 bounds=bounds, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return r.x


# ----------------------------------------------------------------------------------------------
def c1_inverse_variance(rng: np.random.Generator) -> None:
    v = np.array([0.5, 1.0, 2.0, 4.0])
    d, n_mc = 50, 20000
    lam_star = (1 / v) / (1 / v).sum()
    lam_qp = simplex_qp(np.diag(v))
    theta = rng.normal(size=d)
    # Monte Carlo risk of the inverse-variance merge vs uniform
    eps = rng.normal(size=(n_mc, len(v), d)) * np.sqrt(v / d)[None, :, None]
    est = theta[None, None, :] + eps

    def risk(lam):
        return float(np.mean(np.sum((np.einsum("k,nkd->nd", lam, est) - theta) ** 2, axis=1)))

    r_star, r_unif = risk(lam_star), risk(np.full(4, 0.25))
    an_star, an_unif = float(np.sum(lam_star ** 2 * v)), float(np.sum(v) / 16)
    check("C1 QP optimum = inverse variance", np.allclose(lam_qp, lam_star, atol=1e-6), f"{lam_qp.round(4)}")
    check("C1 MC risk matches analytic (opt)", abs(r_star - an_star) / an_star < 0.03, f"{r_star:.4f} vs {an_star:.4f}")
    check("C1 MC risk matches analytic (uniform)", abs(r_unif - an_unif) / an_unif < 0.03, f"{r_unif:.4f} vs {an_unif:.4f}")
    RESULTS["C1"] = {"v": v.tolist(), "lam_opt": lam_star.tolist(), "risk_opt": an_star, "risk_uniform": an_unif}


# ----------------------------------------------------------------------------------------------
def c2_controlled_designs(rng: np.random.Generator) -> None:
    d, nu2 = 8, 1.0
    kap = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    v = nu2 * (d - 1 + kap ** 2)
    # eq. 20 by direct inversion
    v_direct = np.array([nu2 * np.trace(np.linalg.inv(np.diag(np.r_[np.ones(d - 1), 1 / k]) ** 2)) for k in kap])
    check("C2 eq.22 variance formula", np.allclose(v, v_direct), f"{v}")
    # eq. 21 bounds for random designs
    ok = True
    for _ in range(200):
        Z = rng.normal(size=(20, d)) * rng.lognormal(0, 1, size=d)[None, :]
        s = np.linalg.svd(Z, compute_uv=False)
        vi = nu2 * np.sum(s ** -2.0)
        lo, hi = nu2 * (s[0] / s[-1]) ** 2 / s[0] ** 2, d * nu2 * (s[0] / s[-1]) ** 2 / s[0] ** 2
        ok &= (lo <= vi * (1 + 1e-12)) and (vi <= hi * (1 + 1e-12))
    check("C2 eq.21 bounds hold on 200 random designs", ok)
    # rules: oracle inverse variance vs uniform vs exp(-beta*kappa) (best beta) vs power law kappa^-p (best p)
    risk = lambda lam: float(np.sum(lam ** 2 * v))
    lam_or = (1 / v) / (1 / v).sum()
    betas = np.geomspace(1e-3, 10, 400)
    r_exp = [risk(softmax(-b * kap)) for b in betas]
    ps = np.linspace(0, 4, 401)
    r_pow = [risk(kap ** -p / np.sum(kap ** -p)) for p in ps]
    r_rev = [risk(softmax(+b * kap)) for b in betas]
    RESULTS["C2"] = {"kappa": kap.tolist(), "v": v.tolist(), "risk_oracle": risk(lam_or),
                     "risk_uniform": risk(np.full(5, 0.2)), "risk_exp_best": min(r_exp),
                     "beta_best": float(betas[int(np.argmin(r_exp))]), "risk_pow_best": min(r_pow),
                     "p_best": float(ps[int(np.argmin(r_pow))]), "risk_reversed_best": min(r_rev)}
    check("C2 kappa-decreasing weights beat uniform in the shared-target LS family",
          min(r_exp) < risk(np.full(5, 0.2)) and min(r_pow) < risk(np.full(5, 0.2)),
          f"oracle {risk(lam_or):.3f} exp {min(r_exp):.3f} pow {min(r_pow):.3f} uniform {risk(np.full(5, .2)):.3f}")
    check("C2 reversed preference never beats uniform here", min(r_rev) >= risk(np.full(5, 0.2)) - 1e-12)
    # counterexamples from the review
    v_I = nu2 * np.sum(np.linalg.svd(np.eye(2), compute_uv=False) ** -2.0)
    v_Z2 = nu2 * np.sum(np.linalg.svd(100 * np.diag([1, 0.1]), compute_uv=False) ** -2.0)
    check("C2 scale counterexample (kappa 10 design far more precise than kappa 1)",
          abs(v_I - 2) < 1e-12 and abs(v_Z2 - 0.0101) < 1e-12, f"v(I)={v_I}, v(100diag(1,.1))={v_Z2:.4f}")
    va = np.sum(np.array([1, 0.1, 0.1]) ** -2.0)
    vb = np.sum(np.array([1, 1, 0.1]) ** -2.0)
    check("C2 same kappa, different variance (201 vs 102)", abs(va - 201) < 1e-9 and abs(vb - 102) < 1e-9)


# ----------------------------------------------------------------------------------------------
def c3_correlation(rng: np.random.Generator) -> None:
    for rho in (0.0, 0.5, 0.9):
        C = np.array([[1, rho], [rho, 1]])
        lam = np.array([0.5, 0.5])
        check(f"C3 Var((e1+e2)/2) = (1+rho)/2 at rho={rho}", abs(lam @ C @ lam - (1 + rho) / 2) < 1e-12)
    # GLS weights can be negative: with variances 1 and 2 the weight on expert 2 is ~ (1 - cov),
    # negative once the correlation exceeds sigma_1/sigma_2 = 0.707 (here cov 1.2, rho 0.85)
    C = np.array([[1.0, 1.2], [1.2, 2.0]])
    gls = np.linalg.solve(C, np.ones(2))
    gls /= gls.sum()
    gls_qp = simplex_qp(C, nonneg=False)
    nn = simplex_qp(C, nonneg=True)
    check("C3 unconstrained GLS has a negative weight", gls.min() < 0 and np.allclose(gls, gls_qp, atol=1e-5), f"{gls.round(4)}")
    check("C3 simplex QP clips to a vertex", np.allclose(nn, [1, 0], atol=1e-5), f"{nn.round(4)}")
    # duplicate expert: uniform softmax over [A, A, B] gives A 2/3 of the mass with no new information
    lam = softmax(np.zeros(3))
    check("C3 duplicated expert doubles its group mass under uniform softmax", abs(lam[:2].sum() - 2 / 3) < 1e-12)
    RESULTS["C3"] = {"gls": gls.tolist(), "simplex": nn.tolist()}


# ----------------------------------------------------------------------------------------------
def c4_orientation() -> None:
    eps = 1e-2
    W1, W2 = np.diag([1, eps]), np.diag([eps, 1])
    check("C4 equal individual kappa", abs(kappa(W1) - kappa(W2)) < 1e-9 and abs(kappa(W1) - 1 / eps) < 1e-6)
    check("C4 complementary average is perfectly conditioned", abs(kappa((W1 + W2) / 2) - 1) < 1e-12)
    check("C4 cancelling pair (W2=-W1) averages to zero with unchanged kappas",
          np.allclose((W1 - W1) / 2, 0) and kappa(-W1) == kappa(W1))
    check("C4 kappa is scale invariant (cannot see over-large updates)", abs(kappa(7 * W1) - kappa(W1)) < 1e-6)


# ----------------------------------------------------------------------------------------------
def c5_distinct_tasks(rng: np.random.Generator) -> None:
    W1, W2 = np.diag([1.0, 1e-4]), np.eye(2)
    lams = np.linspace(0, 1, 101)
    # isotropic inputs: excess loss of (1-lam) W2 + lam W1 over the equal average, averaged over both tasks
    def loss(lam):
        W = lam * W1 + (1 - lam) * W2
        return 0.5 * (np.sum((W - W1) ** 2) + np.sum((W - W2) ** 2))
    L = np.array([loss(l) for l in lams])
    excess = L - loss(0.5)
    analytic = (lams - 0.5) ** 2 * (1 - 1e-4) ** 2
    check("C5 excess loss = (lam-1/2)^2 (1-1e-4)^2", np.allclose(excess, analytic, atol=1e-12))
    lam_rule = softmax(-1e-3 * np.array([kappa(W1), kappa(W2)]))[0]
    check("C5 the kappa rule moves weight away from the high-kappa expert and loses",
          lam_rule < 0.5 and loss(lam_rule) > loss(0.5), f"lam_highk={lam_rule:.3f}, excess={loss(lam_rule) - loss(.5):.4f}")
    RESULTS["C5"] = {"lam_rule_beta1e-3": float(lam_rule), "excess": float(loss(lam_rule) - loss(0.5))}


# ----------------------------------------------------------------------------------------------
def c6_quadratics() -> None:
    t1, t2 = np.array([1.0, 1.0]), np.array([2.0, 2.0])
    H1, H2 = np.diag([100.0, 1.0]), np.diag([1.0, 100.0])
    Hbar = 0.5 * (H1 + H2)
    th = np.linalg.solve(Hbar, 0.5 * (H1 @ t1 + H2 @ t2))
    check("C6 curvature-weighted optimum = (102/101, 201/101)", np.allclose(th, [102 / 101, 201 / 101]))
    L = lambda x: 0.5 * (0.5 * (x - t1) @ H1 @ (x - t1) + 0.5 * (x - t2) @ H2 @ (x - t2))
    # best point on the span of the experts with free scale (a t1 + b t2 has equal coordinates)
    best = minimize(lambda ab: L(ab[0] * t1 + ab[1] * t2), np.array([0.5, 0.5])).fun
    check("C6 no linear combination of experts reaches it", best > L(th) + 1e-3, f"best-span {best:.4f} vs opt {L(th):.4f}")
    check("C6 both Hessians have kappa 100", kappa(H1) == 100 and kappa(H2) == 100)
    RESULTS["C6"] = {"opt": th.tolist(), "loss_opt": float(L(th)), "loss_best_span": float(best)}


# ----------------------------------------------------------------------------------------------
def c7_entropy_softmax(rng: np.random.Generator) -> None:
    c = rng.normal(size=5) * 3
    q = rng.dirichlet(np.ones(5))
    tau = 0.7
    obj = lambda lam: float(lam @ c + tau * np.sum(lam * np.log(np.maximum(lam, 1e-300) / q)))
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    r = minimize(obj, np.full(5, 0.2), bounds=[(1e-12, 1)] * 5, constraints=cons, method="SLSQP",
                 options={"ftol": 1e-15, "maxiter": 1000})
    closed = q * np.exp(-c / tau)
    closed /= closed.sum()
    check("C7 entropy-regularised optimum = prior-weighted softmax", np.allclose(r.x, closed, atol=1e-5),
          f"max err {np.abs(r.x - closed).max():.2e}")
    check("C7 shift invariance / scale sensitivity",
          np.allclose(softmax(-(c + 5) / tau), softmax(-c / tau)) and not np.allclose(softmax(-2 * c / tau), softmax(-c / tau)))


# ----------------------------------------------------------------------------------------------
def c8_regression(rng: np.random.Generator) -> None:
    dout, din, k = 6, 10, 3
    Ds = [rng.normal(size=(dout, din)) for _ in range(k)]
    Xs = [rng.normal(size=(200, din)) * rng.lognormal(0, 1, size=din)[None, :] for _ in range(k)]
    pis = np.array([0.2, 0.3, 0.5])
    Cs = [X.T @ X / len(X) for X in Xs]
    G = sum(p * C for p, C in zip(pis, Cs))
    B = sum(p * D @ C for p, D, C in zip(pis, Ds, Cs))
    sol = np.linalg.solve(G.T, B.T).T  # B G^-1
    # direct least squares on stacked, sqrt(pi/n)-weighted data
    Xall = np.vstack([np.sqrt(p / len(X)) * X for p, X in zip(pis, Xs)])
    Yall = np.vstack([np.sqrt(p / len(X)) * X @ D.T for p, X, D in zip(pis, Xs, Ds)])
    ls = np.linalg.lstsq(Xall, Yall, rcond=None)[0].T
    check("C8 RegMean normal-equation solution = stacked least squares", np.allclose(sol, ls, atol=1e-8))
    # same second moment -> plain average
    Bs = sum(p * D @ Cs[0] for p, D in zip(pis, Ds))
    check("C8 identical C_i reduce to pi-weighted averaging",
          np.allclose(np.linalg.solve(Cs[0].T, Bs.T).T, sum(p * D for p, D in zip(pis, Ds))))
    # first-order perturbation bound eq.34
    dB, dG = 1e-6 * rng.normal(size=B.shape), 1e-6 * rng.normal(size=G.shape)
    dG = 0.5 * (dG + dG.T)
    pert = np.linalg.solve((G + dG).T, (B + dB).T).T - sol
    bound = np.linalg.norm(np.linalg.inv(G), 2) * (np.linalg.norm(dB, 2) + np.linalg.norm(sol, 2) * np.linalg.norm(dG, 2))
    check("C8 eq.34 first-order perturbation bound", np.linalg.norm(pert, 2) <= bound * 1.01,
          f"{np.linalg.norm(pert, 2):.3e} <= {bound:.3e}")
    rho = 1e-1
    ridge = np.linalg.solve((G + rho * np.eye(din)).T, B.T).T
    check("C8 ridge solution solves the regularised objective",
          np.allclose(ridge @ (G + rho * np.eye(din)), B, atol=1e-9))
    Z = rng.normal(size=(30, 5)) * np.array([1, 2, 5, 10, 50])[None, :]
    check("C8 kappa(Z^T Z) = kappa(Z)^2", abs(kappa(Z.T @ Z) / kappa(Z) ** 2 - 1) < 1e-8)


# ----------------------------------------------------------------------------------------------
def c9_lora_gauge(rng: np.random.Generator) -> None:
    dout, din, r = 64, 48, 8
    Bm, A = rng.normal(size=(dout, r)), rng.normal(size=(r, din))
    R = rng.normal(size=(r, r)) @ np.diag(np.geomspace(1, 100, r)) @ rng.normal(size=(r, r))
    B2, A2 = Bm @ R, np.linalg.solve(R, A)
    check("C9 product invariant under gauge R", np.allclose(Bm @ A, B2 @ A2, atol=1e-8))
    check("C9 factor kappas change under gauge", abs(kappa(A) - kappa(A2)) / kappa(A) > 0.1 and abs(kappa(Bm) - kappa(B2)) / kappa(Bm) > 0.1,
          f"kappa(A) {kappa(A):.1f}->{kappa(A2):.1f}, kappa(B) {kappa(Bm):.1f}->{kappa(B2):.1f}")
    QB, RB = np.linalg.qr(Bm)
    QA, RA = np.linalg.qr(A.T)
    core = np.linalg.svd(RB @ RA.T, compute_uv=False)
    full = np.linalg.svd(Bm @ A, compute_uv=False)[:r]
    check("C9 QR-core spectrum equals the product's nonzero spectrum", np.allclose(core, full, rtol=1e-10))
    full_s = np.linalg.svd(Bm @ A, compute_uv=False)
    check("C9 ordinary kappa2 of a rank-r product is numerically infinite", full_s[-1] / full_s[0] < 1e-12,
          f"sigma_min/sigma_max={full_s[-1] / full_s[0]:.1e}")
    # B_avg A_avg != average of products (cross terms)
    B1, A1, B3, A3 = rng.normal(size=(dout, r)), rng.normal(size=(r, din)), rng.normal(size=(dout, r)), rng.normal(size=(r, din))
    check("C9 averaging factors is not averaging products",
          not np.allclose(0.5 * (B1 + B3) @ (0.5 * (A1 + A3)), 0.5 * (B1 @ A1 + B3 @ A3)))
    cat = np.hstack([0.5 * B1, 0.5 * B3]) @ np.vstack([A1, A3])
    check("C9 concatenated adapter is exact (rank <= 2r)", np.allclose(cat, 0.5 * (B1 @ A1 + B3 @ A3)))


# ----------------------------------------------------------------------------------------------
def c10_coordinates(rng: np.random.Generator) -> None:
    n = 6
    W, Ws = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    L = rng.normal(size=(n, n))
    Cx = L @ L.T + 0.1 * np.eye(n)
    R = np.diag(np.geomspace(1, 1000, n)) @ np.linalg.qr(rng.normal(size=(n, n)))[0]
    Ri = np.linalg.inv(R)
    err = np.trace((W - Ws) @ Cx @ (W - Ws).T)
    err2 = np.trace((W @ Ri - Ws @ Ri) @ (R @ Cx @ R.T) @ (W @ Ri - Ws @ Ri).T)
    check("C10 functional error eq.10 invariant under reparameterisation", abs(err - err2) / err < 1e-8)
    check("C10 kappa changes under the same reparameterisation", abs(kappa(W) - kappa(W @ Ri)) / kappa(W) > 0.5,
          f"{kappa(W):.1f} -> {kappa(W @ Ri):.1f}")


# ----------------------------------------------------------------------------------------------
def c11_noise_vs_conditioning(rng: np.random.Generator, base_mats: dict[str, np.ndarray] | None) -> None:
    """Does kappa(W* + E) increase with the noise level s? Not when W* is near-singular.

    W* = U diag(sigma) V^T with a tiny tail (sigma_min/sigma_max = 1e-6), E Gaussian with per-entry std s.
    Smoothed analysis: sigma_min(W* + E) >~ s / sqrt(n) w.h.p., so for s/sqrt(n) >> sigma_min(W*) the
    condition number FALLS as s grows (more noise = better conditioned), until s ~ sigma_max/sqrt(n).
    If experts were noisy estimates of one target, exp(-beta*kappa) would then upweight the NOISIEST.
    """
    n = 128
    U = np.linalg.qr(rng.normal(size=(n, n)))[0]
    V = np.linalg.qr(rng.normal(size=(n, n)))[0]
    sig = np.geomspace(1.0, 1e-6, n)
    Wst = U @ np.diag(sig) @ V.T
    levels = np.geomspace(1e-9, 1e-1, 17)
    rows = []
    for s in levels:
        ks = [kappa(Wst + s * rng.normal(size=(n, n))) for _ in range(20)]
        rows.append({"s": float(s), "kappa_median": float(np.median(ks)), "kappa_q10": float(np.quantile(ks, .1)),
                     "kappa_q90": float(np.quantile(ks, .9))})
    k_med = np.array([r["kappa_median"] for r in rows])
    # in the window s in [1e-6, 1e-2] (i.e. s/sqrt(n) above sigma_min*sqrt-ish), kappa must DECREASE with s
    win = (levels >= 1e-6) & (levels <= 1e-2)
    dec = np.all(np.diff(k_med[win]) < 0)
    check("C11 near-singular W*: kappa decreases as weight noise increases (smoothed analysis regime)", bool(dec),
          f"kappa {k_med[levels == 1e-6][0]:.2e} @s=1e-6 -> {k_med[np.isclose(levels, 1e-2)][0]:.2e} @s=1e-2")
    # well-conditioned W*: kappa increases with noise once noise is not negligible
    sig2 = np.geomspace(1.0, 0.5, n)
    Wwc = U @ np.diag(sig2) @ V.T
    k2 = [float(np.median([kappa(Wwc + s * rng.normal(size=(n, n))) for _ in range(20)])) for s in levels]
    inc = np.all(np.diff(np.array(k2)[levels >= 1e-3]) > 0)
    check("C11 well-conditioned W*: kappa increases with weight noise", bool(inc))
    RESULTS["C11"] = {"near_singular": rows, "well_conditioned": [{"s": float(s), "kappa_median": k} for s, k in zip(levels, k2)]}
    if base_mats:
        out = {}
        for name, W0 in base_mats.items():
            s0 = np.linalg.svd(W0, compute_uv=False)
            m_, n_ = W0.shape
            res = []
            for s in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2):
                ks = [kappa(W0 + s * rng.normal(size=W0.shape)) for _ in range(5)]
                res.append({"s": s, "kappa_median": float(np.median(ks))})
            out[name] = {"kappa_W0": float(s0[0] / s0[-1]), "sigma_min_W0": float(s0[-1]), "noise": res}
            print(f"   real base {name}: kappa0={s0[0] / s0[-1]:.3e} ->",
                  ", ".join(f"s={r['s']:.0e}:{r['kappa_median']:.3e}" for r in res))
        RESULTS["C11_real_base"] = out


def main() -> None:
    rng = np.random.default_rng(0)
    base_mats = None
    if len(sys.argv) > 2 and sys.argv[2]:
        blob = np.load(sys.argv[2])
        base_mats = {k: blob[k].astype(np.float64) for k in blob.files}
    c1_inverse_variance(rng)
    c2_controlled_designs(rng)
    c3_correlation(rng)
    c4_orientation()
    c5_distinct_tasks(rng)
    c6_quadratics()
    c7_entropy_softmax(rng)
    c8_regression(rng)
    c9_lora_gauge(rng)
    c10_coordinates(rng)
    c11_noise_vs_conditioning(rng, base_mats)
    RESULTS["fails"] = FAILS
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=1))
    print(f"\n{len(FAILS)} FAIL(s): {FAILS}" if FAILS else "\nALL CHECKS PASS")


if __name__ == "__main__":
    main()
