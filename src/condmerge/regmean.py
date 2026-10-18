"""RegMean-type solves and conditioning-guided regularisation (review sec. 3.6).

For one linear map with expert updates ``D_i`` (dout x din) and uncentred input second
moments ``G_i = E[x x^T]`` (din x din, each expert's own activations), the response-matching
merge solves

    min_D  sum_i E_i || (D - D_i) x ||^2  [+ rho || D - R ||_F^2]
    =>     D (G + rho I) = sum_i D_i G~_i + rho R,      G = sum_i G~_i,

where ``G~_i = a G_i + (1 - a) diag(G_i)`` is RegMean's off-diagonal shrinkage (``a = 1``:
none) and the ridge target ``R`` is either 0 (``ridge_to='zero'``) or the uniform average of
the updates (``ridge_to='mean'``: directions the inputs do not identify fall back to averaging).

Conditioning enters through the matrix actually inverted, ``G + rho I``. First-order error
propagation (review eq. 33-34): ``||dD|| <= ||(G + rho I)^-1|| (||dB|| + ||D|| ||dG||)``.
``kappa_target`` sets the smallest ``rho`` with ``kappa2(G + rho I) <= kappa_target``:
``rho = max(0, (l_max - kappa_target l_min) / (kappa_target - 1))``.

Solves are float64 Cholesky (lstsq fallback); no explicit inverse is formed.
"""

from __future__ import annotations

import math

import torch


def shrink_gram(g: torch.Tensor, a: float) -> torch.Tensor:
    if a == 1.0:
        return g
    d = torch.diag(torch.diagonal(g))
    return a * g + (1.0 - a) * d


def gram_kappa(g: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(0.5 * (g + g.T).double())
    lo, hi = float(ev[0]), float(ev[-1])
    return math.inf if lo <= 0 else hi / lo


def kappa_ridge(g: torch.Tensor, kappa_target: float) -> tuple[float, float, float]:
    """(rho, kappa before, kappa after) for the cap kappa2(G + rho I) <= kappa_target."""
    ev = torch.linalg.eigvalsh(0.5 * (g + g.T).double())
    lo, hi = float(ev[0]), float(ev[-1])
    k_before = math.inf if lo <= 0 else hi / lo
    if math.isinf(kappa_target) or k_before <= kappa_target:
        return 0.0, k_before, k_before
    rho = max(0.0, (hi - kappa_target * lo) / (kappa_target - 1.0))
    return rho, k_before, (hi + rho) / (lo + rho)


def kappa_shrink(g_sum: torch.Tensor, kappa_target: float, iters: int = 20) -> tuple[float, float]:
    """Largest a in [0, 1] with kappa2(a G + (1 - a) diag G) <= kappa_target (bisection); (a, kappa_after).

    The axis-aligned counterpart of :func:`kappa_ridge`: same conditioning target, but the
    regulariser moves G toward its diagonal (privileging the coordinate axes) instead of adding
    an isotropic ridge. If even diag(G) violates the target, returns a = 0.
    """
    k1 = gram_kappa(g_sum)
    if k1 <= kappa_target:
        return 1.0, k1
    k0 = gram_kappa(shrink_gram(g_sum, 0.0))
    if k0 > kappa_target:
        return 0.0, k0
    lo, hi = 0.0, 1.0  # kappa(lo) <= target < kappa(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if gram_kappa(shrink_gram(g_sum, mid)) <= kappa_target:
            lo = mid
        else:
            hi = mid
    return lo, gram_kappa(shrink_gram(g_sum, lo))


def regmean_solve(deltas: list[torch.Tensor], grams: list[torch.Tensor], shrink: float = 1.0,
                  ridge: float | None = None, kappa_target: float | None = None, ridge_to: str = "mean",
                  return_info: bool = False, kappa_shrink_target: float | None = None,
                  rel_ridge: float | None = None):
    """Merged update for one linear map. ``deltas``: K x (dout, din); ``grams``: K x (din, din).

    ``kappa_shrink_target``: choose the shrinkage ``a`` per map by the kappa cap (overrides ``shrink``).
    ``rel_ridge``: isotropic ridge of fixed relative strength, ``rho = rel_ridge * tr(G) / din``.
    """
    dev = deltas[0].device
    if kappa_shrink_target is not None:
        shrink, _ = kappa_shrink(sum(g.to(dev).double() for g in grams), kappa_shrink_target)
    gs = [shrink_gram(g.to(dev).double(), shrink) for g in grams]
    ds = [d.to(dev).double() for d in deltas]
    g = sum(gs)
    rhs = sum(d @ gi for d, gi in zip(ds, gs, strict=True))
    rho, k_before, k_after = 0.0, float("nan"), float("nan")
    if kappa_target is not None:
        rho, k_before, k_after = kappa_ridge(g, kappa_target)
    elif return_info:
        k_before = k_after = gram_kappa(g)
    if rel_ridge is not None:
        ridge = (ridge or 0.0) + float(rel_ridge) * float(torch.trace(g)) / g.shape[0]
    if ridge is not None:
        rho += float(ridge)
        if return_info:
            ev = torch.linalg.eigvalsh(0.5 * (g + g.T))
            k_after = float((ev[-1] + rho) / (ev[0] + rho)) if float(ev[0]) + rho > 0 else math.inf
    if rho > 0:
        target = sum(ds) / len(ds) if ridge_to == "mean" else torch.zeros_like(ds[0])
        rhs = rhs + rho * target
        g = g + rho * torch.eye(g.shape[0], dtype=g.dtype, device=dev)
    gs_sym = 0.5 * (g + g.T)
    try:
        chol = torch.linalg.cholesky(gs_sym)
        sol = torch.cholesky_solve(rhs.T, chol).T  # D = rhs G^-1  <=>  G D^T = rhs^T
    except RuntimeError:
        sol = torch.linalg.lstsq(gs_sym, rhs.T).solution.T
    if return_info:
        return sol, {"rho": rho, "kappa_before": k_before, "kappa_after": k_after, "shrink": shrink,
                     "trace_mean": float(torch.trace(gs_sym)) / gs_sym.shape[0]}
    return sol
