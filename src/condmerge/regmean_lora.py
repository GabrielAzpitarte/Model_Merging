"""RegMean-type merging for LoRA experts: full input-space and adapter-subspace solves, with conditioning knobs.

Per linear map, RegMean solves the response-matching regression (review eq. 30-32)
    min_D  sum_i E_{x~G_i} ||(D - D_i) x||^2   ->   D (sum_i G_i) = sum_i D_i G_i
with RegMean's off-diagonal shrinkage G~ = a G + (1 - a) diag(G) and an optional ridge rho I (review eq. 35).
The kappa-guided ridge picks the smallest rho with kappa2(sum G~ + rho I) <= kappa_target:
    rho = max(0, (l_max - kappa_target * l_min) / (kappa_target - 1)).
Adapter-subspace solve: every LoRA update satisfies D_i = D_i P P^T for P an orthonormal basis of
span{A_1^T, ..., A_K^T} (dimension m <= K r), so restricting D = M P^T is exact for each expert and gives the
m x m system  M (sum_i P^T G~_i P + rho I) = sum_i D_i G~_i P,  with output rank <= m.
All solves run in float64 with linalg.solve on the symmetric system (never an explicit inverse).
"""

from __future__ import annotations

import torch


def shrink(g: torch.Tensor, a: float) -> torch.Tensor:
    return g if a == 1.0 else a * g + (1.0 - a) * torch.diag_embed(torch.diagonal(g, dim1=-2, dim2=-1))


def kappa_sym(g: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(g.double())
    lo, hi = float(ev[0]), float(ev[-1])
    return float("inf") if lo <= 0 else hi / lo


def kappa_ridge(gsum: torch.Tensor, kappa_target: float | None) -> float:
    if kappa_target is None:
        return 0.0
    ev = torch.linalg.eigvalsh(gsum.double())
    lo, hi = float(ev[0]), float(ev[-1])
    return max(0.0, (hi - kappa_target * lo) / (kappa_target - 1.0))


def regmean_full(deltas: torch.Tensor, grams: torch.Tensor, a: float = 1.0, ridge: float = 0.0,
                 kappa_target: float | None = None, ridge_to: str = "zero") -> tuple[torch.Tensor, dict]:
    """deltas (K, dout, din), grams (K, din, din) -> merged update (dout, din) and diagnostics.

    ridge_to='zero': plain Tikhonov; 'mean': D (G + rho I) = sum D_i G~_i + rho * mean(D_i) (as master's regmean_solve).
    """
    d, g = deltas.double(), shrink(grams.double(), a)
    gsum = g.sum(0)
    rho = ridge + kappa_ridge(gsum, kappa_target)
    rhs = torch.einsum("kod,kde->oe", d, g)
    if ridge_to == "mean" and rho > 0:
        rhs = rhs + rho * d.mean(0)
    lhs = gsum + rho * torch.eye(gsum.shape[0], dtype=gsum.dtype, device=gsum.device)
    out = torch.linalg.solve(lhs, rhs.T).T
    return out.float(), {"kappa_G": kappa_sym(gsum), "kappa_solve": kappa_sym(lhs), "rho": rho,
                         "trace_mean": float(torch.diagonal(gsum).mean())}


def adapter_basis(a_factors: list[torch.Tensor], tol: float = 1e-10) -> torch.Tensor:
    """Orthonormal basis (din, m) of the union of the adapters' input row spaces."""
    stack = torch.cat([a.double() for a in a_factors], 0).T  # din x (K r)
    u, s, _ = torch.linalg.svd(stack, full_matrices=False)
    m = int((s > tol * s[0]).sum())
    return u[:, :m]


def regmean_subspace(deltas: torch.Tensor, grams: torch.Tensor, basis: torch.Tensor, a: float = 1.0,
                     ridge: float = 0.0, kappa_target: float | None = None, ridge_to: str = "zero") -> tuple[torch.Tensor, dict]:
    d, g, p = deltas.double(), shrink(grams.double(), a), basis.double()
    gp = torch.einsum("dm,kde,en->kmn", p, g, p)  # P^T G~_i P
    gsum = gp.sum(0)
    rho = ridge + kappa_ridge(gsum, kappa_target)
    rhs = torch.einsum("kod,kde,em->om", d, g, p)
    if ridge_to == "mean" and rho > 0:
        rhs = rhs + rho * (d.mean(0) @ p)   # exact: mean(D_i) lies in the adapter span
    lhs = gsum + rho * torch.eye(gsum.shape[0], dtype=gsum.dtype, device=gsum.device)
    m = torch.linalg.solve(lhs, rhs.T).T
    return (m @ p.T).float(), {"kappa_G": kappa_sym(gsum), "kappa_solve": kappa_sym(lhs), "rho": rho, "dim": int(p.shape[1])}
