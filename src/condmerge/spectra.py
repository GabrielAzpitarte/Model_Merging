"""Spectral statistics of full weights, task updates and merged matrices.

Conventions (review, eqs. 3-4 and 38-39):

* ``kappa2(M) = s_1 / s_q`` with ``q = min(m, n)``; it is ``inf`` when the
  numerical rank (relative tolerance ``tol``) is below ``q``.
* ``kappa_plus(M; tol) = s_1 / s_r`` on the ``r`` singular values above
  ``tol * s_1`` --- a *different* statistic (it discards the null space).
* stable rank ``||M||_F^2 / ||M||_2^2``; entropy effective rank ``exp(H(p))``
  with ``p_j = s_j^2 / sum s^2``.
* ``kappa_q(M; f) = s_1 / s_{ceil(f q)}``: quantile ratio, tail-robust.

Linear maps (the matrices the condition-number rule is defined on) are the 72
attention/MLP weights plus the patch-embedding convolution reshaped to
``(768, 3*32*32)``; the position embedding is a lookup table and is *not*
treated as a linear map. All SVDs run in float64 on the stored float32 values.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# float32 machine epsilon; numpy's matrix_rank default is max(m, n) * eps * s_1
EPS32 = float(np.finfo(np.float32).eps)
REL_TOLS = (None, 1e-6, 1e-4, 1e-3, 1e-2)  # None -> max(m,n)*eps32 (numpy default for fp32 data)


def is_linear_map(name: str, t: torch.Tensor) -> bool:
    if name == "embeddings.patch_embedding.weight":
        return True
    return t.ndim == 2 and name.endswith(".weight") and "position_embedding" not in name


def as_matrix(name: str, t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], -1) if t.ndim == 4 else t


def linear_map_names(sd: dict[str, torch.Tensor]) -> list[str]:
    return [k for k, v in sd.items() if is_linear_map(k, v)]


def bias_owner(name: str) -> str | None:
    """Weight whose lambda a bias inherits under the 'tied' bias policy."""
    if name.endswith(".bias"):
        w = name[: -len(".bias")] + ".weight"
        if "layer_norm" in name or "layernorm" in name:
            return None
        return w
    return None


def layer_index(name: str) -> int:
    """Transformer block index; -1 for embeddings, 12 for post-layernorm."""
    if name.startswith("encoder.layers."):
        return int(name.split(".")[2])
    if name.startswith("embeddings") or name.startswith("pre_layrnorm"):
        return -1
    return 12


def module_type(name: str) -> str:
    for key in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"):
        if key in name:
            return key
    return "other"


@torch.no_grad()
def svdvals64(m: torch.Tensor, device: str = "cuda") -> np.ndarray:
    return torch.linalg.svdvals(m.to(device=device, dtype=torch.float64)).cpu().numpy()


def spectral_stats(s: np.ndarray, shape: tuple[int, int]) -> dict[str, float]:
    """All scalar statistics from a descending singular-value vector."""
    m, n = shape
    q = min(m, n)
    s = np.asarray(s, dtype=np.float64)
    out: dict[str, float] = {}
    s1 = float(s[0]) if s.size else 0.0
    out["smax"] = s1
    out["smin"] = float(s[q - 1])
    out["fro"] = float(np.sqrt(np.sum(s**2)))
    out["nuc"] = float(np.sum(s))
    if s1 <= 0:
        out.update(kappa2=math.inf, stable_rank=0.0, entropy_rank=0.0)
        return out
    default_tol = max(m, n) * EPS32
    rank_default = int(np.sum(s > default_tol * s1))
    out["rank_default"] = rank_default
    out["kappa2"] = float(s1 / s[q - 1]) if (rank_default == q and s[q - 1] > 0) else math.inf
    out["kappa2_raw"] = float(s1 / s[q - 1]) if s[q - 1] > 0 else math.inf  # ignores the rank rule
    for tol in REL_TOLS[1:]:
        r = int(np.sum(s > tol * s1))
        out[f"rank_{tol:g}"] = r
        out[f"kappa_plus_{tol:g}"] = float(s1 / s[r - 1])
    for f in (0.5, 0.9, 0.99):
        j = max(int(math.ceil(f * q)) - 1, 0)
        out[f"kappa_q{int(f * 100)}"] = float(s1 / s[j]) if s[j] > 0 else math.inf
    out["stable_rank"] = float(np.sum(s**2) / s1**2)
    p = s**2 / np.sum(s**2)
    p = p[p > 0]
    out["entropy_rank"] = float(np.exp(-np.sum(p * np.log(p))))
    return out


@torch.no_grad()
def first_order_kappa_shift(w0: torch.Tensor, deltas: list[torch.Tensor], device: str = "cuda") -> dict:
    """First-order prediction of kappa2(W0 + D) - kappa2(W0) for each update D.

    For simple extreme singular values, ``d s_1 = u_1^T D v_1`` and
    ``d s_q = u_q^T D v_q``, so
    ``d kappa ~= kappa0 * (u_1^T D v_1 / s_1 - u_q^T D v_q / s_q)``:
    the literal score is (to first order) a *linear rank-2 probe* of the update
    along the base's extreme singular pairs.
    """
    w = w0.to(device=device, dtype=torch.float64)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    q = s.shape[0]
    u1, v1, uq, vq = u[:, 0], vh[0], u[:, q - 1], vh[q - 1]
    s1, sq = float(s[0]), float(s[q - 1])
    k0 = s1 / sq
    preds, d1s, dqs = [], [], []
    for d in deltas:
        d = d.to(device=device, dtype=torch.float64)
        d1 = float(u1 @ d @ v1)
        dq = float(uq @ d @ vq)
        d1s.append(d1)
        dqs.append(dq)
        preds.append(k0 * (d1 / s1 - dq / sq))
    gaps = s.cpu().numpy()
    return {"kappa0": k0, "pred_shift": preds, "d_s1": d1s, "d_sq": dqs, "s1": s1, "sq": sq,
            "gap_top": float(gaps[0] - gaps[1]), "gap_bottom": float(gaps[q - 2] - gaps[q - 1])}


@torch.no_grad()
def subspace_overlap(a: torch.Tensor, b: torch.Tensor, k: int, device: str = "cuda") -> tuple[float, float]:
    """Mean squared cosine between top-k left (and right) singular subspaces of two updates."""
    ua, _, vha = torch.linalg.svd(a.to(device=device, dtype=torch.float64), full_matrices=False)
    ub, _, vhb = torch.linalg.svd(b.to(device=device, dtype=torch.float64), full_matrices=False)
    left = float((ua[:, :k].T @ ub[:, :k]).pow(2).sum() / k)
    right = float((vha[:k] @ vhb[:k].T).pow(2).sum() / k)
    return left, right
