"""Coefficient rules and the merge operator.

Merge (review eq. 1)::

    W_merge[p] = W0[p] + alpha * sum_i lambda_i[p] * (W_i[p] - W0[p]),   sum_i lambda_i[p] = 1

At ``alpha = 1`` this *is* the normalised full-weight average
``sum_i lambda_i W_i`` (the professor's literal form). Task Arithmetic with
per-task scale ``s`` is uniform lambda with ``alpha = K * s``.

Coefficients (review eq. 2)::

    lambda_i[p] = q_i exp(-beta * c_i[p]) / sum_j q_j exp(-beta * c_j[p])

computed with log-sum-exp. Policies, fixed here so that every rule shares them:

* ``beta = 0`` returns the prior exactly (uniform), whatever the scores;
* a score of ``+inf`` (e.g. kappa2 of a rank-deficient matrix) gets lambda 0
  when at least one expert has a finite score; if all are ``+inf`` the prior is
  returned; ``-inf`` scores share all the mass equally; NaN raises;
* parameters that are not linear maps (LayerNorm, class/position embeddings)
  get the prior; a linear layer's bias inherits its weight's lambda
  (``bias_policy='tied'``) or the prior (``'uniform'``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .spectra import bias_owner


def softmax_coeffs(scores: np.ndarray, beta: float, prior: np.ndarray | None = None) -> np.ndarray:
    c = np.asarray(scores, dtype=np.float64)
    k = c.shape[0]
    q = np.full(k, 1.0 / k) if prior is None else np.asarray(prior, dtype=np.float64) / np.sum(prior)
    if np.any(np.isnan(c)):
        raise ValueError(f"NaN score: {c}")
    if beta == 0:
        return q.copy()
    if np.all(np.isposinf(c)):
        return q.copy()
    if np.any(np.isneginf(c)):
        m = np.isneginf(c).astype(np.float64) * q
        return m / m.sum()
    logits = np.log(q) - beta * c  # +inf scores -> -inf logits -> weight 0
    mx = np.max(logits[np.isfinite(logits)])
    w = np.exp(logits - mx)
    w[~np.isfinite(logits)] = 0.0
    return w / w.sum()


def concentration(lam: np.ndarray) -> dict[str, float]:
    """max weight and the effective number of experts exp(H(lambda))."""
    lam = np.asarray(lam, dtype=np.float64)
    p = lam[lam > 0]
    return {"max": float(lam.max()), "n_eff": float(np.exp(-np.sum(p * np.log(p))))}


@dataclass
class Bank:
    """Base + experts on one device, stored as base and stacked deltas per parameter."""

    names: list[str]
    base: dict[str, torch.Tensor]
    deltas: dict[str, torch.Tensor]  # (K, *shape)
    experts: list[str]

    @classmethod
    def build(cls, base: dict, experts: dict[str, dict], device: str = "cuda") -> "Bank":
        names = list(base)
        tasks = list(experts)
        b = {k: base[k].to(device) for k in names}
        d = {k: torch.stack([experts[t][k].to(device) - b[k] for t in tasks]) for k in names}
        return cls(names=names, base=b, deltas=d, experts=tasks)

    def subset(self, idx: list[int]) -> "Bank":
        ii = torch.tensor(idx, device=next(iter(self.deltas.values())).device)
        return Bank(self.names, self.base, {k: v.index_select(0, ii) for k, v in self.deltas.items()},
                    [self.experts[i] for i in idx])

    @property
    def k(self) -> int:
        return len(self.experts)

    @torch.no_grad()
    def merge(self, coeffs: dict[str, np.ndarray], alpha: float, default: np.ndarray | None = None,
              bias_policy: str = "tied") -> dict[str, torch.Tensor]:
        """``coeffs`` maps linear-map weight names to (K,) lambdas; others use ``default`` (uniform)."""
        k = self.k
        default = np.full(k, 1.0 / k) if default is None else default
        out = {}
        for name in self.names:
            lam = coeffs.get(name)
            if lam is None and bias_policy == "tied":
                owner = bias_owner(name)
                if owner is not None:
                    lam = coeffs.get(owner)
            if lam is None:
                lam = default
            lam_t = torch.as_tensor(np.asarray(lam, dtype=np.float32), device=self.deltas[name].device)
            d = self.deltas[name]
            out[name] = self.base[name] + alpha * torch.tensordot(lam_t, d, dims=1)
        return out


def layerwise_coeffs(score_table: dict[str, np.ndarray], beta: float, prior=None) -> dict[str, np.ndarray]:
    """Apply the softmax rule independently to each linear map's (K,) score vector."""
    return {p: softmax_coeffs(s, beta, prior) for p, s in score_table.items()}


def summarize_coeffs(coeffs: dict[str, np.ndarray]) -> dict[str, float]:
    if not coeffs:
        return {"n_eff_mean": math.nan, "n_eff_min": math.nan, "max_mean": math.nan}
    conc = [concentration(v) for v in coeffs.values()]
    return {"n_eff_mean": float(np.mean([c["n_eff"] for c in conc])),
            "n_eff_min": float(np.min([c["n_eff"] for c in conc])),
            "n_eff_median": float(np.median([c["n_eff"] for c in conc])),
            "max_mean": float(np.mean([c["max"] for c in conc])),
            "max_max": float(np.max([c["max"] for c in conc]))}


def mean_lambda_per_expert(coeffs: dict[str, np.ndarray], sizes: dict[str, int] | None = None) -> np.ndarray:
    """Average lambda per expert across linear maps (parameter-count weighted if sizes given)."""
    keys = list(coeffs)
    w = np.array([sizes[k] if sizes else 1.0 for k in keys], dtype=np.float64)
    m = np.stack([coeffs[k] for k in keys])
    return (w[:, None] * m).sum(0) / w.sum()
