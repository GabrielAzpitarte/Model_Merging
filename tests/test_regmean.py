"""Exactness checks for the RegMean-type solver (review sec. 3.6)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.regmean import gram_kappa, kappa_ridge, regmean_solve  # noqa: E402


def _spd(d: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4 * d, d, generator=g, dtype=torch.float64) * scale
    return x.T @ x / (4 * d)


def test_identical_grams_give_average():
    g = _spd(12, 0)
    ds = [torch.randn(5, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(i)) for i in range(3)]
    out = regmean_solve(ds, [g, g, g])
    assert torch.allclose(out, sum(ds) / 3, atol=1e-10)


def test_normal_equation_and_regression_optimum():
    gs = [_spd(10, s) for s in range(4)]
    ds = [torch.randn(6, 10, dtype=torch.float64, generator=torch.Generator().manual_seed(10 + i)) for i in range(4)]
    out = regmean_solve(ds, gs)
    lhs = out @ sum(gs)
    rhs = sum(d @ g for d, g in zip(ds, gs))
    assert torch.allclose(lhs, rhs, atol=1e-9)


def test_kappa_cap_and_ridge_limits():
    g = _spd(16, 3)
    g[0, 0] *= 1e-6  # make it ill-conditioned
    g = 0.5 * (g + g.T)
    rho, kb, ka = kappa_ridge(g, 100.0)
    assert kb > 100 and math.isclose(ka, 100.0, rel_tol=1e-8)
    assert math.isclose(gram_kappa(g + rho * torch.eye(16, dtype=torch.float64)), 100.0, rel_tol=1e-6)
    ds = [torch.randn(3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(i)) for i in range(2)]
    far = regmean_solve(ds, [g, _spd(16, 4)], ridge=1e12, ridge_to="mean")
    assert torch.allclose(far, sum(ds) / 2, atol=1e-6)
    zero = regmean_solve(ds, [g, _spd(16, 4)], ridge=1e12, ridge_to="zero")
    assert float(zero.abs().max()) < 1e-6


def test_shrink_zero_is_coordinatewise_weighting():
    gs = [_spd(8, s) for s in range(3)]
    ds = [torch.randn(4, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(20 + i)) for i in range(3)]
    out = regmean_solve(ds, gs, shrink=0.0)
    w = torch.stack([torch.diagonal(g) for g in gs])  # (K, din)
    ref = sum(d * w[i][None, :] for i, d in enumerate(ds)) / w.sum(0)[None, :]
    assert torch.allclose(out, ref, atol=1e-10)
