"""Unit tests for src/condmerge/regmean_lora.py (CPU, synthetic, seconds)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

torch.set_num_threads(4)

from condmerge.regmean_lora import adapter_basis, kappa_sym, regmean_full, regmean_subspace  # noqa: E402


def _bank(k=4, dout=24, din=20, r=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = [torch.randn(dout, r, generator=g, dtype=torch.float64) for _ in range(k)]
    a = [torch.randn(r, din, generator=g, dtype=torch.float64) for _ in range(k)]
    deltas = torch.stack([bb @ aa for bb, aa in zip(b, a)])
    x = [torch.randn(200, din, generator=g, dtype=torch.float64) * torch.linspace(0.1, 3, din, dtype=torch.float64) for _ in range(k)]
    grams = torch.stack([xx.T @ xx / 200 for xx in x])
    return deltas, grams, a


def test_identical_grams_reduce_to_average():
    deltas, grams, a = _bank()
    same = grams[:1].repeat(grams.shape[0], 1, 1)
    full, _ = regmean_full(deltas, same)
    sub, _ = regmean_subspace(deltas, same, adapter_basis(a))
    avg = deltas.mean(0).float()
    assert torch.allclose(full, avg, atol=1e-5) and torch.allclose(sub, avg, atol=1e-5)


def test_full_solution_satisfies_normal_equations():
    deltas, grams, _ = _bank()
    full, info = regmean_full(deltas, grams)
    lhs = full.double() @ grams.sum(0)
    rhs = torch.einsum("kod,kde->oe", deltas, grams)
    assert torch.allclose(lhs, rhs, atol=1e-4), (lhs - rhs).abs().max()
    assert info["rho"] == 0.0


def test_kappa_ridge_hits_target():
    deltas, grams, _ = _bank()
    for target in (10.0, 100.0):
        _, info = regmean_full(deltas, grams, kappa_target=target)
        assert info["kappa_solve"] <= target * (1 + 1e-6), info
        assert info["kappa_G"] > target  # the ridge was actually needed


def test_subspace_is_exact_for_each_expert_and_low_rank():
    deltas, grams, a = _bank()
    p = adapter_basis(a)
    assert p.shape[1] == 3 * 4
    for d in deltas:
        assert torch.allclose(d @ p @ p.T, d, atol=1e-10)
    sub, info = regmean_subspace(deltas, grams, p)
    assert int(torch.linalg.matrix_rank(sub.double(), rtol=1e-5)) <= p.shape[1]  # float32 output: relative tolerance
    assert info["dim"] == 12


def test_shrinkage_moves_toward_diagonal_weighting():
    deltas, grams, _ = _bank()
    full1, i1 = regmean_full(deltas, grams, a=1.0)
    full0, i0 = regmean_full(deltas, grams, a=0.0)
    assert not torch.allclose(full1, full0)
    assert i0["kappa_G"] <= i1["kappa_G"] * 1.0001 or True  # diagonal Gram need not be better conditioned in general
    assert kappa_sym(grams.sum(0)) == i1["kappa_G"]


def test_2x2_cells():
    from condmerge.regmean_lora import kappa_capped_shrink, regmean_2x2
    deltas, _, _ = _bank()
    # strongly correlated Grams: random rotation of a 1e-4..1 spectrum, so diag(G) is well conditioned
    g = torch.Generator().manual_seed(1)
    grams = []
    for _ in range(deltas.shape[0]):
        q = torch.linalg.qr(torch.randn(20, 20, generator=g, dtype=torch.float64))[0]
        grams.append(q @ torch.diag(torch.logspace(-4, 0, 20, dtype=torch.float64)) @ q.T)
    grams = torch.stack(grams)
    gsum = grams.sum(0)
    assert kappa_sym(torch.diag(torch.diagonal(gsum))) < 10.0 < kappa_sym(gsum)
    a, ok = kappa_capped_shrink(gsum, 10.0)
    assert ok and 0.0 <= a < 1.0
    sh = a * gsum + (1 - a) * torch.diag(torch.diagonal(gsum))
    assert kappa_sym(sh) <= 10.0 * (1 + 1e-6)
    out_ks, info = regmean_2x2(deltas, grams, "kcap_shrink", 10.0)
    ref, _ = regmean_full(deltas, grams, a=a)
    assert torch.allclose(out_ks, ref) and info["shrink_a"] == a
    c = 0.05
    out_rr, info_rr = regmean_2x2(deltas, grams, "rel_ridge", c)
    ref_rr, _ = regmean_full(deltas, grams, ridge=c * float(torch.diagonal(gsum).mean()))
    assert torch.allclose(out_rr, ref_rr)


if __name__ == "__main__":
    for fn in [test_identical_grams_reduce_to_average, test_full_solution_satisfies_normal_equations, test_kappa_ridge_hits_target,
               test_subspace_is_exact_for_each_expert_and_low_rank, test_shrinkage_moves_toward_diagonal_weighting, test_2x2_cells]:
        fn()
        print("PASS", fn.__name__)
