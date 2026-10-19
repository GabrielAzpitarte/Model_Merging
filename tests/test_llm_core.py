"""Unit tests of the LLM extension's coefficient policies and exact concatenated-LoRA merge (CPU only)."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.llm import rules  # noqa: E402
from condmerge.llm.runner import ConcatLoRA  # noqa: E402


def test_uniform_branch_and_argmin_ties():
    t = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 9.0]])
    lam, info = rules.coefficients(t, 0.0)
    assert info["branch"] == "uniform" and np.allclose(lam, 1 / 3)
    lam, info = rules.coefficients(t, math.inf)
    assert np.allclose(lam[0], [1, 0, 0]) and np.allclose(lam[1], [0.5, 0.5, 0]) and info["n_tie_rows"] == 1


def test_infinite_scores():
    t = np.array([[np.inf, 1.0, 2.0], [np.inf, np.inf, np.inf]])
    lam, info = rules.coefficients(t, 1.0)
    assert lam[0, 0] == 0 and abs(lam[0].sum() - 1) < 1e-12
    assert np.allclose(lam[1], 1 / 3) and info["n_all_inf_rows"] == 1 and info["n_inf_scores"] == 4


def test_softmax_direction_and_scale_override():
    t = np.array([[1.0, 2.0], [10.0, 30.0]])
    lam, info = rules.coefficients(t, 1.0)
    assert lam[0, 0] > lam[0, 1] and lam[1, 0] > lam[1, 1]  # lower score -> larger lambda
    sc = rules.scale_of(t)
    lam2, info2 = rules.coefficients(t * 1.0, 2.0, scale=2 * sc)
    assert np.allclose(lam, lam2) and abs(info2["beta"] - info["beta"]) < 1e-15


def test_common_shift_invariance():
    """log kappa_i - log kappa0 (common within a matrix) gives the same softmax."""
    t = np.log(np.array([[100.0, 120.0, 150.0], [1e6, 3e6, 2e5]]))
    sc = rules.scale_of(t)
    shift = np.array([[np.log(90.0)], [np.log(5e5)]])
    a, _ = rules.coefficients(t, 1.0, scale=sc)
    b, _ = rules.coefficients(t - shift, 1.0, scale=sc)
    assert np.allclose(a, b)


def test_neff_and_jitter():
    lam = np.array([[1.0, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]])
    assert np.allclose(rules.n_eff(lam), [1.0, 4.0])
    j = rules.jitter(np.full((3, 4), 0.25), 1e-7, 0)
    assert np.allclose(j.sum(1), 1) and np.abs(j - 0.25).max() < 1e-6 and np.abs(j - 0.25).max() > 0


def test_concat_lora_exact():
    torch.manual_seed(0)
    k, r, din, dout = 3, 16, 40, 24
    base = torch.nn.Linear(din, dout, bias=False)
    a = [np.random.default_rng(i).standard_normal((r, din)).astype(np.float32) * 0.1 for i in range(k)]
    b = [np.random.default_rng(10 + i).standard_normal((dout, r)).astype(np.float32) * 0.1 for i in range(k)]
    w = ConcatLoRA(base, np.concatenate(a, 0), np.concatenate(b, 1), "cpu")
    lam = np.array([0.2, 0.5, 0.3])
    alpha = 1.6
    w.set_w(alpha * np.repeat(lam, r))
    x = torch.randn(5, din)
    dense = base.weight.detach().double() + alpha * sum(
        lam[i] * torch.from_numpy(b[i]).double() @ torch.from_numpy(a[i]).double() for i in range(k))
    ref = x.double() @ dense.T
    out = w(x).double()
    assert torch.allclose(out, ref, atol=1e-5)
    # extras: an additional low-rank term
    u, v = np.random.default_rng(5).standard_normal((dout, 2)).astype(np.float32), \
        np.random.default_rng(6).standard_normal((2, din)).astype(np.float32)
    w.extra = [(torch.from_numpy(u), torch.from_numpy(v), 0.7)]
    ref2 = ref + 0.7 * x.double() @ (torch.from_numpy(u).double() @ torch.from_numpy(v).double()).T
    assert torch.allclose(w(x).double(), ref2, atol=1e-5)
