"""LoRA merge correctness on the real KnOTS bank (CPU): products, not factors; exact vs concatenated; gauge."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)  # shared machine: never use torch's default (all cores)

from condmerge.paths import TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.lora import LoRABank, core_svals, gauge, load_knots_lora  # noqa: E402

_BANK = {}


def bank() -> LoRABank:
    if "b" not in _BANK:
        _BANK["b"] = LoRABank(TASKS8, "cpu")
    return _BANK["b"]


def test_one_hot_merge_is_the_expert():
    b = bank()
    for i, t in enumerate(TASKS8):
        lam = torch.zeros(8)
        lam[i] = 1.0
        merged = b.merged({}, 1.0, default=lam)
        for n in b.linear[:6]:
            bb, aa, g = b.factors[t][n]
            assert torch.allclose(merged[n], b.base[n] + g * bb @ aa, atol=1e-6)
        untouched = [k for k in b.names if k not in b.linear]
        assert all(torch.equal(merged[k], b.base[k]) for k in untouched)


def test_exact_merge_equals_concatenated_adapter():
    b = bank()
    lam = torch.tensor(np.random.default_rng(0).dirichlet(np.ones(8)), dtype=torch.float32)
    alpha = 0.8
    merged = b.merged({}, alpha, default=lam)
    for n in b.linear[:6]:
        bcat = torch.cat([alpha * lam[i] * b.factors[t][n][2] * b.factors[t][n][0] for i, t in enumerate(TASKS8)], 1)
        acat = torch.cat([b.factors[t][n][1] for t in TASKS8], 0)
        assert bcat.shape[1] == 8 * 16
        assert torch.allclose(merged[n], b.base[n] + bcat @ acat, atol=1e-5)


def test_factor_averaging_is_not_product_averaging():
    b = bank()
    n = b.linear[0]
    bavg = sum(b.factors[t][n][0] for t in TASKS8) / 8
    aavg = sum(b.factors[t][n][1] for t in TASKS8) / 8
    exact = b.merged({}, 1.0)[n] - b.base[n]
    rel = float((bavg @ aavg - exact).norm() / exact.norm())
    assert rel > 0.3, rel


def test_gauge_invariance_of_product_merge():
    b = bank()
    t, n = TASKS8[2], b.linear[5]
    bb, aa, g = b.factors[t][n]
    r = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))[0] @ torch.diag(torch.logspace(0, 3, 16, dtype=torch.float64))
    b2, a2 = gauge(bb.double(), aa.double(), r)
    assert torch.allclose(b2 @ a2, bb.double() @ aa.double(), atol=1e-8)
    assert torch.allclose(core_svals(b2, a2, g), core_svals(bb, aa, g), rtol=1e-8)


def test_rank16_compression_is_rank16_and_exact_merge_is_not():
    b = bank()
    exact = b.merged({}, 1.0)
    comp = b.merged({}, 1.0, compress_rank=16)
    n = b.linear[3]
    s_exact = torch.linalg.svdvals((exact[n] - b.base[n]).double())
    s_comp = torch.linalg.svdvals((comp[n] - b.base[n]).double())
    assert int((s_comp > 1e-6 * s_comp[0]).sum()) <= 16
    assert int((s_exact > 1e-6 * s_exact[0]).sum()) > 16


if __name__ == "__main__":
    for f in [test_one_hot_merge_is_the_expert, test_exact_merge_equals_concatenated_adapter,
              test_factor_averaging_is_not_product_averaging, test_gauge_invariance_of_product_merge,
              test_rank16_compression_is_rank16_and_exact_merge_is_not]:
        f()
        print("PASS", f.__name__)
