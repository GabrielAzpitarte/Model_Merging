"""KnOTS LoRA bank: per-(matrix, expert) conditioning statistics, exactness and gauge checks.

Writes results/aa/lora/lora_stats.csv and results/aa/lora/lora_checks.json:
* QR-core nonzero spectrum vs dense SVD of gamma*B@A (exactness);
* ordinary kappa2 of the rank-16 update (numerically infinite) vs kappa_plus at several tolerances;
* full-weight kappa2(W0 + Delta) and kappa2(W0), with the interlacing bound sigma_q(W0+Delta) <= sigma_{q-r}(W0);
* factor condition numbers kappa(A), kappa(B) before / after random gauge transforms R (well and badly
  conditioned): the product and all product statistics must be unchanged, the factor scores change.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from condmerge.aa_common import layer_index, load_base_unprefixed, module_type  # noqa: E402
from condmerge.lora import core_svals, delta, gauge, kappa, load_knots_lora  # noqa: E402

TAUS = (1e-1, 1e-2, 1e-3, 1e-4)
EPS32 = float(np.finfo(np.float32).eps)


def main() -> None:
    dev = "cpu"  # float64 SVDs: CPU LAPACK is ~50x faster than fp64 on A6000
    torch.set_num_threads(16)
    out = RESULTS / "aa" / "lora"
    out.mkdir(parents=True, exist_ok=True)
    base = load_base_unprefixed()
    fac = {t: load_knots_lora(t) for t in TASKS8}
    names = sorted(fac[TASKS8[0]])
    rng = torch.Generator().manual_seed(0)
    rows, checks = [], {"core_vs_dense_max_rel_err": 0.0, "gauge": [], "interlacing_violations": 0}
    for n in names:
        w0 = base[n].double().to(dev)
        s0 = torch.linalg.svdvals(w0).cpu()
        q = s0.numel()
        for t in TASKS8:
            b, a, g = fac[t][n]
            d = delta(b, a, g).double().to(dev)
            sd = torch.linalg.svdvals(d).cpu()
            core = core_svals(b, a, g)
            r = core.numel()
            rel = float(((sd[:r] - core).abs() / core[0]).max())
            checks["core_vs_dense_max_rel_err"] = max(checks["core_vs_dense_max_rel_err"], rel)
            sw = torch.linalg.svdvals(w0 + d).cpu()
            if sw[-1] > s0[q - r - 1] * (1 + 1e-10):
                checks["interlacing_violations"] += 1
            row = {"param": n, "layer": layer_index(n), "module": module_type(n), "expert": t, "rank_config": r,
                   "gamma": g, "kappa2_W": float(sw[0] / sw[-1]), "kappa2_W0": float(s0[0] / s0[-1]),
                   "smin_W": float(sw[-1]), "smin_W0": float(s0[-1]), "s_q_minus_r_W0": float(s0[q - r - 1]),
                   "smax_W": float(sw[0]), "smax_W0": float(s0[0]),
                   "kappa2_D_dense": float(sd[0] / sd[-1]) if sd[-1] > 0 else math.inf,
                   "sd_tail_over_head": float(sd[-1] / sd[0]),
                   "fro_D": float(core.pow(2).sum().sqrt()), "spec_D": float(core[0]),
                   "stable_rank_D": float(core.pow(2).sum() / core[0] ** 2),
                   "entropy_rank_D": float(torch.exp(-((core ** 2 / core.pow(2).sum()) * (core ** 2 / core.pow(2).sum()).log()).sum())),
                   "kappa_A": kappa(a), "kappa_B": kappa(b),
                   "numrank_D_eps": int((core > max(b.shape[0], a.shape[1]) * EPS32 * core[0]).sum())}
            for tau in TAUS:
                rr = int((core > tau * core[0]).sum())
                row[f"kplus_D_{tau:g}"] = float(core[0] / core[rr - 1])
                row[f"numrank_D_{tau:g}"] = rr
            rows.append(row)
        # gauge check on one expert per matrix
        b, a, g = fac[TASKS8[0]][n]
        for cond in (1e1, 1e4):
            u = torch.linalg.qr(torch.randn(16, 16, generator=rng, dtype=torch.float64))[0]
            v = torch.linalg.qr(torch.randn(16, 16, generator=rng, dtype=torch.float64))[0]
            rmat = u @ torch.diag(torch.logspace(0, math.log10(cond), 16, dtype=torch.float64)) @ v.T
            b2, a2 = gauge(b.double(), a.double(), rmat)
            prod_err = float((b2 @ a2 - b.double() @ a.double()).norm() / (b.double() @ a.double()).norm())
            c1, c2 = core_svals(b, a, g), core_svals(b2, a2, g)
            checks["gauge"].append({"param": n, "R_cond": cond, "prod_rel_err": prod_err,
                                    "core_spectrum_rel_err": float(((c1 - c2).abs() / c1[0]).max()),
                                    "kappa_A_before": kappa(a), "kappa_A_after": kappa(a2),
                                    "kappa_B_before": kappa(b), "kappa_B_after": kappa(b2)})
        print(n, f"kappa0={rows[-1]['kappa2_W0']:.3e}",
              "kappa_W range", f"{min(r['kappa2_W'] for r in rows[-8:]):.3e}-{max(r['kappa2_W'] for r in rows[-8:]):.3e}", flush=True)
    with (out / "lora_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    g = checks["gauge"]
    checks["gauge_summary"] = {
        "max_prod_rel_err": max(x["prod_rel_err"] for x in g),
        "max_core_spectrum_rel_err": max(x["core_spectrum_rel_err"] for x in g),
        "median_kappaA_ratio_cond1e4": float(np.median([x["kappa_A_after"] / x["kappa_A_before"] for x in g if x["R_cond"] == 1e4])),
        "median_kappaB_ratio_cond1e4": float(np.median([x["kappa_B_after"] / x["kappa_B_before"] for x in g if x["R_cond"] == 1e4])),
    }
    (out / "lora_checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps({k: v for k, v in checks.items() if k != "gauge"}, indent=1))


if __name__ == "__main__":
    main()
