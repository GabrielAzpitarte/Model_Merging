"""2x2 decomposition of the kappa-cap edge on LoRA RegMean: {isotropic ridge, diag shrink} x {kappa-adaptive, fixed}.

KnOTS bank, C budget, full solve, alpha fixed at 0.8 (dev-best), Grams from N=512 and N=16 (3 calib subsets).
Cell grids: kcap_ridge and kcap_shrink at kappa_t in {1e3, 1e4}; fixed_shrink a in {0.95, 0.9}; rel_ridge with
c matched to the typical kcap ridge: c = median_maps(lambda_max / (tr G / d)) / kappa_t (computed from the N=512
Grams), ridge to zero and to mean. Dev. Writes results/aa/lora/regmean_2x2_dev.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condmerge.paths import CACHE, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

from aa_lora_regmean import input_name  # noqa: E402
from aa_lora_replicate_banks import load_bank  # noqa: E402
from condmerge.aa_common import TaskData, load_base_unprefixed, new_model  # noqa: E402
from condmerge.regmean_lora import regmean_2x2  # noqa: E402

ALPHA = 0.8


def main() -> None:
    dev = "cuda"
    t0 = time.time()
    base = load_base_unprefixed()
    base_dev = {k: v.to(dev) for k, v in base.items()}
    bank = load_bank("knots_release")
    names = sorted(bank[TASKS8[0]])
    deltas = {n: torch.stack([bank[t][n] for t in TASKS8]).double() for n in names}
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    gram_sets = {"n512": "", "n16_s0": "_n16", "n16_s1": "_n16_s1", "n16_s2": "_n16_s2"}
    g512 = torch.load(CACHE / "aa" / "grams_lora_knots_release_C.pt", weights_only=False)["grams"]
    ratios = []
    for n in names:
        gs = g512[input_name(n)].double().sum(0)
        ev = torch.linalg.eigvalsh(gs)
        ratios.append(float(ev[-1] / torch.diagonal(gs).mean()))
    r = float(np.median(ratios))
    print(f"median lambda_max / (tr/d) over maps = {r:.2f}", flush=True)
    cells = [("kcap_ridge", 1e3, "zero"), ("kcap_ridge", 1e4, "zero"), ("kcap_shrink", 1e3, None), ("kcap_shrink", 1e4, None),
             ("fixed_shrink", 0.95, None), ("fixed_shrink", 0.9, None),
             ("rel_ridge", r / 1e3, "zero"), ("rel_ridge", r / 1e4, "zero"), ("rel_ridge", r / 1e3, "mean"), ("rel_ridge", r / 1e4, "mean")]
    out = (RESULTS / "aa" / "lora" / "regmean_2x2_dev.jsonl").open("a")
    for gname, sfx in gram_sets.items():
        g = torch.load(CACHE / "aa" / f"grams_lora_knots_release_C{sfx}.pt", weights_only=False)["grams"]
        for cell, param, rt in cells:
            params = dict(base_dev)
            infos = []
            for n in names:
                m, info = regmean_2x2(deltas[n], g[input_name(n)].double(), cell, param, ridge_to=rt or "zero")
                params[n] = base_dev[n] + ALPHA * m.to(dev)
                infos.append(info)
            res = data.evaluate(model, "dev", params=params)
            rec = {"grams": gname, "cell": cell, "param": param, "ridge_to": rt, "alpha": ALPHA, "avg_acc": res["avg_acc"],
                   "acc": res["acc"], "median_rho": float(np.median([i.get("rho", 0.0) for i in infos])),
                   "median_kappa_solve": float(np.median([i["kappa_solve"] for i in infos])),
                   "median_shrink_a": float(np.median([i.get("shrink_a", float("nan")) for i in infos])),
                   "frac_cap_unreachable": float(np.mean([not i.get("cap_reachable", True) for i in infos]))}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{gname} {cell:12s} {param:.3g} {rt}: {100 * res['avg_acc']:.2f} (median rho {rec['median_rho']:.3g}, "
                  f"kappa_solve {rec['median_kappa_solve']:.3g}, a {rec['median_shrink_a']:.3f}, unreachable {rec['frac_cap_unreachable']:.2f}) "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
