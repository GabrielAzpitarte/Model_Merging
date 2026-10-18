"""Does the RegMean solve become unstable with noisy statistics, and does a kappa cap fix it?

Review sec. 4.1 row 3: "Weak eigenvalues of the actual aggregate system amplify noisy
statistics; stable regularisation fixes it." Here the statistics' noise is controlled by the
number of calibration images per task (budget C: each expert on its own task's calib subset,
labels unused). For each (n_images, seed): per-expert Grams in memory, then fixed RegMean
shrinkage a vs the kappa-capped ridge toward the uniform average, at alpha in {1.0, 1.4}; dev only.
Evidence FOR: unregularised/fixed-shrink accuracy drops and varies across seeds as n shrinks while
the kappa cap stays stable. AGAINST: no degradation, or the cap gives no advantage over shrinkage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import BASE_MODEL, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.clip import Evaluator, load_base_state, load_experts, summarize  # noqa: E402
from condmerge.data import load_cache  # noqa: E402
from condmerge.grams import collect_grams, input_name  # noqa: E402
from condmerge.regmean import regmean_solve  # noqa: E402
from condmerge.runlog import append_record, config_key, done_keys  # noqa: E402
from condmerge.spectra import as_matrix, linear_map_names  # noqa: E402


def main() -> None:
    from transformers import CLIPVisionConfig, CLIPVisionModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="16,64,512")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--shrinks", default="1.0,0.95,0.9,0.5,0.0")
    ap.add_argument("--kappas", default="1e2,1e3,1e4,1e5,1e6")
    ap.add_argument("--alphas", default="1.0,1.4")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = RESULTS / "regmean" / f"instability_C_{args.split}.jsonl"
    done = done_keys(out)
    dev = args.device
    t0 = time.time()
    base = load_base_state()
    experts = load_experts(TASKS8)
    names = linear_map_names(base)
    ev = Evaluator(dev, splits=(args.split,), tasks=TASKS8)
    ref = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"][args.split]
    model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(BASE_MODEL)).eval().to(dev)
    deltas = {p: [as_matrix(p, experts[e][p]).double() - as_matrix(p, base[p]).double() for e in TASKS8] for p in names}
    calib = {e: load_cache(e, "calib", mmap=True)[0] for e in TASKS8}
    alphas = [float(a) for a in args.alphas.split(",")]
    methods = [("shrink", float(a), None) for a in args.shrinks.split(",")] + \
              [("kcap_mean", 1.0, float(k)) for k in args.kappas.split(",")]
    for n in [int(x) for x in args.ns.split(",")]:
        for seed in ([0] if n >= 512 else [int(s) for s in args.seeds.split(",")]):
            todo = [(m, a, k, al) for (m, a, k) in methods for al in alphas
                    if config_key({"method": m, "shrink": a, "kappa_target": k, "alpha": al, "n": n, "seed": seed}) not in done]
            if not todo:
                continue
            t1 = time.time()
            grams = {}
            for e in TASKS8:
                model.load_state_dict({k: v.to(dev) for k, v in experts[e].items()}, strict=True)
                x = calib[e]
                if n < x.shape[0]:
                    x = x[np.sort(np.random.default_rng(seed).choice(x.shape[0], n, replace=False))]
                grams[e] = collect_grams(model, [x], names, dev)["grams"]
            inputs = {p: input_name(p) for p in names}
            print(f"[n={n} seed={seed}] grams {time.time() - t1:.0f}s", flush=True)
            for method, shrink, kt in methods:
                cfgs = [{"method": method, "shrink": shrink, "kappa_target": kt, "alpha": al, "n": n, "seed": seed}
                        for al in alphas]
                if all(config_key(c) in done for c in cfgs):
                    continue
                merged, kb, ka = {}, [], []
                for p in names:
                    gs = [grams[e][inputs[p]] for e in TASKS8]
                    sol, info = regmean_solve([d.to(dev) for d in deltas[p]], gs, shrink=shrink, kappa_target=kt,
                                              ridge_to="mean", return_info=True)
                    merged[p] = sol.reshape(base[p].shape).float().cpu()
                    if not math.isnan(info["kappa_before"]):
                        kb.append(info["kappa_before"])
                        ka.append(info["kappa_after"])
                for c in cfgs:
                    if config_key(c) in done:
                        continue
                    a = c["alpha"]
                    sd = {k: base[k] + a * (merged[k] if k in merged else
                                            torch.stack([experts[e][k] - base[k] for e in TASKS8]).mean(0)) for k in base}
                    acc = ev.evaluate(sd, args.split)
                    m = summarize(acc, ref)
                    append_record(out, c, {**m, "acc": acc},
                                  {"kappa_before_median": float(np.median(kb)) if kb else None,
                                   "kappa_before_max": float(np.max(kb)) if kb else None,
                                   "kappa_after_median": float(np.median(ka)) if ka else None})
                    print(f"  n={n:<4d} s={seed} {method:9s} a_shrink={shrink:<4g} kt={kt!s:7s} alpha={a:<4g} "
                          f"norm={100 * m['mean_norm']:.2f} min={100 * m['min_norm']:.1f} "
                          f"[kappa(G) med {np.median(kb) if kb else float('nan'):.3g} -> {np.median(ka) if ka else float('nan'):.3g}]",
                          flush=True)
            del grams
            torch.cuda.empty_cache()
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
