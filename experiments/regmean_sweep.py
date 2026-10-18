"""Inverse-problem route: RegMean-type merges with fixed shrinkage vs conditioning-guided ridge.

Hypothesis (review sec. 4.1, row 3, "the instability lies in the solve"): small eigenvalues of
the inverted Gram G = sum_i G_i amplify noise in the estimated statistics; a regulariser set by
the condition number of G (kappa cap) should beat a fixed shrinkage, most visibly when the
calibration statistics are noisy (few images).

Per linear map: D = solve(sum_i D_i G~_i [+ rho R], G [+ rho I]) (src/condmerge/regmean.py);
non-linear parameters: uniform average. Final weights W0 + alpha * D (alpha = 1: standard RegMean).
Budgets: grams 'C' = expert i on task i's calib (task-separated unlabelled), 'B' = expert i on the
pooled calib (no task ids). Selection of (method, alpha) on dev (D), as for every rule.
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

from condmerge.paths import CACHE_ARCH, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.clip import Evaluator, load_base_state, load_experts, summarize  # noqa: E402
from condmerge.regmean import regmean_solve  # noqa: E402
from condmerge.runlog import append_record, config_key, done_keys  # noqa: E402
from condmerge.spectra import as_matrix, linear_map_names  # noqa: E402

SHRINKS = (1.0, 0.99, 0.95, 0.9, 0.7, 0.5, 0.0)
KAPPAS = (1e2, 1e3, 1e4, 1e5, 1e6, 1e8)


def gram_file(expert: str, budget: str, tag: str) -> Path:
    d = CACHE_ARCH / "grams"
    base = f"{expert}__task-{expert}" if budget == "C" else f"{expert}__pool"
    return d / f"{base}{tag}.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="C", choices=["C", "B"])
    ap.add_argument("--gram-tag", default="", help="suffix of gram files (e.g. _n32_s0 for subsampled stats)")
    ap.add_argument("--methods", default="shrink,kcap_mean,kcap_zero")
    ap.add_argument("--shrinks", default=",".join(map(str, SHRINKS)))
    ap.add_argument("--kappas", default=",".join(f"{k:g}" for k in KAPPAS))
    ap.add_argument("--alphas", default="1.0,1.2,1.4,1.6,2.0")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or RESULTS / "regmean" / f"regmean_{args.budget}{args.gram_tag}_{args.split}.jsonl")
    done = done_keys(out)
    t0 = time.time()
    base = load_base_state()
    experts = load_experts(TASKS8)
    names = linear_map_names(base)
    grams = {e: torch.load(gram_file(e, args.budget, args.gram_tag), weights_only=False) for e in TASKS8}
    inputs_of = grams[TASKS8[0]]["inputs_of"]
    n_img = {e: grams[e]["n_images"] for e in TASKS8}
    ev = Evaluator(args.device, splits=(args.split,), tasks=TASKS8)
    ref = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"][args.split]
    dev = args.device
    deltas = {p: [as_matrix(p, experts[e][p]).double() - as_matrix(p, base[p]).double() for e in TASKS8] for p in names}
    print(f"ready {time.time() - t0:.0f}s; grams {args.budget}{args.gram_tag} n_images {n_img}", flush=True)
    configs = []
    for m in args.methods.split(","):
        if m == "shrink":
            configs += [("shrink", float(a), None, "mean") for a in args.shrinks.split(",")]
        elif m in ("kcap_mean", "kcap_zero"):
            configs += [(m, 1.0, float(k), m.split("_")[1]) for k in args.kappas.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    for method, shrink, kt, ridge_to in configs:
        cfgs = [{"method": method, "shrink": shrink, "kappa_target": kt, "ridge_to": ridge_to, "alpha": a,
                 "budget": args.budget, "gram_tag": args.gram_tag, "split": args.split} for a in alphas]
        if all(config_key(c) in done for c in cfgs):
            continue
        t1 = time.time()
        merged_delta, info = {}, {}
        for p in names:
            gs = [grams[e]["grams"][inputs_of[p]].to(dev) for e in TASKS8]
            ds = [d.to(dev) for d in deltas[p]]
            sol, inf_ = regmean_solve(ds, gs, shrink=shrink, kappa_target=kt, ridge_to=ridge_to, return_info=True)
            merged_delta[p] = sol.reshape(base[p].shape).float().cpu()
            info[p] = inf_
        kb = [v["kappa_before"] for v in info.values() if not math.isnan(v["kappa_before"])]
        ka = [v["kappa_after"] for v in info.values() if not math.isnan(v["kappa_after"])]
        solve_s = time.time() - t1
        for c in cfgs:
            if config_key(c) in done:
                continue
            a = c["alpha"]
            sd = {}
            for k in base:
                if k in merged_delta:
                    sd[k] = base[k] + a * merged_delta[k]
                else:
                    sd[k] = base[k] + a * torch.stack([experts[e][k] - base[k] for e in TASKS8]).mean(0)
            acc = ev.evaluate(sd, args.split)
            m = summarize(acc, ref)
            append_record(out, c, {**m, "acc": acc},
                          {"kappa_before_median": float(np.median(kb)) if kb else None,
                           "kappa_after_median": float(np.median(ka)) if ka else None,
                           "kappa_after_max": float(np.max(ka)) if ka else None,
                           "rho_median": float(np.median([v["rho"] for v in info.values()])),
                           "solve_seconds": solve_s, "n_images": n_img})
            print(f"{method:9s} shrink={shrink:<5g} kt={kt!s:8s} a={a:<4g} norm={100 * m['mean_norm']:.2f} "
                  f"acc={100 * m['mean_acc']:.2f} min={100 * m['min_norm']:.1f}  "
                  f"[k_before med {np.median(kb) if kb else float('nan'):.3g}, after med {np.median(ka) if ka else float('nan'):.3g}]",
                  flush=True)
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
