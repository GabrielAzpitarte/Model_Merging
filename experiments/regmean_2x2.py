"""2x2 mechanism check for the small, consistent edge of the kappa-capped ridge over RegMean shrinkage.

Factors: regulariser GEOMETRY {isotropic ridge (rotation-invariant), axis-aligned diag shrinkage}
x STRENGTH {kappa-adaptive per map, fixed}. Cells (all ridge-to-mean where a ridge is used):
  1 kcap_ridge   : rho per map from kappa2(G + rho I) <= kappa_t
  2 kcap_shrink  : a per map, largest with kappa2(a G + (1-a) diag G) <= kappa_t
  3 fixed_shrink : a fixed (RegMean default 0.95)
  4 rel_ridge    : rho = c tr(G)/d with c = median_maps(lambda_max / (tr/d)) / kappa_t, so the
                   typical strength matches cell 1 (only WHERE the ridge goes differs)
Verdict rule (pre-declared): if cell 2 keeps cell 1's edge over cell 3 and cell 4 loses it ->
the edge is about conditioning; the reverse -> it is about isotropy / rotation invariance.
Budget C, alpha = 1, n in {16 (3 subsets), 512}.
Post-hoc strength sweep (equal grids for the kappa cap and the fixed relative ridge):
--cells kcap_ridge,rel_ridge --kappas <grid> --out <separate file>.
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--split", default="dev")
    ap.add_argument("--sets", default="16:0,16:1,16:2,512:0")
    ap.add_argument("--kappas", default="1e5,1e4")
    ap.add_argument("--shrinks", default="0.95,0.9")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cells", default="kcap_ridge,kcap_shrink,rel_ridge,fixed_shrink")
    ap.add_argument("--rel-kappas", default="", help="kappa_t list defining the rel_ridge strengths (default: --kappas)")
    ap.add_argument("--out", default="", help="output jsonl (default: results/regmean/twobytwo_C_<split>.jsonl)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else RESULTS / "regmean" / f"twobytwo_C_{args.split}.jsonl"
    wanted = set(args.cells.split(","))
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
    inputs = {p: input_name(p) for p in names}
    kappas = [float(k) for k in args.kappas.split(",")]
    rel_kappas = [float(k) for k in args.rel_kappas.split(",") if k]
    for spec in args.sets.split(","):
        n, seed = (int(x) for x in spec.split(":"))
        grams = {}
        for e in TASKS8:
            model.load_state_dict({k: v.to(dev) for k, v in experts[e].items()}, strict=True)
            x = calib[e]
            if n < x.shape[0]:
                x = x[np.sort(np.random.default_rng(seed).choice(x.shape[0], n, replace=False))]
            grams[e] = collect_grams(model, [x], names, dev)["grams"]
        # strength matching for cell 4: c = median over maps of lambda_max / (tr/d), divided by kappa_t
        ratios = []
        for key in sorted({inputs[p] for p in names}):
            g = sum(grams[e][key].to(dev).double() for e in TASKS8)
            lmax = float(torch.linalg.eigvalsh(0.5 * (g + g.T))[-1])
            ratios.append(lmax / (float(torch.trace(g)) / g.shape[0]))
        r_med = float(np.median(ratios))
        cells = []
        for kt in kappas:
            cells += [("kcap_ridge", {"kappa_target": kt}, kt), ("kcap_shrink", {"kappa_shrink_target": kt}, kt)]
            if not rel_kappas:
                cells.append(("rel_ridge", {"rel_ridge": r_med / kt}, kt))
        cells += [("rel_ridge", {"rel_ridge": r_med / kt}, kt) for kt in rel_kappas]
        cells += [("fixed_shrink", {"shrink": float(a)}, None) for a in args.shrinks.split(",") if a]
        cells = [c for c in cells if c[0] in wanted]
        print(f"[n={n} seed={seed}] grams done {time.time() - t0:.0f}s; median lambda_max/(tr/d) = {r_med:.3g}", flush=True)
        for cell, kw, kt in cells:
            cfg = {"cell": cell, "kappa_t": kt, "shrink": kw.get("shrink"), "n": n, "seed": seed, "alpha": args.alpha}
            if config_key(cfg) in done:
                continue
            t1 = time.time()
            merged, shrinks, rhos = {}, [], []
            for p in names:
                gs = [grams[e][inputs[p]] for e in TASKS8]
                sol, info = regmean_solve([d.to(dev) for d in deltas[p]], gs, ridge_to="mean", return_info=True, **kw)
                merged[p] = sol.reshape(base[p].shape).float().cpu()
                shrinks.append(info["shrink"])
                rhos.append(info["rho"] / max(info["trace_mean"], 1e-300))
            a = args.alpha
            sd = {k: base[k] + a * (merged[k] if k in merged else
                                    torch.stack([experts[e][k] - base[k] for e in TASKS8]).mean(0)) for k in base}
            acc = ev.evaluate(sd, args.split)
            m = summarize(acc, ref)
            append_record(out, cfg, {**m, "acc": acc},
                          {"shrink_median": float(np.median(shrinks)), "shrink_min": float(np.min(shrinks)),
                           "rel_rho_median": float(np.median(rhos)), "c_rel": kw.get("rel_ridge"),
                           "lambda_ratio_median": r_med, "seconds": time.time() - t1})
            print(f"  n={n:<4d} s={seed} {cell:13s} kt={kt!s:8s} shrink={kw.get('shrink')!s:5s} "
                  f"norm={100 * m['mean_norm']:.2f} min={100 * m['min_norm']:.1f} "
                  f"[a med {np.median(shrinks):.3g}, rho/(tr/d) med {np.median(rhos):.2e}] {time.time() - t1:.0f}s", flush=True)
        del grams
        torch.cuda.empty_cache()
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
