"""H-INV0 for full fine-tuning (CLIP ViT-B/32, budget C, n = 16 images per task, subsets 0-2, and n = 512):
is plain RegMean's low accuracy (Report 2: ~65 vs ~88 normalised at n = 16) a numerical failure of the fp16
evaluation or genuine statistical overfitting of the solve?

Same Grams as experiments/regmean_2x2.py (calib subsets by seed). Merges: plain RegMean (a = 1, no ridge),
kappa-cap ridge 1e5 (to mean), RegMean shrinkage 0.95, at alpha = 1. Each evaluated on dev with fp16 autocast
(the pipeline's precision) and in fp32; finiteness of features, max |W|, top-class share.
Writes results/inverse/collapse_audit_fullft.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from condmerge.paths import BASE_MODEL, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

from condmerge.clip import Evaluator, load_base_state, load_experts, summarize  # noqa: E402
from condmerge.data import load_cache  # noqa: E402
from condmerge.grams import collect_grams, input_name  # noqa: E402
from condmerge.regmean import regmean_solve  # noqa: E402
from condmerge.spectra import as_matrix, linear_map_names  # noqa: E402

OUT = ROOT / "results" / "inverse"


def main() -> None:
    from transformers import CLIPVisionConfig, CLIPVisionModel

    dev = "cuda:0"
    t0 = time.time()
    base = load_base_state()
    experts = load_experts(TASKS8)
    names = linear_map_names(base)
    ref = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"]["dev"]
    ev16 = Evaluator(dev, splits=("dev",), tasks=TASKS8, amp=True)
    ev32 = Evaluator(dev, splits=("dev",), tasks=TASKS8, amp=False)
    model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(BASE_MODEL)).eval().to(dev)
    deltas = {p: [as_matrix(p, experts[e][p]).double() - as_matrix(p, base[p]).double() for e in TASKS8] for p in names}
    calib = {e: load_cache(e, "calib", mmap=True)[0] for e in TASKS8}
    report = {}
    for n, seed in ((16, 0), (16, 1), (16, 2), (512, 0)):
        grams = {}
        for e in TASKS8:
            model.load_state_dict({k: v.to(dev) for k, v in experts[e].items()}, strict=True)
            x = calib[e]
            if n < x.shape[0]:
                x = x[np.sort(np.random.default_rng(seed).choice(x.shape[0], n, replace=False))]
            grams[e] = collect_grams(model, [x], names, dev)["grams"]
        for label, kw in (("plain", {}), ("kcap1e5", {"kappa_target": 1e5}), ("shrink0.95", {"shrink": 0.95})):
            merged, kap = {}, []
            for p in names:
                gs = [grams[e][input_name(p)] for e in TASKS8]
                sol, info = regmean_solve([d.to(dev) for d in deltas[p]], gs, ridge_to="mean", return_info=True, **kw)
                merged[p] = sol.reshape(base[p].shape).float().cpu()
                kap.append(info["kappa_before"])
            sd = {k: base[k] + (merged[k] if k in merged else
                                torch.stack([experts[e][k] - base[k] for e in TASKS8]).mean(0)) for k in base}
            wmax = max(float(sd[p].abs().max()) for p in names)
            res = {"max_abs_weight": wmax, "max_abs_base_weight": max(float(base[p].abs().max()) for p in names),
                   "median_kappa_G": float(np.median(kap)), "max_kappa_G": float(np.max(kap))}
            for tag, ev in (("fp16", ev16), ("fp32", ev32)):
                acc = ev.evaluate(sd, "dev")
                m = summarize(acc, ref)
                ev.load(sd)
                nonfinite = {}
                with torch.no_grad():
                    for t in TASKS8:
                        x, _ = ev.data[(t, "dev")]
                        f = ev.features(x[:256])
                        nonfinite[t] = float((~torch.isfinite(f)).any(-1).float().mean())
                res[tag] = {"mean_norm": m["mean_norm"], "mean_acc": m["mean_acc"], "acc": acc,
                            "frac_nonfinite_features": nonfinite}
            res["fp32_minus_fp16_norm_pt"] = 100 * (res["fp32"]["mean_norm"] - res["fp16"]["mean_norm"])
            report[f"n{n}_s{seed}/{label}"] = res
            print(f"n={n} s={seed} {label:10s} fp16 {100 * res['fp16']['mean_norm']:.2f} fp32 {100 * res['fp32']['mean_norm']:.2f} "
                  f"(norm) wmax {wmax:.3g} nonfinite16 {max(res['fp16']['frac_nonfinite_features'].values()):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "collapse_audit_fullft.json").write_text(json.dumps(report, indent=1, default=float))
        del grams
        torch.cuda.empty_cache()
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
