"""H-INV1 (protocol 02 sec. 8): does conditioning-based adaptation add value over a competently tuned ordinary ridge
in LoRA RegMean? Equal tuning budgets, equal calibration, equal solve and output rank.

Settings (task-separated Grams, budget C, cached by scripts/aa_lora_grams.py):
  KnOTS release N=512; KnOTS N=16 subsets s0 (first 16), s1, s2 (random); lora_c5a N=512; vit20 ("lora") N=512.
Methods (full input-space solve D (sum G~_i + rho I) = sum D_i G~_i, ridge target zero), 7 strengths each:
  kcap   rho = smallest ridge with kappa2(G + rho I) <= kappa_t,  kappa_t in KAPPAS            (kappa-adaptive per map)
  fixed  rho = c tr(G)/d,  c_j = r_med / kappa_t,j  (r_med = median over maps of lambda_max/(tr G/d) of THIS setting's
         Grams), i.e. the same typical strength as kcap at kappa_t,j; only WHERE the ridge goes differs
  shrink RegMean off-diagonal shrinkage a in SHRINKS (a = 1 is plain RegMean)
  x alpha in {0.8, 1.0, 1.3}; plus uniform averaging at the same alphas.
Evaluation in fp32 (amendment 1 of protocol 02; the fp16 test accuracy of each selected config is also reported).
Selection on dev (mean accuracy over the 8 tasks; LoRA results are ABSOLUTE accuracy), frozen, then test with
per-image correctness and a paired, task-stratified bootstrap (2000) of kcap - fixed and kcap - shrink.
Rank-16 variant: every selected method's merged update truncated to rank 16 (alpha re-selected on dev among the
same 3 values), reported separately.  Writes results/inverse/lora_ridge_equal_{dev.jsonl,test.json}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from condmerge.paths import CACHE, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

from aa_lora_regmean import input_name  # noqa: E402
from aa_lora_replicate_banks import load_bank  # noqa: E402
from condmerge.aa_common import TaskData, load_base_unprefixed, new_model  # noqa: E402
from condmerge.regmean_lora import kappa_ridge, regmean_full  # noqa: E402

OUT = ROOT / "results" / "inverse"
KAPPAS = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5]
SHRINKS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
ALPHAS = [0.8, 1.0, 1.3]
SETTINGS = [("knots_release", "C", "KnOTS N512"), ("knots_release", "C_n16", "KnOTS N16 s0"),
            ("knots_release", "C_n16_s1", "KnOTS N16 s1"), ("knots_release", "C_n16_s2", "KnOTS N16 s2"),
            ("lora_c5a", "C", "lora_c5a N512"), ("lora", "C", "vit20 N512")]


def truncate(m: torch.Tensor, r: int) -> torch.Tensor:
    """Best rank-r approximation (float64 SVD on CPU: GPU float64 SVD is slow on these cards)."""
    u, s, vh = torch.linalg.svd(m.double().cpu(), full_matrices=False)
    return ((u[:, :r] * s[:r]) @ vh[:r]).float().to(m.device)


AMP = None  # fp32 evaluation (protocol 02 amendment 1: plain RegMean's fp16 'collapses' are overflow artefacts)


def correctness(model, data: TaskData, params: dict, amp=None) -> dict[str, np.ndarray]:
    out = {}
    with torch.no_grad():
        for t in TASKS8:
            x, y = data.data[(t, "test")]
            preds = [(data.features(model, x[i:i + 1024], params, amp) @ data.heads[t].T).argmax(-1)
                     for i in range(0, x.shape[0], 1024)]
            out[t] = (torch.cat(preds) == y).cpu().numpy()
    return out


def boot(ca: dict, cb: dict, B: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    point = float(np.mean([ca[t].mean() - cb[t].mean() for t in TASKS8]))
    bs = np.zeros(B)
    for t in TASKS8:
        d = ca[t].astype(np.float64) - cb[t].astype(np.float64)
        idx = rng.integers(0, len(d), size=(B, len(d)))
        bs += d[idx].mean(1)
    bs /= len(TASKS8)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff_pt": 100 * point, "lo_pt": 100 * float(lo), "hi_pt": 100 * float(hi)}


def main() -> None:
    import argparse

    global KAPPAS, SHRINKS, ALPHAS, SETTINGS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="plumbing test: one setting, tiny grids, separate output")
    args = ap.parse_args()
    if args.quick:
        KAPPAS, SHRINKS, ALPHAS, SETTINGS = [1e4], [0.95], [1.0], SETTINGS[:1]
        OUT = OUT / "quick_test"
    dev = "cuda"
    t0 = time.time()
    base = load_base_unprefixed()
    base_dev = {k: v.to(dev) for k, v in base.items()}
    data = TaskData(TASKS8, ("dev", "test"), dev)
    model = new_model(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    devlog = (OUT / "lora_ridge_equal_dev.jsonl").open("a")
    report = {}
    for bank_name, gsuf, label in SETTINGS:
        b = load_bank(bank_name)
        names = sorted(b[TASKS8[0]])
        g_all = torch.load(CACHE / "aa" / f"grams_lora_{bank_name}_{gsuf}.pt", weights_only=False)["grams"]
        d_all = {n: torch.stack([b[t][n] for t in TASKS8]).double().to(dev) for n in names}
        ratios = []
        for key, g in g_all.items():
            gs = g.double().to(dev).sum(0)
            ratios.append(float(torch.linalg.eigvalsh(0.5 * (gs + gs.T))[-1]) / (float(torch.trace(gs)) / gs.shape[0]))
        r_med = float(np.median(ratios))
        methods = [("uniform", None)] + [("kcap", k) for k in KAPPAS] + [("fixed", r_med / k) for k in KAPPAS] + \
                  [("shrink", a) for a in SHRINKS]
        rows, merged_cache = [], {}
        for meth, s in methods:
            merged, rhos = {}, []
            for n in names:
                d, g = d_all[n], g_all[input_name(n)].double().to(dev)
                if meth == "uniform":
                    merged[n] = d.mean(0).float()
                    continue
                if meth == "kcap":
                    m, info = regmean_full(d, g, a=1.0, kappa_target=s, ridge_to="zero")
                elif meth == "fixed":
                    rho = s * float(torch.trace(g.sum(0))) / g.shape[-1]
                    m, info = regmean_full(d, g, a=1.0, ridge=rho, ridge_to="zero")
                else:
                    m, info = regmean_full(d, g, a=s)
                merged[n] = m
                rhos.append(info["rho"] / max(info["trace_mean"], 1e-300))
            merged_cache[(meth, s)] = merged
            for a in ALPHAS:
                params = dict(base_dev)
                for n in names:
                    params[n] = base_dev[n] + a * merged[n].to(dev)
                r = data.evaluate(model, "dev", params=params, amp=AMP)
                row = {"setting": label, "bank": bank_name, "grams": gsuf, "method": meth, "strength": s, "alpha": a,
                       "dev_acc": r["avg_acc"], "acc": r["acc"], "median_rel_rho": float(np.median(rhos)) if rhos else None,
                       "r_med": r_med}
                rows.append(row)
                devlog.write(json.dumps(row) + "\n")
                devlog.flush()
            print(f"{label} {meth} {s}: best dev {100 * max(x['dev_acc'] for x in rows if x['method'] == meth and x['strength'] == s):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        # frozen selection per method (ties -> alpha closer to 1, then the weaker regulariser)
        sel = {}
        for meth in ("uniform", "kcap", "fixed", "shrink"):
            cand = [x for x in rows if x["method"] == meth]
            sel[meth] = sorted(cand, key=lambda x: (-round(x["dev_acc"], 10), abs(x["alpha"] - 1.0)))[0]
        rep = {"r_med": r_med, "selected": sel, "test": {}, "rank16": {}}
        corr = {}
        for meth, x in sel.items():
            merged = merged_cache[(meth, x["strength"])]
            params = dict(base_dev)
            for n in names:
                params[n] = base_dev[n] + x["alpha"] * merged[n].to(dev)
            c = correctness(model, data, params, AMP)
            c16 = correctness(model, data, params, "fp16")
            corr[meth] = c
            rep["test"][meth] = {"acc": float(np.mean([c[t].mean() for t in TASKS8])),
                                 "acc_fp16_eval": float(np.mean([c16[t].mean() for t in TASKS8])),
                                 "per_task": {t: float(c[t].mean()) for t in TASKS8}, "strength": x["strength"],
                                 "alpha": x["alpha"], "dev_acc": x["dev_acc"]}
            # rank-16 truncation of the same selected update, alpha re-selected on dev
            best16 = None
            tr = {n: truncate(merged[n], 16) for n in names}
            for a in ALPHAS:
                p16 = dict(base_dev)
                for n in names:
                    p16[n] = base_dev[n] + a * tr[n].to(dev)
                r16 = data.evaluate(model, "dev", params=p16, amp=AMP)["avg_acc"]
                if best16 is None or r16 > best16[0]:
                    best16 = (r16, a)
            p16 = dict(base_dev)
            for n in names:
                p16[n] = base_dev[n] + best16[1] * tr[n].to(dev)
            c16r = correctness(model, data, p16, AMP)
            corr[meth + "_r16"] = c16r
            rep["rank16"][meth] = {"acc": float(np.mean([c16r[t].mean() for t in TASKS8])), "alpha": best16[1],
                                   "dev_acc": best16[0]}
        rep["kcap_minus_fixed"] = boot(corr["kcap"], corr["fixed"])
        rep["kcap_minus_shrink"] = boot(corr["kcap"], corr["shrink"])
        rep["fixed_minus_shrink"] = boot(corr["fixed"], corr["shrink"])
        rep["kcap_minus_fixed_r16"] = boot(corr["kcap_r16"], corr["fixed_r16"])
        report[label] = rep
        print(f"== {label}: test kcap {100 * rep['test']['kcap']['acc']:.2f} fixed {100 * rep['test']['fixed']['acc']:.2f} "
              f"shrink {100 * rep['test']['shrink']['acc']:.2f} uniform {100 * rep['test']['uniform']['acc']:.2f}; "
              f"kcap-fixed {rep['kcap_minus_fixed']['diff_pt']:+.2f} [{rep['kcap_minus_fixed']['lo_pt']:+.2f}, "
              f"{rep['kcap_minus_fixed']['hi_pt']:+.2f}]", flush=True)
        (OUT / "lora_ridge_equal_test.json").write_text(json.dumps(report, indent=1, default=float))
        del g_all, d_all, merged_cache
        torch.cuda.empty_cache()
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
