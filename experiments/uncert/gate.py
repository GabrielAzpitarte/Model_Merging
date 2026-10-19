"""Uncertainty-gated merge correction (protocol 02 sec. 9, second stage). RUN ONLY IF the pre-registered diagnostic
passed (results/uncert/diagnostic_summary.json: diagnostic_passed = true); otherwise this script refuses to run.

CLIP ViT-B/32 full FT, budgets C (task-separated) and B (pooled), n = 16 images per task, the same 8 calibration
subsets as the diagnostic (upool). Per map p, with the subset's own statistics only:
    plain    D = R + F_hat,                    F_hat = B G^-1
    gate     D = R + gamma_p F_hat,            gamma_p = 1 / (1 + c * u_rel_p)        (u_rel: bootstrap relative
             variance of F from the diagnostic records of the same subset), ONE global c in C_GRID
    gate_shuf  the same gate with u_rel permuted across maps (null: does the uncertainty's allocation matter?)
    ridge    D = R + B (G + rho I)^-1,          rho = c_rel tr(G)/d,  c_rel in RIDGE_GRID      (tuned fixed ridge)
    kcap     D = R + B (G + rho I)^-1,          rho = smallest with kappa(G + rho I) <= kappa_t, kappa_t in KCAP_GRID
    shrink   G~_i = a G_i + (1 - a) diag G_i,   a in SHRINK_GRID (RegMean)
    uniform  D = R
alpha = 1 everywhere (Report 2 convention); non-linear parameters averaged. Each method's knob is selected on dev
(mean normalised accuracy, averaged over the 8 subsets), then frozen and evaluated on test per subset.
Writes results/uncert/gate_{dev.jsonl, test.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condmerge.paths import BASE_MODEL, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False

from condmerge.clip import Evaluator, load_base_state, load_experts, summarize  # noqa: E402
from condmerge.data import load_cache  # noqa: E402
from condmerge.grams import input_name  # noqa: E402
from condmerge.spectra import as_matrix, linear_map_names  # noqa: E402
from diagnostic import Acts, gram  # noqa: E402

OUT = ROOT / "results" / "uncert"
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
RIDGE_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
KCAP_GRID = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
SHRINK_GRID = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]


def solve(B: torch.Tensor, G: torch.Tensor, rho: float = 0.0) -> torch.Tensor:
    Gs = 0.5 * (G + G.T)
    if rho > 0:
        Gs = Gs + rho * torch.eye(Gs.shape[0], dtype=Gs.dtype, device=Gs.device)
    try:
        return torch.cholesky_solve(B.T, torch.linalg.cholesky(Gs)).T
    except RuntimeError:
        return torch.linalg.lstsq(Gs, B.T).solution.T


def main() -> None:
    from transformers import CLIPVisionConfig, CLIPVisionModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C:16,B:16")
    ap.add_argument("--force", action="store_true", help="run even if the diagnostic did not pass (exploratory)")
    args = ap.parse_args()
    summ = json.loads((OUT / "diagnostic_summary.json").read_text())
    if not summ.get("diagnostic_passed") and not args.force:
        print("diagnostic did not pass the pre-registered criterion: gate stage not run", flush=True)
        (OUT / "gate_test.json").write_text(json.dumps({"status": "not run: diagnostic failed its criterion"}))
        return
    dev = "cuda:0"
    t0 = time.time()
    recs = [json.loads(l) for l in (OUT / "diagnostic_records.jsonl").read_text().splitlines() if l.strip()]
    base = load_base_state()
    experts = load_experts(TASKS8)
    names = linear_map_names(base)
    K = len(TASKS8)
    D = {p: torch.stack([as_matrix(p, experts[e][p]).double() - as_matrix(p, base[p]).double() for e in TASKS8]).float().to(dev)
         for p in names}
    other = {k: torch.stack([experts[e][k] - base[k] for e in TASKS8]).mean(0) for k in base if k not in D}
    ev = {sp: Evaluator(dev, splits=(sp,), tasks=TASKS8) for sp in ("dev", "test")}
    ref = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"]
    model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(BASE_MODEL)).eval().to(dev)
    acts = Acts(model, names, dev)
    pool = {t: load_cache(t, "upool", mmap=True)[0] for t in TASKS8}
    log = (OUT / "gate_records.jsonl").open("a")
    results = {}
    methods = [("uniform", None), ("plain", None)] + [("gate", c) for c in C_GRID] + [("gate_shuf", c) for c in C_GRID] + \
              [("ridge", c) for c in RIDGE_GRID] + [("kcap", k) for k in KCAP_GRID] + [("shrink", a) for a in SHRINK_GRID]

    def merged_update(meth, knob, st, Dp, u_p):
        R = st["R"]
        if meth == "uniform":
            return R
        if meth == "plain":
            return R + solve(st["B"], st["G"])
        if meth in ("gate", "gate_shuf"):  # gate_shuf: the same gate with u permuted across maps (null)
            return R + solve(st["B"], st["G"]) / (1.0 + knob * u_p)
        if meth == "ridge":
            return R + solve(st["B"], st["G"], knob * st["trd"])
        if meth == "kcap":
            return R + solve(st["B"], st["G"], max(0.0, (st["lmax"] - knob * st["lmin"]) / (knob - 1.0)))
        Gt = knob * st["Gs"] + (1 - knob) * torch.diag_embed(st["dvec"])
        Bt = torch.einsum("kod,kde->oe", Dp - R, Gt)
        return R + solve(Bt, Gt.sum(0))

    for cfg in args.configs.split(","):
        budget, n = cfg.split(":")[0], int(cfg.split(":")[1])
        subs = sorted({r["subset"] for r in recs if r["budget"] == budget and r["n"] == n})
        rows = []
        for s in subs:
            u = {r["map"]: r["u_rel"] for r in recs if r["budget"] == budget and r["n"] == n and r["subset"] == s}
            perm = np.random.default_rng(1000 + s).permutation(len(names))
            u_shuf = {p: u[names[perm[j]]] for j, p in enumerate(names)}
            idx = np.arange(s * n, (s + 1) * n)
            A = {}
            for e in TASKS8:
                model.load_state_dict({k: v.to(dev) for k, v in experts[e].items()}, strict=True)
                imgs = pool[e][idx] if budget == "C" else np.concatenate([pool[t][idx] for t in TASKS8])
                A[e] = acts(imgs)
            by_input = {}
            for p in names:
                by_input.setdefault(input_name(p), []).append(p)
            stats = {}
            for k, ps in by_input.items():
                Gs = torch.stack([gram(A[e][k]) for e in TASKS8])
                G = Gs.sum(0)
                ev_ = torch.linalg.eigvalsh(0.5 * (G + G.T))
                dvec = torch.diagonal(Gs, dim1=-2, dim2=-1).clone()
                for p in ps:
                    Dp = D[p].double()
                    R = Dp.mean(0)
                    stats[p] = {"R": R, "B": torch.einsum("kod,kde->oe", Dp - R, Gs), "G": G, "lmin": float(ev_[0]),
                                "lmax": float(ev_[-1]), "trd": float(torch.trace(G)) / G.shape[0], "Gs": Gs, "dvec": dvec}
            del A
            for meth, knob in methods:
                sd = dict(base)
                for p in names:
                    upd = merged_update(meth, knob, stats[p], D[p].double(), u_shuf[p] if meth == "gate_shuf" else u[p])
                    sd[p] = base[p] + upd.reshape(base[p].shape).float().cpu()
                for k2, v in other.items():
                    sd[k2] = base[k2] + v
                row = {"budget": budget, "n": n, "subset": s, "method": meth, "knob": knob}
                for sp in ("dev", "test"):
                    m = summarize(ev[sp].evaluate(sd, sp), ref[sp])
                    row[f"{sp}_norm"], row[f"{sp}_acc"] = m["mean_norm"], m["mean_acc"]
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
            del stats
            torch.cuda.empty_cache()
            print(f"[{budget} n={n} s={s}] done ({time.time() - t0:.0f}s)", flush=True)
        # selection on dev ONLY (mean over subsets), then the frozen knob's test per subset
        sel, test = {}, {}
        for meth in ("uniform", "plain", "gate", "gate_shuf", "ridge", "kcap", "shrink"):
            knobs = sorted({r["knob"] for r in rows if r["method"] == meth}, key=lambda x: -1 if x is None else x)
            best = max(knobs, key=lambda kb: np.mean([r["dev_norm"] for r in rows if r["method"] == meth and r["knob"] == kb]))
            sel[meth] = best
            test[meth] = [next(r["test_norm"] for r in rows if r["method"] == meth and r["knob"] == best and r["subset"] == s)
                          for s in subs]
        results[cfg] = {"selected_knob_on_dev": sel, "test_norm_per_subset": test,
                        "test_norm_mean": {m: float(np.mean(v)) for m, v in test.items()},
                        "gate_minus_ridge_per_subset": [a - b for a, b in zip(test["gate"], test["ridge"])],
                        "gate_minus_kcap_per_subset": [a - b for a, b in zip(test["gate"], test["kcap"])],
                        "gate_minus_shuffled_gate_per_subset": [a - b for a, b in zip(test["gate"], test["gate_shuf"])],
                        "note": "test numbers of every knob are logged in gate_records.jsonl; only the dev-selected knob is reported"}
        (OUT / "gate_test.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
