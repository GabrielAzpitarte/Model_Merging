"""Audit of the ~4.1 % collapses of plain LoRA RegMean (protocol 02 sec. 8, H-INV0): numerical failure or
statistical overfitting?

For each collapsed configuration (task-separated Grams, full input-space solve, no regulariser) and, for contrast,
the same Grams with a kappa-capped ridge:
  Grams      symmetry, eigenvalue range, kappa, numerical rank, PSD violations (per attention input)
  solves     LU (torch.linalg.solve, the original), Cholesky, explicit inverse, pseudo-inverse, LU in fp32;
             relative residual ||D G - rhs|| / ||rhs||, ||D||_F, max |D|, relative difference to the LU solution
  weights    finiteness and max |W0 + alpha D| vs max |W0|
  forward    fp16 autocast (the pipeline's evaluation precision) vs fp32 on dev: accuracy, fraction of
             non-finite pooled features, max |hidden state| per encoder layer (hooks, 256 images/task),
             share of the most frequent predicted class per task
Verdict per configuration: NUMERICAL if fp32 evaluation of the same weights recovers clearly (non-finite values
in fp16), or if solvers disagree / fail their residuals; STATISTICAL if every check is finite and accurate and fp32
also fails.  Writes results/inverse/collapse_audit.json.
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
from condmerge.regmean_lora import kappa_ridge  # noqa: E402

OUT = ROOT / "results" / "inverse"
CASES = [  # (bank, gram file suffix, alphas, label)
    ("knots_release", "C", (0.8, 1.0, 1.3), "KnOTS N=512"),
    ("knots_release", "C_n16", (0.8, 1.0, 1.3), "KnOTS N=16 (first 16)"),
    ("lora_c5a", "C", (0.8, 1.0, 1.3), "lora_c5a N=512"),
]


def solves(d: torch.Tensor, g: torch.Tensor, rho: float = 0.0) -> dict:
    """d (K, out, in), g (K, in, in) float64 -> {method: solution} for D (sum G + rho I) = sum D_i G_i."""
    gs = g.sum(0) + rho * torch.eye(g.shape[-1], dtype=g.dtype, device=g.device)
    rhs = torch.einsum("kod,kde->oe", d, g)
    out = {"lu": torch.linalg.solve(gs, rhs.T).T}
    try:
        ch = torch.linalg.cholesky(0.5 * (gs + gs.T))
        out["cholesky"] = torch.cholesky_solve(rhs.T, ch).T
    except RuntimeError:
        out["cholesky"] = None
    out["inverse"] = rhs @ torch.linalg.inv(gs)
    out["pinv"] = rhs @ torch.linalg.pinv(gs, hermitian=True)
    out["lu_fp32"] = torch.linalg.solve(gs.float(), rhs.T.float()).T.double()
    return out, gs, rhs


def main() -> None:
    dev = "cuda"
    t0 = time.time()
    base = load_base_unprefixed()
    base_dev = {k: v.to(dev) for k, v in base.items()}
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    report = {}
    for bank_name, gsuf, alphas, label in CASES:
        b = load_bank(bank_name)
        names = sorted(b[TASKS8[0]])
        g_all = torch.load(CACHE / "aa" / f"grams_lora_{bank_name}_{gsuf}.pt", weights_only=False)["grams"]
        rep = {"label": label, "grams": {}, "solves": {}, "configs": {}}
        for key, g in sorted(g_all.items()):
            g = g.double().to(dev)
            gs = g.sum(0)
            ev = torch.linalg.eigvalsh(0.5 * (gs + gs.T))
            rep["grams"][key] = {"rel_asymmetry": float((gs - gs.T).abs().max() / gs.abs().max()),
                                 "eig_min": float(ev[0]), "eig_max": float(ev[-1]),
                                 "kappa": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
                                 "n_eig_below_1e-12max": int((ev < 1e-12 * ev[-1]).sum()),
                                 "psd_violation": bool(ev[0] < -1e-10 * ev[-1]),
                                 "per_expert_min_eig_over_max": [float(torch.linalg.eigvalsh(0.5 * (x + x.T))[0] /
                                                                      torch.linalg.eigvalsh(0.5 * (x + x.T))[-1]) for x in g]}
        for reg in ("plain", "kcap1e4"):
            merged = {}
            sol_stats = {}
            for n in names:
                d = torch.stack([b[t][n] for t in TASKS8]).double().to(dev)
                g = g_all[input_name(n)].double().to(dev)
                rho = 0.0 if reg == "plain" else kappa_ridge(g.sum(0), 1e4)
                sols, gs, rhs = solves(d, g, rho)
                ref = sols["lu"]
                st = {}
                for m, x in sols.items():
                    if x is None:
                        st[m] = None
                        continue
                    st[m] = {"rel_residual": float((x @ gs - rhs).norm() / rhs.norm()), "fro": float(x.norm()),
                             "max_abs": float(x.abs().max()), "rel_diff_vs_lu": float((x - ref).norm() / ref.norm()),
                             "finite": bool(torch.isfinite(x).all())}
                st["mean_update_fro"] = float(d.mean(0).norm())
                sol_stats[n] = st
                merged[n] = ref.float()
            rep["solves"][reg] = sol_stats
            for a in alphas:
                params = dict(base_dev)
                for n in names:
                    params[n] = base_dev[n] + a * merged[n]
                wmax = max(float(params[n].abs().max()) for n in names)
                w0max = max(float(base_dev[n].abs().max()) for n in names)
                res = {"weights_finite": all(bool(torch.isfinite(params[n]).all()) for n in names),
                       "max_abs_weight": wmax, "max_abs_base_weight": w0max}
                for amp in ("fp16", None):
                    ev_ = data.evaluate(model, "dev", params=params, amp=amp)
                    res[f"acc_{amp or 'fp32'}"] = ev_["avg_acc"]
                    res[f"acc_per_task_{amp or 'fp32'}"] = ev_["acc"]
                # forward diagnostics on 256 dev images per task
                stats = {}
                for amp in ("fp16", None):
                    hmax = {}
                    nonfinite, top_share = {}, {}
                    hooks = []
                    from torch.func import functional_call

                    enc = model.encoder if hasattr(model, "encoder") else model.vision_model.encoder
                    for li, layer in enumerate(enc.layers):
                        def hk(_m, _i, o, li=li):
                            h = o[0] if isinstance(o, tuple) else o
                            hmax[li] = max(hmax.get(li, 0.0), float(h.float().abs().nan_to_num(posinf=1e30).max()))
                        hooks.append(layer.register_forward_hook(hk))
                    with torch.no_grad():
                        for t in TASKS8:
                            x, y = data.data[(t, "dev")]
                            f = data.features(model, x[:256], params, amp)
                            nonfinite[t] = float((~torch.isfinite(f)).any(-1).float().mean())
                            pred = (f @ data.heads[t].T).argmax(-1)
                            top_share[t] = float(torch.bincount(pred, minlength=data.heads[t].shape[0]).max() / len(pred))
                    for h in hooks:
                        h.remove()
                    stats[amp or "fp32"] = {"max_abs_hidden_per_layer": [hmax[i] for i in sorted(hmax)],
                                            "frac_nonfinite_features": nonfinite, "top_class_share": top_share}
                res["forward"] = stats
                fp16_bad = any(v > 0 for v in stats["fp16"]["frac_nonfinite_features"].values())
                solver_bad = any(s["lu"]["rel_residual"] > 1e-6 for s in sol_stats.values() if isinstance(s, dict))
                recovered = res["acc_fp32"] - res["acc_fp16"] > 0.05
                res["verdict"] = ("NUMERICAL (fp16 non-finite, recovered in fp32)" if fp16_bad and recovered else
                                  "NUMERICAL (fp16 non-finite; fp32 also fails)" if fp16_bad else
                                  "NUMERICAL (solver residual)" if solver_bad else
                                  "STATISTICAL (finite, accurate solve; fails in fp32 too)" if res["acc_fp32"] < 0.2 else
                                  "no collapse")
                rep["configs"][f"{reg}@alpha{a}"] = res
                print(f"{label} {reg} a={a}: fp16 {100 * res['acc_fp16']:.2f} fp32 {100 * res['acc_fp32']:.2f} "
                      f"wmax {wmax:.3g} (base {w0max:.3g}) -> {res['verdict']} ({time.time() - t0:.0f}s)", flush=True)
        report[f"{bank_name}/{gsuf}"] = rep
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "collapse_audit.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
