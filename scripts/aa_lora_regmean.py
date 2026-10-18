"""RegMean-type merging on LoRA banks: "does the instability lie in the solve?" (review hypothesis table, row 3).

For every adapted attention map: D = (sum_i D_i G~_i)(sum_i G~_i + rho I)^-1  (full input space, dense output)
or the exact adapter-subspace solve (output rank <= K r = 128); see src/condmerge/regmean_lora.py.
Gram budgets (artifacts/cache/aa/grams_lora_<bank>_<budget>.pt, from scripts/aa_lora_grams.py):
  C  task-separated (expert i on task i calib)   Bp pooled through each expert   B0 pooled through the base
Knobs: RegMean off-diagonal shrinkage a in {1, 0.95, 0.9, 0.5}; kappa-guided ridge (kappa_target in {1e2, 1e3, 1e4},
a = 1, ridge toward zero or toward the uniform average, as master's regmean_solve); alpha in {0.8, 1.0, 1.3}. Baselines: uniform average at the same alpha grid, best single expert (dev).
Dev only; linear algebra in float64 on CPU. Writes results/aa/lora/regmean_<bank>_dev.jsonl and
regmean_<bank>_maps.json (per-map kappa of the solve).
"""

from __future__ import annotations

import argparse
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

from aa_lora_replicate_banks import load_bank  # noqa: E402
from condmerge.aa_common import TaskData, load_base_unprefixed, new_model  # noqa: E402
from condmerge.regmean_lora import regmean_full, regmean_subspace  # noqa: E402


def input_name(param: str) -> str:
    layer = param.split(".")[2]
    return f"L{layer}.out_in" if "out_proj" in param else f"L{layer}.attn_in"


def rowspace_basis(deltas: torch.Tensor, rank: int = 16, tol: float = 1e-10) -> torch.Tensor:
    """Orthonormal basis of the union of the experts' input row spaces (exact for rank-r updates)."""
    vs = []
    for d in deltas.double():
        _, s, vh = torch.linalg.svd(d, full_matrices=False)
        vs.append(vh[: int((s > tol * s[0]).sum().clamp(max=rank))])
    u, s, _ = torch.linalg.svd(torch.cat(vs, 0).T, full_matrices=False)
    return u[:, : int((s > tol * s[0]).sum())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="knots_release")
    ap.add_argument("--budgets", default="C,Bp,B0")
    ap.add_argument("--alphas", default="0.8,1.0,1.3")
    ap.add_argument("--gram-suffix", default="", help="e.g. _n16 for Grams from 16 calib images per task")
    ap.add_argument("--solves", default="full,sub")
    ap.add_argument("--compress", type=int, default=0, help="truncate every merged update to this rank (0 = exact)")
    ap.add_argument("--quick", action="store_true", help="only plain, shrink 0.95, kappa 1e4 (mean/zero) configs")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    dev = "cuda"
    t0 = time.time()
    base = load_base_unprefixed()
    bank = load_bank(args.bank)
    names = sorted(bank[TASKS8[0]])
    deltas = {n: torch.stack([bank[t][n] for t in TASKS8]).double() for n in names}
    bases = {n: rowspace_basis(deltas[n]) for n in names}
    grams = {b: torch.load(CACHE / "aa" / f"grams_lora_{args.bank}_{b}{args.gram_suffix}.pt", weights_only=False)
             for b in args.budgets.split(",")}
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    base_dev = {k: v.to(dev) for k, v in base.items()}
    csfx = f"_r{args.compress}" if args.compress else ""
    out = (RESULTS / "aa" / "lora" / f"regmean_{args.bank}{args.gram_suffix}{csfx}_dev.jsonl").open("a")
    maps_diag = {}

    def compress(m: torch.Tensor) -> torch.Tensor:
        if not args.compress:
            return m
        u, s_, vh = torch.linalg.svd(m.double(), full_matrices=False)
        k = args.compress
        return ((u[:, :k] * s_[:k]) @ vh[:k]).float()

    def evaluate(merged: dict[str, torch.Tensor], a: float) -> dict:
        params = dict(base_dev)
        for n in names:
            params[n] = base_dev[n] + a * compress(merged[n]).to(dev)
        return data.evaluate(model, "dev", params=params)

    def record(cfg: dict, merged: dict, extra: dict):
        for a in alphas:
            r = evaluate(merged, a)
            rec = {**cfg, "alpha": a, "avg_acc": r["avg_acc"], "acc": r["acc"], **extra}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{json.dumps(cfg)} a={a}: {r['avg_acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    record({"method": "uniform", "bank": args.bank}, {n: deltas[n].mean(0).float() for n in names}, {})
    for budget, blob in grams.items():
        g_by_input = blob["grams"]
        for solve in args.solves.split(","):
            configs = [(a, None, "zero") for a in (1.0, 0.95, 0.9, 0.5)] + \
                      [(1.0, kt, rt) for kt in (1e2, 1e3, 1e4) for rt in ("zero", "mean")]
            if args.quick:
                configs = [(1.0, None, "zero"), (0.95, None, "zero"), (1.0, 1e4, "zero"), (1.0, 1e4, "mean")]
            for shrink_a, kt, rt in configs:
                merged, kap, kap_s, rhos = {}, [], [], []
                for n in names:
                    g = g_by_input[input_name(n)].double()
                    if g.shape[0] == 1:
                        g = g.repeat(len(TASKS8), 1, 1)
                    if solve == "full":
                        m, info = regmean_full(deltas[n], g, a=shrink_a, kappa_target=kt, ridge_to=rt)
                    else:
                        m, info = regmean_subspace(deltas[n], g, bases[n], a=shrink_a, kappa_target=kt, ridge_to=rt)
                    merged[n] = m
                    kap.append(info["kappa_G"])
                    kap_s.append(info["kappa_solve"])
                    rhos.append(info["rho"])
                    if shrink_a == 1.0 and kt is None:
                        maps_diag.setdefault(f"{budget}/{solve}", {})[n] = info
                cfg = {"method": "regmean", "bank": args.bank, "grams": args.gram_suffix or "_n512", "budget": budget, "solve": solve, "shrink": shrink_a,
                       "kappa_target": kt, "ridge_to": rt if kt is not None else None}
                record(cfg, merged, {"median_kappa_G": float(np.median(kap)), "max_kappa_G": float(np.max(kap)),
                                     "median_kappa_solve": float(np.median(kap_s)), "median_rho": float(np.median(rhos))})
    (RESULTS / "aa" / "lora" / f"regmean_{args.bank}{args.gram_suffix}{csfx}_maps.json").write_text(json.dumps(maps_diag, indent=1, default=float))
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
