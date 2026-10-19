"""Validation and diagnostics of the LLM spectra (CPU). Protocol 02 sec. 5.

A  validation on representative matrices (layers 0, 1, mid, last x q/k/v/o; base + 3 experts):
   gesdd (cache) vs gesvd (scipy, float64); smallest singular triplet residuals ||W v - s u|| / s_1 and
   ||W^T u - s v|| / s_1; an independent sigma_min by inverse iteration on the triangular QR factor (no Gram
   matrix); fp32 singular values (CPU LAPACK) of the fp32-assembled matrix; the Gram route
   sqrt(eig(W W^T)) (squares kappa); fp32 vs fp64 assembly of W_i.
B  source-precision sensitivity, ALL matrices, the T8 experts: R re-rounding replicates W0 + E, E_jk ~ U(+-ulp_jk/2)
   (an equally valid preimage of the stored BF16 checkpoint); per matrix: Kendall tau of the expert ordering by kappa,
   relative kappa change, lambda change at the kappa dev-grid b with mean n_eff ~ 0.45 K.
C  diagnostics from the cache, ALL matrices: kappa0; across-expert spread; variance share of log sigma_max vs
   log sigma_min in log kappa; counts below tau_bf16 and below the computational bound; interlacing check;
   within-matrix Spearman(kappa, ||Delta||_F); argmin-expert frequencies; Kendall's W of orderings.
Outputs: results/llm/spectral/<bank>_{validation,rerounding,diagnostics}.json (+ per-matrix CSV).
Usage: spectral_checks.py --bank l31_8b --T 8 [--parts A,B,C] [--replicates 2] [--procs 6]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "llm" / "spectral"


def _init(threads: int) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    import torch

    torch.set_num_threads(threads)


def _ulp_bf16(w0b):
    import numpy as np

    x = w0b.float().abs().numpy().astype(np.float64)
    e = np.full_like(x, 2.0 ** -133)
    nz = x > 0
    e[nz] = np.exp2(np.floor(np.log2(x[nz])) - 7)
    return e


def _inverse_iteration_smin(w, iters=300, tol=1e-13):
    """sigma_min of w (m x n, float64 torch) from R of a QR factorisation and power iteration on (R^T R)^-1
    via two triangular solves per step (the Gram matrix is never formed)."""
    import torch

    m, n = w.shape
    a = w if m >= n else w.T
    _, r = torch.linalg.qr(a)  # a = q r, r (q x q) upper triangular, sigma(a) = sigma(r)
    x = torch.randn(r.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    x /= x.norm()
    prev = None
    for it in range(iters):
        y = torch.linalg.solve_triangular(r.T, x[:, None], upper=False)
        z = torch.linalg.solve_triangular(r, y, upper=True)[:, 0]
        x = z / z.norm()
        s = float((r @ x).norm())
        if prev is not None and abs(s - prev) <= tol * max(s, 1e-300):
            break
        prev = s
    return s, it + 1


def part_a(bank: str, T: int) -> dict:
    import numpy as np
    import scipy.linalg as sl
    import torch

    from condmerge.llm import banks as BK
    from condmerge.llm import rules

    n = BK.BANKS[bank].n_layers
    layers = [0, 1, n // 2, n - 1]
    tasks = BK.tasks(bank, T)[:3]
    sp = rules.spectra(bank)
    rows = []
    for li in layers:
        for p in BK.PROJ:
            m = f"model.layers.{li}.self_attn.{p}"
            w0b = BK.base_weight(bank, m)
            w0 = w0b.double()
            vers = sp[m]["versions"]
            for name in ["base"] + tasks:
                t0 = time.time()
                if name == "base":
                    w, w32 = w0, w0b.float()
                else:
                    a, b = BK.load_lora_module(bank, name, m)
                    w = w0 + torch.from_numpy(b).double() @ torch.from_numpy(a).double()
                    w32 = w0b.float() + torch.from_numpy(b) @ torch.from_numpy(a)
                s_dd = sp[m]["s"][vers.index(name)]
                s_vd = sl.svd(w.numpy(), compute_uv=False, lapack_driver="gesvd")
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                uq, vq, sq = u[:, -1], vh[-1], s[-1]
                res1 = float((w @ vq - sq * uq).norm() / s[0])
                res2 = float((w.T @ uq - sq * vq).norm() / s[0])
                s_ii, n_it = _inverse_iteration_smin(w)
                s32 = torch.linalg.svdvals(w32).double().numpy()  # fp32 LAPACK on CPU (no GPU use)
                g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
                ev = torch.linalg.eigvalsh(g).numpy()
                s_gram_min = float(np.sqrt(max(ev[0], 0.0)))
                s_asm32 = torch.linalg.svdvals(w32.double()).numpy()
                tau = sp[m]["tau"]
                rows.append({
                    "module": m, "version": name, "q": int(len(s_dd)), "s1": float(s_dd[0]), "smin_gesdd": float(s_dd[-1]),
                    "kappa_gesdd": float(s_dd[0] / s_dd[-1]),
                    "rel_diff_smin_gesvd": float(abs(s_vd[-1] - s_dd[-1]) / s_dd[-1]),
                    "rel_diff_smin_inverse_iteration": float(abs(s_ii - s_dd[-1]) / s_dd[-1]), "inv_iter_steps": n_it,
                    "residual_Wv_minus_su_over_s1": res1, "residual_WTu_minus_sv_over_s1": res2,
                    "rel_err_smin_fp32_svd": float(abs(s32[-1] - s_dd[-1]) / s_dd[-1]),
                    "rel_err_smin_gram_route": float(abs(s_gram_min - s_dd[-1]) / s_dd[-1]),
                    "rel_diff_kappa_fp32_assembly": float(abs(s_asm32[0] / s_asm32[-1] - s_dd[0] / s_dd[-1]) / (s_dd[0] / s_dd[-1])),
                    "tau_bf16": tau, "smin_over_tau": float(s_dd[-1] / tau),
                    "abs_err_bound": float(sp[m]["err"][vers.index(name)]), "seconds": time.time() - t0})
                print(f"A {m} {name}: kappa {rows[-1]['kappa_gesdd']:.4g} gesvd {rows[-1]['rel_diff_smin_gesvd']:.1e} "
                      f"invit {rows[-1]['rel_diff_smin_inverse_iteration']:.1e} res {res1:.1e} fp32 "
                      f"{rows[-1]['rel_err_smin_fp32_svd']:.1e} gram {rows[-1]['rel_err_smin_gram_route']:.1e} "
                      f"asm32 {rows[-1]['rel_diff_kappa_fp32_assembly']:.1e} smin/tau {rows[-1]['smin_over_tau']:.3g}",
                      flush=True)
    # ordering of the 3 experts per matrix under each computation
    return {"rows": rows}


def _rerounding_one(job):
    bank, module, tasks, seed = job
    import numpy as np
    import torch

    from condmerge.llm import banks as BK

    w0b = BK.base_weight(bank, module)
    ulp = _ulp_bf16(w0b)
    import zlib

    rng = np.random.default_rng(seed * 100003 + zlib.crc32(module.encode()))
    e = torch.from_numpy((rng.random(ulp.shape) - 0.5) * ulp)
    w0 = w0b.double() + e
    out = [torch.linalg.svdvals(w0).numpy()[[0, -1]]]
    for t in tasks:
        a, b = BK.load_lora_module(bank, t, module)
        s = torch.linalg.svdvals(w0 + torch.from_numpy(b).double() @ torch.from_numpy(a).double()).numpy()
        out.append(s[[0, -1]])
    return module, seed, np.asarray(out)


def part_b(bank: str, T: int, replicates: int, procs: int, threads: int) -> dict:
    import multiprocessing as mp

    import numpy as np
    from scipy.stats import kendalltau

    from condmerge.llm import banks as BK
    from condmerge.llm import rules

    tasks = BK.tasks(bank, T)
    mods = BK.module_names(bank)
    jobs = [(bank, m, tasks, s) for s in range(replicates) for m in mods]
    res = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(threads,)) as pool:
        for i, (m, seed, arr) in enumerate(pool.imap_unordered(_rerounding_one, jobs)):
            res[(m, seed)] = arr
            if (i + 1) % 16 == 0:
                print(f"B [{i + 1}/{len(jobs)}] ({time.time() - t0:.0f}s)", flush=True)
    orig = rules.kappa_table(bank, T, "raw")
    grid_path = ROOT / "configs" / "llm"
    b_ref = None
    for f in sorted(grid_path.glob("devgrid_*.json")):
        g = json.loads(f.read_text())
        if g["bank"] == bank and g["T"] == T:
            diag = g["rules"]["kappa"]["diagnostics"]
            b_ref = min((d for d in diag if d["b"] != "inf"), key=lambda d: abs(d["mean_n_eff"] - 0.45 * g["K"]))["b"]
    if b_ref is None:
        b_ref = rules.b_for_targets(orig, [0.45 * len(tasks)])[0]["b"]
    sc = rules.scale_of(orig)
    lam0, _ = rules.coefficients(orig, b_ref, scale=sc)
    per = []
    for seed in range(replicates):
        rep = np.stack([res[(m, seed)][1:, 0] / res[(m, seed)][1:, 1] for m in mods])
        lam1, _ = rules.coefficients(rep, b_ref, scale=sc)
        for j, m in enumerate(mods):
            kt = kendalltau(orig[j], rep[j]).statistic
            per.append({"module": m, "kind": BK.kind_of(m), "seed": seed, "kendall_tau": float(kt),
                        "max_rel_kappa_change": float(np.max(np.abs(rep[j] - orig[j]) / orig[j])),
                        "argmin_same": bool(np.argmin(rep[j]) == np.argmin(orig[j])),
                        "max_abs_lambda_change": float(np.max(np.abs(lam1[j] - lam0[j]))),
                        "base_kappa_rel_change": float(abs(res[(m, seed)][0, 0] / res[(m, seed)][0, 1] /
                                                          (rules.spectra(bank)[m]["s"][0, 0] / rules.spectra(bank)[m]["s"][0, -1]) - 1))})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{bank}_T{T}_rerounding.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    summ = {}
    for kind in BK.PROJ:
        sel = [r for r in per if r["kind"] == kind]
        summ[kind] = {"median_kendall_tau": float(np.median([r["kendall_tau"] for r in sel])),
                      "frac_argmin_same": float(np.mean([r["argmin_same"] for r in sel])),
                      "median_max_rel_kappa_change": float(np.median([r["max_rel_kappa_change"] for r in sel])),
                      "median_max_abs_lambda_change": float(np.median([r["max_abs_lambda_change"] for r in sel]))}
    return {"b_ref": b_ref, "replicates": replicates, "by_kind": summ}


def part_c(bank: str, T: int) -> dict:
    import numpy as np
    from scipy.stats import spearmanr

    from condmerge.llm import banks as BK
    from condmerge.llm import rules

    sp = rules.spectra(bank)
    tasks = BK.tasks(bank, T)
    rows_idx = [sp[BK.module_names(bank)[0]]["versions"].index(t) for t in tasks]
    per = []
    argmins = []
    for m in BK.module_names(bank):
        z = sp[m]
        s, tau = z["s"], z["tau"]
        se = s[rows_idx]
        k = se[:, 0] / se[:, -1]
        l1, lq = np.log(se[:, 0]), np.log(se[:, -1])
        v1, vq, cov = np.var(l1), np.var(lq), np.cov(l1, lq, bias=True)[0, 1]
        vk = np.var(np.log(k))
        r = 16
        s0 = s[0]
        q = len(s0)
        viol = 0
        for i in range(len(tasks)):
            sj = se[i]
            lo = np.concatenate([s0[r:], np.zeros(r)])
            hi = np.concatenate([np.full(r, np.inf), s0[:-r]])
            viol += int(np.sum(sj < lo * (1 - 1e-12) - 1e-12) + np.sum(sj > hi * (1 + 1e-12) + 1e-12))
        dn = z["dnorm_f"][rows_idx]
        rho = spearmanr(k, dn).statistic if np.std(k) > 0 else float("nan")
        argmins.append(int(np.argmin(k)))
        per.append({"module": m, "kind": BK.kind_of(m), "layer": BK.layer_of(m), "q": q, "kappa0": float(s0[0] / s0[-1]),
                    "kappa_min": float(k.min()), "kappa_max": float(k.max()), "rel_spread": float((k.max() - k.min()) / (s0[0] / s0[-1])),
                    "cv_kappa": float(np.std(k) / np.mean(k)), "var_log_kappa": float(vk),
                    "share_var_log_s1": float(v1 / vk) if vk > 0 else float("nan"),
                    "share_var_log_smin": float(vq / vk) if vk > 0 else float("nan"),
                    "cov_term_share": float(-2 * cov / vk) if vk > 0 else float("nan"),
                    "tau_bf16": tau, "smin0_over_tau": float(s0[-1] / tau), "median_smin_over_tau": float(np.median(se[:, -1]) / tau),
                    "n_below_tau_base": int((s0 < tau).sum()), "n_below_tau_expert_median": int(np.median((se < tau).sum(1))),
                    "abs_err_bound_over_smin": float(np.max(z["err"][rows_idx] / se[:, -1])),
                    "interlacing_violations": viol, "spearman_kappa_vs_dnorm": float(rho), "argmin_expert": tasks[int(np.argmin(k))]})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{bank}_T{T}_diagnostics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    # Kendall's W of the expert orderings across matrices (per kind and overall)
    def kendall_w(tab):
        ranks = np.argsort(np.argsort(tab, axis=1), axis=1) + 1  # (m, K)
        m, kk = ranks.shape
        rs = ranks.sum(0)
        s_ = np.sum((rs - rs.mean()) ** 2)
        return float(12 * s_ / (m ** 2 * (kk ** 3 - kk)))
    tab = rules.kappa_table(bank, T, "raw")
    kinds = [BK.kind_of(m) for m in BK.module_names(bank)]
    summ = {"kendall_W_all": kendall_w(tab),
            "kendall_W_by_kind": {k: kendall_w(tab[[i for i, x in enumerate(kinds) if x == k]]) for k in BK.PROJ},
            "argmin_frequency": {t: int(np.sum(np.asarray(argmins) == i)) for i, t in enumerate(tasks)},
            "total_interlacing_violations": int(sum(r["interlacing_violations"] for r in per))}
    for kind in BK.PROJ:
        sel = [r for r in per if r["kind"] == kind]
        summ[kind] = {key: float(np.median([r[key] for r in sel])) for key in
                      ("kappa0", "rel_spread", "cv_kappa", "share_var_log_s1", "share_var_log_smin", "smin0_over_tau",
                       "median_smin_over_tau", "n_below_tau_base", "abs_err_bound_over_smin", "spearman_kappa_vs_dnorm")}
        summ[kind]["frac_smin_below_tau"] = float(np.mean([r["median_smin_over_tau"] < 1 for r in sel]))
        summ[kind]["kappa0_range"] = [float(min(r["kappa0"] for r in sel)), float(max(r["kappa0"] for r in sel))]
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--parts", default="C,A,B")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    _init(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    for part in args.parts.split(","):
        t0 = time.time()
        if part == "A":
            res = part_a(args.bank, args.T)
            name = "validation"
        elif part == "B":
            res = part_b(args.bank, args.T, args.replicates, args.procs, args.threads)
            name = "rerounding"
        else:
            res = part_c(args.bank, args.T)
            name = "diagnostics"
        res["seconds"] = time.time() - t0
        (OUT / f"{args.bank}_T{args.T}_{name}.json").write_text(json.dumps(res, indent=1, default=float))
        print(f"part {part} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
