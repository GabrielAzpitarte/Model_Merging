"""Does the condition number predict which expert deserves more weight, layer by layer? (diagnostic, budget D)

Inputs: results/aa/grad_pref/grad_pref_alpha{a}.csv (from aa_grad_preference.py) and per-(map, expert)
spectral statistics (computed here, cached to artifacts/cache/aa/fullft_stats.npz).

For each linear map p, across the K=8 experts:
  * Spearman(pref^p, kappa^p), with pref = -(dL/dlambda centred) using the tied (weight+bias) derivative.
    The literal rule predicts NEGATIVE correlation (higher kappa -> lower preference).
  * the same for log kappa (identical ranks), the update norm ||Delta||_F, sigma_min(W_i), and the base-relative
    log kappa(W_i)/kappa(W0) (identical ranks to kappa within a layer -> reported once).
  * partial rank correlation of pref with kappa controlling for ||Delta||_F.
First-order effect of moving from uniform toward softmax(-beta*s) (beta -> 0+):
    dL ~= -(beta/K) * sum_p sum_i g_i^p (s_i^p - mean_i s^p)
  reported per score s in {kappa (raw, one global beta as in the literal rule), log kappa, ||Delta||_F,
  layer-standardised kappa}; negative dL = the rule's direction lowers dev CE.
Permutation test: shuffle each map's scores across experts (10k draws) for mean Spearman and for dL.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402


def _map_stats(args):
    """CPU float64 statistics for one linear map across experts (process-pool worker)."""
    w0, ws = args
    w0 = w0.reshape(w0.shape[0], -1).astype(np.float64)
    s0 = np.linalg.svd(w0, compute_uv=False)
    res = []
    for w in ws:
        w = w.reshape(w0.shape).astype(np.float64)
        d = w - w0
        s = np.linalg.svd(w, compute_uv=False)
        sd = np.linalg.svd(d, compute_uv=False)
        p = sd ** 2 / np.sum(sd ** 2)
        p = p[p > 0]
        row = {"kappa_W": s[0] / s[-1], "smax_W": s[0], "smin_W": s[-1], "kappa_W0": s0[0] / s0[-1],
               "smax_W0": s0[0], "smin_W0": s0[-1], "fro_D": np.sqrt(np.sum(sd ** 2)), "spec_D": sd[0],
               "stable_rank_D": np.sum(sd ** 2) / sd[0] ** 2, "entropy_rank_D": np.exp(-np.sum(p * np.log(p)))}
        for tau in (1e-2, 1e-3):
            r = int(np.sum(sd > tau * sd[0]))
            row[f"kplus_D_{tau:g}"] = sd[0] / sd[r - 1]
        res.append(row)
    return res


def compute_stats(path: Path) -> dict:
    import os
    from concurrent.futures import ProcessPoolExecutor

    from condmerge.aa_common import is_linear_map, load_base_unprefixed, load_expert_unprefixed

    base = load_base_unprefixed()
    experts = {t: load_expert_unprefixed(t) for t in TASKS8}
    names = [k for k in base if is_linear_map(k, base[k])]
    jobs = [(base[n].numpy(), [experts[t][n].numpy() for t in TASKS8]) for n in names]
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(_map_stats, jobs))
    keys = list(res[0][0].keys())
    arr = {k: np.array([[r[k] for r in rr] for rr in res]) for k in keys}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, names=np.array(names), tasks=np.array(TASKS8), **arr)
    return load_stats(path)


def load_stats(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def partial_spearman(x, y, z) -> float:
    """Rank correlation of x and y after linearly removing z's ranks from both."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1,2.4")
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--out", default=str(RESULTS / "aa" / "grad_pref"))
    args = ap.parse_args()
    out = Path(args.out)
    spath = CACHE / "aa" / "fullft_stats.npz"
    st = load_stats(spath) if spath.exists() else compute_stats(spath)
    names = list(st["names"])
    tasks = list(st["tasks"])
    rng = np.random.default_rng(0)
    report = {}
    per_map_rows = []
    for a in args.alphas.split(","):
        rows = list(csv.DictReader((out / f"grad_pref_alpha{float(a):g}.csv").open()))
        g = np.zeros((len(names), len(tasks)))
        idx = {n: i for i, n in enumerate(names)}
        for r in rows:
            if r["param"] in idx:
                g[idx[r["param"]], tasks.index(r["expert"])] = float(r["grad_tied"])
        gc = g - g.mean(1, keepdims=True)
        pref = -gc
        scores = {
            "kappa": st["kappa_W"],
            "log_kappa": np.log(st["kappa_W"]),
            "kappa_layer_std": (st["kappa_W"] - st["kappa_W"].mean(1, keepdims=True)) / (st["kappa_W"].std(1, keepdims=True) + 1e-300),
            "fro_D": st["fro_D"],
            "smin_W": st["smin_W"],
            "stable_rank_D": st["stable_rank_D"],
            "entropy_rank_D": st["entropy_rank_D"],
        }
        res = {}
        for sname, S in scores.items():
            rho = np.array([spearmanr(pref[p], S[p]).statistic for p in range(len(names))])
            # first-order dL of moving toward softmax(-beta*S), beta->0+, up to the positive factor beta/K
            Sc = S - S.mean(1, keepdims=True)
            dL = -float(np.sum(g * Sc))
            dL_per_map = -np.sum(g * Sc, axis=1)
            # permutation null: shuffle scores within each map
            null_rho, null_dL = [], []
            for _ in range(args.perm):
                perm = np.argsort(rng.random(S.shape), axis=1)
                Sp = np.take_along_axis(S, perm, 1)
                Spc = Sp - Sp.mean(1, keepdims=True)
                null_dL.append(-float(np.sum(g * Spc)))
                # rank correlation under permutation: mean over maps of corr(rank(pref), rank(Sp))
            rp = np.apply_along_axis(rankdata, 1, pref)
            rs = np.apply_along_axis(rankdata, 1, S)
            rpc = rp - rp.mean(1, keepdims=True)
            for _ in range(args.perm):
                perm = np.argsort(rng.random(S.shape), axis=1)
                rsp = np.take_along_axis(rs, perm, 1)
                rspc = rsp - rsp.mean(1, keepdims=True)
                null_rho.append(float(np.mean(np.sum(rpc * rspc, 1) / np.sqrt(np.sum(rpc ** 2, 1) * np.sum(rspc ** 2, 1)))))
            null_rho, null_dL = np.array(null_rho), np.array(null_dL)
            res[sname] = {
                "mean_spearman": float(np.nanmean(rho)),
                "frac_negative": float(np.mean(rho < 0)),
                "p_mean_spearman_le": float(np.mean(null_rho <= np.nanmean(rho))),
                "p_mean_spearman_ge": float(np.mean(null_rho >= np.nanmean(rho))),
                "first_order_dL": dL,
                "p_dL_le": float(np.mean(null_dL <= dL)),
                "top_maps_by_abs_dL": [(names[i], float(dL_per_map[i])) for i in np.argsort(-np.abs(dL_per_map))[:5]],
                "frac_dL_from_top5": float(np.sum(np.sort(np.abs(dL_per_map))[-5:]) / np.sum(np.abs(dL_per_map))),
            }
        # partial: kappa vs pref controlling for the update norm
        pr = np.array([partial_spearman(pref[p], st["kappa_W"][p], st["fro_D"][p]) for p in range(len(names))])
        res["kappa_partial_given_froD"] = {"mean": float(np.nanmean(pr)), "frac_negative": float(np.nanmean(pr < 0))}
        # how kappa relates to the update norm within a layer (C11 prediction: negative where kappa0 is huge)
        kd = np.array([spearmanr(st["kappa_W"][p], st["fro_D"][p]).statistic for p in range(len(names))])
        big = st["kappa_W0"][:, 0] > 1e4
        res["spearman_kappa_vs_froD"] = {"mean_all": float(np.nanmean(kd)), "mean_kappa0_gt_1e4": float(np.nanmean(kd[big])),
                                         "mean_kappa0_le_1e4": float(np.nanmean(kd[~big])), "n_big": int(big.sum())}
        ks = np.array([spearmanr(st["kappa_W"][p], st["smin_W"][p]).statistic for p in range(len(names))])
        res["spearman_kappa_vs_sminW"] = float(np.nanmean(ks))
        report[a] = res
        for p, n in enumerate(names):
            per_map_rows.append({"alpha": a, "param": n, "kappa0": st["kappa_W0"][p, 0],
                                 "kappa_min": st["kappa_W"][p].min(), "kappa_max": st["kappa_W"][p].max(),
                                 "rho_pref_kappa": spearmanr(pref[p], st["kappa_W"][p]).statistic,
                                 "rho_pref_froD": spearmanr(pref[p], st["fro_D"][p]).statistic,
                                 "rho_kappa_froD": kd[p], "argmin_kappa": tasks[int(np.argmin(st["kappa_W"][p]))],
                                 "argmax_pref": tasks[int(np.argmax(pref[p]))]})
    (out / "analysis_grad_vs_kappa.json").write_text(json.dumps(report, indent=1))
    with (out / "per_map_grad_vs_kappa.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_map_rows[0].keys()))
        w.writeheader()
        w.writerows(per_map_rows)
    for a, res in report.items():
        print(f"=== alpha {a}")
        for k, v in res.items():
            if isinstance(v, dict) and "mean_spearman" in v:
                print(f"  {k:16s} mean rho(pref,s)={v['mean_spearman']:+.3f} frac<0={v['frac_negative']:.2f} "
                      f"p<= {v['p_mean_spearman_le']:.4f} p>= {v['p_mean_spearman_ge']:.4f} | first-order dL={v['first_order_dL']:+.3e} "
                      f"p(dL<=)={v['p_dL_le']:.4f} top5 share {v['frac_dL_from_top5']:.2f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
