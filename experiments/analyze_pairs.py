"""Does the higher-condition expert of a pair deserve less weight?

From results/pairs/pair_grid_dev.jsonl (global lambda on expert i, 1-lambda on j):

* lam*_ij: argmax of the pair's mean normalised dev accuracy, refined by a
  3-point parabola; ``gain`` = f(lam*) - f(0.5) says how much the choice matters;
* expert "worth" u (sum-zero) from least squares lam*_ij - 0.5 ~ u_i - u_j
  (28 pairs carry only ~7 expert-level degrees of freedom);
* predictors, oriented as (i minus j), all computed before looking at lam*:
  - mean log kappa ratio over linear maps (param-weighted) and the fraction of maps with kappa_i > kappa_j
    (professor's hypothesis: lam*_ij < 0.5 when expert i is more ill-conditioned);
  - log update-norm ratio; transfer asymmetry; gain-over-zeroshot asymmetry.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8  # noqa: E402
from condmerge.runlog import read_jsonl  # noqa: E402
from condmerge.scores import load_stats, stat_table  # noqa: E402


def refine(lams: np.ndarray, f: np.ndarray) -> float:
    k = int(np.argmax(f))
    if 0 < k < len(lams) - 1:
        x0, x1, x2 = lams[k - 1:k + 2]
        y0, y1, y2 = f[k - 1:k + 2]
        den = (y0 - 2 * y1 + y2)
        if den < 0:
            h = x1 - x0
            return float(np.clip(x1 + 0.5 * h * (y0 - y2) / den, x0, x2))
    return float(lams[k])


def main() -> None:
    recs = read_jsonl(RESULTS / "pairs" / "pair_grid_dev.jsonl")
    exp = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())
    ref, zs, cross = exp["expert"]["dev"], exp["zeroshot"]["dev"], exp["cross"]["dev"]
    rows = load_stats()
    kap = stat_table(rows, "full", "kappa2_raw")
    dn = stat_table(rows, "update", "fro")
    sizes = {r["param"]: r["m"] * r["n"] for r in rows if r["kind"] == "base"}
    params = sorted(kap)
    w = np.array([sizes[p] for p in params], dtype=float)
    logk = np.stack([np.log(kap[p]) for p in params])  # (P, 8)
    dn_tot = np.sqrt(np.sum(np.stack([dn[p] ** 2 for p in params]), axis=0))
    curves = defaultdict(dict)
    for r in recs:
        c = r["cfg"]
        curves[(tuple(c["pair"]), c["alpha"])][c["lam"]] = r["metrics"]["mean_norm"]
    out = []
    for (pair, alpha), d in sorted(curves.items()):
        lams = np.array(sorted(d))
        f = np.array([d[x] for x in lams])
        if len(lams) < 11:
            continue
        i, j = TASKS8.index(pair[0]), TASKS8.index(pair[1])
        ls = refine(lams, f)
        f05 = d.get(0.5, np.nan)
        mlr = float(np.sum(w * (logk[:, i] - logk[:, j])) / w.sum())
        frac = float(np.sum(w * (logk[:, i] > logk[:, j])) / w.sum())
        out.append({"pair": pair, "alpha": alpha, "i": i, "j": j, "lam_star": ls, "lam_grid": float(lams[np.argmax(f)]),
                    "f_star": float(f.max()), "f_half": float(f05), "gain": float(f.max() - f05),
                    "mean_logkappa_ratio": mlr, "frac_kappa_i_gt_j": frac,
                    "log_dnorm_ratio": float(np.log(dn_tot[i] / dn_tot[j])),
                    "transfer_asym": float(cross[pair[0]][pair[1]] / ref[pair[1]] - cross[pair[1]][pair[0]] / ref[pair[0]]),
                    "gain_zs_asym": float((ref[pair[0]] - zs[pair[0]]) - (ref[pair[1]] - zs[pair[1]]))})
    res = {"pairs": out, "tests": {}}
    for alpha in sorted({o["alpha"] for o in out}):
        oa = [o for o in out if o["alpha"] == alpha]
        y = np.array([o["lam_star"] - 0.5 for o in oa])
        # expert worth u: y_ij = u_i - u_j, sum u = 0
        A = np.zeros((len(oa) + 1, 8))
        for r_, o in enumerate(oa):
            A[r_, o["i"]], A[r_, o["j"]] = 1, -1
        A[-1] = 1
        u, *_ = np.linalg.lstsq(A, np.append(y, 0), rcond=None)
        resid = y - A[:-1] @ u
        t = {"n_pairs": len(oa), "mean_abs_dev_from_half": float(np.mean(np.abs(y))),
             "median_gain": float(np.median([o["gain"] for o in oa])),
             "worth_u": dict(zip(TASKS8, map(float, u))),
             "worth_model_R2": float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)) if np.sum(y ** 2) > 0 else float("nan")}
        for pred in ("mean_logkappa_ratio", "frac_kappa_i_gt_j", "log_dnorm_ratio", "transfer_asym", "gain_zs_asym"):
            x = np.array([o[pred] for o in oa])
            sp = spearmanr(x, y)
            t[f"spearman_{pred}"] = [float(sp.statistic), float(sp.pvalue)]
        # professor's hypothesis as a sign test on informative pairs (|lam*-0.5| >= 0.05)
        inf_ = [o for o in oa if abs(o["lam_star"] - 0.5) >= 0.05]
        agree = sum((o["mean_logkappa_ratio"] > 0) == (o["lam_star"] < 0.5) for o in inf_)
        t["sign_test_kappa"] = {"n_informative": len(inf_), "agree": agree,
                                "p_two_sided": float(binomtest(agree, len(inf_)).pvalue) if inf_ else float("nan")}
        # expert-level: worth vs mean log kappa over maps (param-weighted), vs update norm
        lk_e = (w[:, None] * logk).sum(0) / w.sum()
        t["expert_level_spearman_worth_vs_meanlogkappa"] = [float(v) for v in spearmanr(u, lk_e)]
        t["expert_level_spearman_worth_vs_dnorm"] = [float(v) for v in spearmanr(u, dn_tot)]
        t["expert_level_spearman_worth_vs_gain_over_zs"] = [float(v) for v in spearmanr(u, [ref[e] - zs[e] for e in TASKS8])]
        res["tests"][str(alpha)] = t
    (RESULTS / "pairs" / "pair_analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["tests"], indent=1))
    for o in out:
        print(f"{o['pair'][0]:>9s}-{o['pair'][1]:<9s} a={o['alpha']} lam*={o['lam_star']:.2f} gain={100 * o['gain']:.2f} "
              f"mlogk={o['mean_logkappa_ratio']:+.4f} frac={o['frac_kappa_i_gt_j']:.2f} ldn={o['log_dnorm_ratio']:+.2f} "
              f"tr={o['transfer_asym']:+.2f}")


if __name__ == "__main__":
    main()
