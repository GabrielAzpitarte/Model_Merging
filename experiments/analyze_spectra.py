"""What does the literal score actually measure on the FusionBench bank?

Reads results/spectra/{stats,first_order,precision,overlap}.csv and writes
results/spectra/summary.json plus a per-map table results/spectra/per_map.csv:

* how far kappa2(W_i) is from kappa2(W0) and how much it varies across experts;
* whether fp32-vs-fp64 SVD noise is small relative to that variation;
* whether the first-order prediction (linear rank-2 probe of D_i) explains the shifts;
* Spearman correlation across experts, within each map, between kappa2(W_i) and
  ||D_i||_F (peer prediction C11: negative in near-singular maps);
* which expert the argmin-kappa limit selects in each map.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8  # noqa: E402
from condmerge.scores import load_stats, stat_table  # noqa: E402


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    d = RESULTS / "spectra"
    rows = load_stats()
    kap = stat_table(rows, "full", "kappa2_raw")
    smin = stat_table(rows, "full", "smin")
    smax = stat_table(rows, "full", "smax")
    dn = stat_table(rows, "update", "fro")
    base = {r["param"]: r for r in rows if r["kind"] == "base"}
    upd_rank = stat_table(rows, "update", "rank_default")
    fo = defaultdict(dict)
    for r in read(d / "first_order.csv"):
        fo[r["param"]][r["expert"]] = (float(r["actual_shift"]), float(r["pred_shift"]))
    prec = defaultdict(dict)
    for r in read(d / "precision.csv"):
        prec[r["param"]][r["expert"]] = (float(r["kappa_fp64"]), float(r["kappa_fp32"]))
    per_map = []
    sel = Counter()
    sel_by_module = defaultdict(Counter)
    for p, v in kap.items():
        b = base[p]
        k0 = b["kappa2_raw"]
        spread = float(np.std(v))
        pr = np.array([prec[p][e][1] - prec[p][e][0] for e in TASKS8])
        act = np.array([fo[p][e][0] for e in TASKS8])
        pred = np.array([fo[p][e][1] for e in TASKS8])
        rho_norm = spearmanr(v, dn[p]).statistic
        rho_fo = pearsonr(act, pred).statistic if np.std(pred) > 0 and np.std(act) > 0 else float("nan")
        winner = TASKS8[int(np.argmin(v))]
        sel[winner] += 1
        sel_by_module[b["module"]][winner] += 1
        per_map.append({
            "param": p, "layer": int(b["layer"]), "module": b["module"], "kappa0": k0,
            "kappa_min": float(v.min()), "kappa_max": float(v.max()), "kappa_spread_sd": spread,
            "rel_spread": spread / k0, "max_rel_shift": float(np.max(np.abs(v - k0)) / k0),
            "fp32_noise_sd_over_spread": float(np.std(pr) / spread) if spread > 0 else float("nan"),
            "first_order_r": rho_fo, "first_order_ratio_med": float(np.median(pred / act)) if np.all(act != 0) else float("nan"),
            "spearman_kappa_vs_dnorm": rho_norm,
            "spearman_smin_vs_dnorm": spearmanr(smin[p], dn[p]).statistic,
            "spearman_smax_vs_dnorm": spearmanr(smax[p], dn[p]).statistic,
            "argmin_expert": winner, "update_rank_default_min": float(upd_rank[p].min()),
            "base_smin_over_dnorm_med": float(b["smin"] / np.median(dn[p])),
        })
    keys = list(per_map[0])
    with (d / "per_map.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_map)
    pm = per_map
    by_mod = defaultdict(list)
    for r in pm:
        by_mod[r["module"]].append(r)
    summary = {
        "n_maps": len(pm),
        "kappa0_range": [min(r["kappa0"] for r in pm), max(r["kappa0"] for r in pm)],
        "median_rel_spread": float(np.median([r["rel_spread"] for r in pm])),
        "median_max_rel_shift": float(np.median([r["max_rel_shift"] for r in pm])),
        "median_fp32_noise_over_spread": float(np.nanmedian([r["fp32_noise_sd_over_spread"] for r in pm])),
        "first_order_r_median": float(np.nanmedian([r["first_order_r"] for r in pm])),
        "spearman_kappa_dnorm_median": float(np.nanmedian([r["spearman_kappa_vs_dnorm"] for r in pm])),
        "argmin_selection_counts": dict(sel),
        "by_module": {m: {
            "n": len(rs),
            "kappa0_median": float(np.median([r["kappa0"] for r in rs])),
            "rel_spread_median": float(np.median([r["rel_spread"] for r in rs])),
            "first_order_r_median": float(np.nanmedian([r["first_order_r"] for r in rs])),
            "spearman_kappa_dnorm_median": float(np.nanmedian([r["spearman_kappa_vs_dnorm"] for r in rs])),
            "spearman_smin_dnorm_median": float(np.nanmedian([r["spearman_smin_vs_dnorm"] for r in rs])),
            "argmin_selection": dict(sel_by_module[m]),
        } for m, rs in by_mod.items()},
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("\nper-map (layer, module, kappa0, rel_spread, fo_r, rho(kappa,|D|), argmin):")
    for r in sorted(pm, key=lambda r: (r["layer"], r["module"])):
        print(f"L{r['layer']:>2} {r['module']:9s} k0={r['kappa0']:10.3g} rel_sd={r['rel_spread']:8.2e} "
              f"fp32/sd={r['fp32_noise_sd_over_spread']:8.2e} fo_r={r['first_order_r']:6.3f} "
              f"rho={r['spearman_kappa_vs_dnorm']:6.2f} sel={r['argmin_expert']}")


if __name__ == "__main__":
    main()
