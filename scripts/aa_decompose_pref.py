"""Is the score-preference association a per-layer mechanism or just expert identity? (diagnostic, budget D)

Within each linear map p, rank pref and score s across the 8 experts (ranks 1..8). Then
  expert effect   a_i = mean_p rank_pref(p, i)            (some experts are wanted everywhere)
  between-expert  corr(a_i, mean_p rank_s(p, i)) over the 8 experts
  within-expert   corr of residual ranks  rank(p,i) - mean_p rank(p,i)  pooled over (p, i)
Null for the within-expert part: permute the maps independently inside each expert column of s
(keeps each expert's score distribution, i.e. the expert effect; destroys the layer-specific link).
Also per expert: Spearman over maps of pref vs score (73 points), averaged.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE, RESULTS  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402


def ranks_within_map(m: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(rankdata, 1, m)


def within_corr(rp: np.ndarray, rs: np.ndarray) -> float:
    ep = rp - rp.mean(0, keepdims=True)
    es = rs - rs.mean(0, keepdims=True)
    return float(np.sum(ep * es) / np.sqrt(np.sum(ep ** 2) * np.sum(es ** 2)))


def main() -> None:
    st = np.load(CACHE / "aa" / "fullft_stats.npz")
    names, tasks = list(st["names"]), list(st["tasks"])
    rng = np.random.default_rng(0)
    out = {}
    for a in ("1", "2.4"):
        rows = list(csv.DictReader((RESULTS / "aa" / "grad_pref" / f"grad_pref_alpha{float(a):g}.csv").open()))
        g = np.zeros((len(names), len(tasks)))
        idx = {n: i for i, n in enumerate(names)}
        for r in rows:
            if r["param"] in idx:
                g[idx[r["param"]], tasks.index(r["expert"])] = float(r["grad_tied"])
        pref = -(g - g.mean(1, keepdims=True))
        rp = ranks_within_map(pref)
        res = {"expert_mean_pref_rank": dict(zip(tasks, rp.mean(0).round(3).tolist()))}
        for sname in ("kappa_W", "fro_D", "stable_rank_D", "entropy_rank_D", "smin_W", "kplus_D_0.01"):
            S = st[sname]
            rs = ranks_within_map(S)
            between = float(np.corrcoef(rp.mean(0), rs.mean(0))[0, 1])
            w = within_corr(rp, rs)
            null = []
            for _ in range(5000):
                Sp = S.copy()
                for i in range(S.shape[1]):
                    Sp[:, i] = S[rng.permutation(S.shape[0]), i]
                null.append(within_corr(rp, ranks_within_map(Sp)))
            null = np.array(null)
            per_exp = [spearmanr(pref[:, i] / np.abs(pref).max(1), S[:, i]).statistic for i in range(len(tasks))]
            res[sname] = {"between_expert_corr": between, "within_expert_corr": w,
                          "p_within_le": float(np.mean(null <= w)), "p_within_ge": float(np.mean(null >= w)),
                          "null_sd": float(null.std()), "expert_mean_score_rank": dict(zip(tasks, rs.mean(0).round(3).tolist())),
                          "mean_per_expert_spearman_over_maps": float(np.mean(per_exp))}
        out[a] = res
        print(f"=== alpha {a}: expert mean pref rank {res['expert_mean_pref_rank']}")
        for k, v in res.items():
            if k.startswith("expert"):
                continue
            print(f"  {k:15s} between={v['between_expert_corr']:+.3f} within={v['within_expert_corr']:+.3f} "
                  f"(null sd {v['null_sd']:.3f}, p<= {v['p_within_le']:.4f}, p>= {v['p_within_ge']:.4f}) "
                  f"score ranks {v['expert_mean_score_rank']}")
    (RESULTS / "aa" / "grad_pref" / "decompose_pref.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
