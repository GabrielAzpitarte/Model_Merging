"""Does the dev-optimal layerwise allocation (oracle, budget D) follow the condition number?

For each oracle fit (results/aa/oracle_lambda/<variant>_seed<s>.json):
  * seed-to-seed agreement of log lambda (Spearman over all (map, expert));
  * within-map Spearman(lambda^p, score^p) for kappa and controls, mean over maps, permutation null;
  * between-expert / within-expert split (as in aa_decompose_pref.py);
  * expert worth (mean lambda over maps) vs mean log kappa;
  * attention vs MLP breakdown of the within-map correlation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE, RESULTS  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402


def ranks(m):
    return np.apply_along_axis(rankdata, 1, m)


def within_corr(ra, rb):
    ea, eb = ra - ra.mean(0, keepdims=True), rb - rb.mean(0, keepdims=True)
    return float(np.sum(ea * eb) / np.sqrt(np.sum(ea ** 2) * np.sum(eb ** 2)))


def mean_rowcorr(ra, rb):
    a = ra - ra.mean(1, keepdims=True)
    b = rb - rb.mean(1, keepdims=True)
    return float(np.mean(np.sum(a * b, 1) / np.sqrt(np.sum(a ** 2, 1) * np.sum(b ** 2, 1))))


def main() -> None:
    st = np.load(CACHE / "aa" / "fullft_stats.npz")
    names, tasks = list(st["names"]), list(st["tasks"])
    od = RESULTS / "aa" / "oracle_lambda"
    fits = {}
    for p in sorted(od.glob("*_seed*.json")):
        rec = json.loads(p.read_text())
        lam = np.array([rec["lambda"][n] for n in names])
        fits[p.stem] = {"lam": lam, "alpha": rec["alpha"], "dev": rec["dev"]["avg_acc"], "test": rec["test"]["avg_acc"],
                        "variant": rec["variant"], "seed": rec["seed"]}
    rng = np.random.default_rng(0)
    attn = np.array(["self_attn" in n for n in names])
    out = {}
    for tag, f in fits.items():
        lam = f["lam"]
        if np.allclose(lam, lam[0]):  # global variant: identical rows
            out[tag] = {"alpha": f["alpha"], "dev": f["dev"], "test": f["test"], "expert_lambda": dict(zip(tasks, lam[0].round(4).tolist())),
                        "spearman_lambda_vs_mean_logkappa": float(spearmanr(lam[0], np.log(st["kappa_W"]).mean(0)).statistic),
                        "spearman_lambda_vs_mean_froD": float(spearmanr(lam[0], st["fro_D"].mean(0)).statistic)}
            continue
        rl = ranks(lam)
        res = {"alpha": f["alpha"], "dev": f["dev"], "test": f["test"],
               "expert_mean_lambda": dict(zip(tasks, lam.mean(0).round(4).tolist())),
               "n_eff_mean": float(np.mean(np.exp(-np.sum(lam * np.log(np.maximum(lam, 1e-12)), 1)))),
               "max_lambda_mean": float(lam.max(1).mean())}
        for sname in ("kappa_W", "smin_W", "fro_D", "stable_rank_D", "entropy_rank_D"):
            S = st[sname]
            rs = ranks(S)
            m_all = mean_rowcorr(rl, rs)
            null = []
            for _ in range(5000):
                perm = np.argsort(rng.random(S.shape), axis=1)
                null.append(mean_rowcorr(rl, np.take_along_axis(rs, perm, 1)))
            null = np.array(null)
            w = within_corr(rl, rs)
            nullw = []
            for _ in range(2000):
                Sp = S.copy()
                for i in range(S.shape[1]):
                    Sp[:, i] = S[rng.permutation(S.shape[0]), i]
                nullw.append(within_corr(rl, ranks(Sp)))
            nullw = np.array(nullw)
            res[sname] = {"mean_within_map_spearman": m_all, "p_le": float(np.mean(null <= m_all)), "p_ge": float(np.mean(null >= m_all)),
                          "attn_mean": mean_rowcorr(rl[attn], rs[attn]), "mlp_mean": mean_rowcorr(rl[~attn], rs[~attn]),
                          "between_expert": float(np.corrcoef(rl.mean(0), rs.mean(0))[0, 1]),
                          "within_expert": w, "p_within_le": float(np.mean(nullw <= w)), "p_within_ge": float(np.mean(nullw >= w))}
        out[tag] = res
    # seed agreement
    agree = {}
    for v in {f["variant"] for f in fits.values()}:
        ss = sorted([t for t, f in fits.items() if f["variant"] == v])
        if len(ss) >= 2:
            a, b = np.log(fits[ss[0]]["lam"]), np.log(fits[ss[1]]["lam"])
            agree[v] = {"spearman_loglambda_all": float(spearmanr(a.ravel(), b.ravel()).statistic),
                        "mean_within_map_spearman": mean_rowcorr(ranks(a), ranks(b))}
    out["seed_agreement"] = agree
    (od / "analysis_oracle_vs_kappa.json").write_text(json.dumps(out, indent=1))
    for tag, res in out.items():
        if tag == "seed_agreement":
            print("seed agreement", json.dumps(res))
            continue
        print(f"=== {tag}: alpha {res['alpha']:.3f} dev {res['dev']:.4f} test {res['test']:.4f}")
        for k, v in res.items():
            if isinstance(v, dict) and "mean_within_map_spearman" in v:
                print(f"  {k:15s} within-map rho={v['mean_within_map_spearman']:+.3f} (p<= {v['p_le']:.4f}, p>= {v['p_ge']:.4f}) "
                      f"attn {v['attn_mean']:+.3f} mlp {v['mlp_mean']:+.3f} | between {v['between_expert']:+.3f} within-expert {v['within_expert']:+.3f} "
                      f"(p<= {v['p_within_le']:.3f}, p>= {v['p_within_ge']:.3f})")
            elif k in ("expert_mean_lambda", "expert_lambda", "n_eff_mean", "max_lambda_mean") or k.startswith("spearman"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
