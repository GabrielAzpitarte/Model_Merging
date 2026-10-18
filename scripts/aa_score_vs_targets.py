"""Every per-(map, expert) score vs every allocation target, with the expert-identity decomposition.

Scores: artifacts/cache/aa/fullft_stats.npz (budget A, weights only) and data_supported_stats.npz (budget B).
Targets (budget D, diagnostics): gradient preference at the uniform merge (alpha 1, 2.4) and the oracle
layerwise lambdas (alpha learned / alpha=1; seed 0).
For each (score, target):
  within_map  mean over maps of Spearman across the 8 experts (perm null: shuffle scores within maps)
  between     corr over experts of mean within-map ranks (8 points)
  within_exp  corr of residual ranks after removing expert means (null: shuffle maps within each expert)
Ranks via double argsort (continuous scores, ties negligible). Writes results/aa/score_vs_targets.{json,md}.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE, RESULTS  # noqa: E402

import numpy as np  # noqa: E402

N_PERM = 4000


def rk(m: np.ndarray) -> np.ndarray:
    return m.argsort(1).argsort(1).astype(np.float64)


def rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), 0.0)


def within_exp(ra: np.ndarray, rb: np.ndarray) -> float:
    ea, eb = ra - ra.mean(0, keepdims=True), rb - rb.mean(0, keepdims=True)
    return float((ea * eb).sum() / np.sqrt((ea ** 2).sum() * (eb ** 2).sum()))


def main() -> None:
    rng = np.random.default_rng(0)
    A = np.load(CACHE / "aa" / "fullft_stats.npz")
    names, tasks = list(A["names"]), list(A["tasks"])
    scores = {f"A:{k}": A[k] for k in ("kappa_W", "smin_W", "fro_D", "stable_rank_D", "entropy_rank_D", "kplus_D_0.01")}
    scores["A:logkappa_rel"] = np.log(A["kappa_W"]) - np.log(A["kappa_W0"])
    bp = CACHE / "aa" / "data_supported_stats.npz"
    if bp.exists():
        B = np.load(bp)
        assert list(B["names"]) == names
        for k in ("fnorm_D", "rel_fnorm_D", "kappa_WC", "dlogkappa_WC", "srank_DC", "erank_DC", "kplus_DC"):
            scores[f"B:{k}"] = B[k]
    targets = {}
    for a in ("1", "2.4"):
        g = np.zeros((len(names), len(tasks)))
        idx = {n: i for i, n in enumerate(names)}
        for r in csv.DictReader((RESULTS / "aa" / "grad_pref" / f"grad_pref_alpha{float(a):g}.csv").open()):
            if r["param"] in idx:
                g[idx[r["param"]], tasks.index(r["expert"])] = float(r["grad_tied"])
        targets[f"gradpref_a{a}"] = -(g - g.mean(1, keepdims=True))
    for tag in ("layerwise_alpha_seed0", "layerwise_a1_seed0"):
        rec = json.loads((RESULTS / "aa" / "oracle_lambda" / f"{tag}.json").read_text())
        targets[f"oracle_{tag.replace('_seed0', '')}"] = np.array([rec["lambda"][n] for n in names])
    attn = np.array(["self_attn" in n for n in names])
    out = {}
    for tn, T in targets.items():
        rt = rk(T)
        out[tn] = {}
        for sn, S in scores.items():
            rs = rk(S)
            wm = float(rowcorr(rt, rs).mean())
            null = np.array([rowcorr(rt, np.take_along_axis(rs, np.argsort(rng.random(rs.shape), 1), 1)).mean() for _ in range(N_PERM)])
            we = within_exp(rt, rs)
            nullw = []
            for _ in range(N_PERM // 2):
                Sp = np.stack([S[rng.permutation(S.shape[0]), i] for i in range(S.shape[1])], 1)
                nullw.append(within_exp(rt, rk(Sp)))
            nullw = np.array(nullw)
            out[tn][sn] = {"within_map": wm, "p_two": float(np.mean(np.abs(null) >= abs(wm))),
                           "attn": float(rowcorr(rt[attn], rs[attn]).mean()), "mlp": float(rowcorr(rt[~attn], rs[~attn]).mean()),
                           "between": float(np.corrcoef(rt.mean(0), rs.mean(0))[0, 1]),
                           "within_exp": we, "p_within_two": float(np.mean(np.abs(nullw - nullw.mean()) >= abs(we - nullw.mean())))}
    (RESULTS / "aa" / "score_vs_targets.json").write_text(json.dumps(out, indent=1))
    lines = ["| target | score | within-map rho (p) | attn | mlp | between-expert (8 pts) | within-expert (p) |", "|---|---|---|---|---|---|---|"]
    for tn, d in out.items():
        for sn, v in d.items():
            lines.append(f"| {tn} | {sn} | {v['within_map']:+.3f} ({v['p_two']:.3f}) | {v['attn']:+.3f} | {v['mlp']:+.3f} | "
                         f"{v['between']:+.2f} | {v['within_exp']:+.3f} ({v['p_within_two']:.3f}) |")
    (RESULTS / "aa" / "score_vs_targets.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
