"""Tables for Report 2 section 3: the pre-declared RegMean 2x2 and the post-hoc strength sweep.

Reads results/regmean/twobytwo_C_{dev,test}.jsonl (pre-declared cells) and
strength_sweep_C_{dev,test}_*.jsonl (equal grids for the kappa-cap ridge, the fixed relative ridge and
fixed shrinkage) and writes results/regmean/twobytwo_summary.json. Selection follows the rule declared in
docs/RESEARCH_LOG.md (2026-10-18 22:40): per arm and per n, the dev-best grid value, with n = 16 scored
by the mean over its calibration subsets.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS  # noqa: E402
from condmerge.runlog import read_jsonl  # noqa: E402

RM = RESULTS / "regmean"
ARMS = ("kcap_ridge", "rel_ridge", "fixed_shrink")


def param(cfg: dict) -> float:
    return cfg["shrink"] if cfg["cell"] == "fixed_shrink" else cfg["kappa_t"]


def index(rows: list[dict]) -> dict:
    """{(cell, param): {n: {seed: (mean_norm %, median rho/(tr/d))}}}"""
    t: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        c = r["cfg"]
        t[(c["cell"], param(c))][c["n"]][c["seed"]] = (100 * r["metrics"]["mean_norm"], r.get("rel_rho_median"))
    return t


def norm(t: dict, cell: str, p, n: int) -> np.ndarray:
    d = t.get((cell, p), {}).get(n, {})
    return np.array([d[s][0] for s in sorted(d)])


def fmt_seeds(x: np.ndarray) -> str:
    return "/".join(f"{v:+.2f}" for v in x)


def predeclared(split: str) -> dict:
    t = index(read_jsonl(RM / f"twobytwo_C_{split}.jsonl"))
    c1, c2, c4, c3 = (("kcap_ridge", 1e5), ("kcap_shrink", 1e5), ("rel_ridge", 1e5), ("fixed_shrink", 0.95))
    out = {}
    print(f"\n== pre-declared 2x2, {split} (kappa_t = 1e5 vs a = 0.95, alpha = 1; mean normalised %)")
    for n in (16, 512):
        v = {k: norm(t, *k, n) for k in (c1, c2, c3, c4)}
        m = {k[0]: float(x.mean()) for k, x in v.items()}
        eff = {"iso_kadaptive": m["kcap_ridge"] - m["kcap_shrink"], "iso_fixed": m["rel_ridge"] - m["fixed_shrink"],
               "adapt_ridge": m["kcap_ridge"] - m["rel_ridge"], "adapt_shrink": m["kcap_shrink"] - m["fixed_shrink"]}
        edge = {k[0]: fmt_seeds(v[k] - v[c3]) for k in (c1, c2, c4)}
        out[n] = {"mean": m, "effects": eff, "diff_vs_shrink_per_seed": edge}
        print(f"  n={n}: " + "  ".join(f"{k} {x:.2f}" for k, x in m.items()))
        print("        vs shrink 0.95 per seed: " + "  ".join(f"{k} {s}" for k, s in edge.items()))
        print("        effects: " + "  ".join(f"{k} {x:+.2f}" for k, x in eff.items()))
    return out


def sweep() -> dict:
    dev = index([r for f in sorted(RM.glob("strength_sweep_C_dev_*.jsonl")) for r in read_jsonl(f)])
    test = index([r for f in sorted(RM.glob("strength_sweep_C_test_*.jsonl")) for r in read_jsonl(f)])
    out: dict = {"dev": {}, "selected": {}, "test": {}}
    print("\n== strength sweep, dev (mean normalised %; n = 16 is the mean of 3 subsets; "
          "rho = median over maps of rho/(tr/d))")
    for arm in ARMS:
        for p in sorted({k[1] for k in dev if k[0] == arm}):
            row = {}
            for n in (16, 512):
                d = dev[(arm, p)].get(n, {})
                if d:
                    row[n] = {"norm": float(np.mean([x[0] for x in d.values()])),
                              "rho": float(np.median([x[1] for x in d.values()])), "k": len(d)}
            out["dev"][f"{arm}|{p:g}"] = row
            print(f"  {arm:12s} {p:<8g} " + "  ".join(
                f"n={n}: {r['norm']:.2f} (rho {r['rho']:.1e}, {r['k']} sets)" for n, r in row.items()))
    for n in (16, 512):
        sel = {}
        for arm in ARMS:
            cand = {k: v[n]["norm"] for k, v in out["dev"].items() if k.startswith(arm + "|") and n in v}
            if cand:
                sel[arm] = max(cand, key=cand.get)
        out["selected"][n] = sel
        print(f"  dev-selected, n={n}: " + "  ".join(f"{a} -> {k.split('|')[1]} ({out['dev'][k][n]['norm']:.2f})"
                                                 for a, k in sel.items()))
        if not test:
            continue
        vals = {a: norm(test, a, float(k.split("|")[1]), n) for a, k in sel.items()}
        if not all(len(v) for v in vals.values()):
            print(f"  test, n={n}: incomplete")
            continue
        res = {a: {"seeds": [float(x) for x in v], "mean": float(v.mean())} for a, v in vals.items()}
        res["kcap_minus_rel_ridge"] = fmt_seeds(vals["kcap_ridge"] - vals["rel_ridge"])
        res["kcap_minus_shrink"] = fmt_seeds(vals["kcap_ridge"] - vals["fixed_shrink"])
        res["rel_ridge_minus_shrink"] = fmt_seeds(vals["rel_ridge"] - vals["fixed_shrink"])
        out["test"][n] = res
        print(f"  test, n={n}: " + "  ".join(f"{a} {r['mean']:.2f}" for a, r in res.items() if isinstance(r, dict)))
        print(f"        kcap - fixed ridge {res['kcap_minus_rel_ridge']}; kcap - shrink {res['kcap_minus_shrink']}; "
              f"fixed ridge - shrink {res['rel_ridge_minus_shrink']}")
    return out


def reproducibility() -> list:
    """Configs evaluated both in the pre-declared run and in the sweep (same split) should agree."""
    pairs = []
    for split in ("dev", "test"):
        a = {r["key"]: r for r in read_jsonl(RM / f"twobytwo_C_{split}.jsonl")}
        for f in sorted(RM.glob(f"strength_sweep_C_{split}_*.jsonl")):
            for r in read_jsonl(f):
                if r["key"] in a:
                    pairs.append(abs(r["metrics"]["mean_norm"] - a[r["key"]]["metrics"]["mean_norm"]) * 100)
    if pairs:
        print(f"\n== reproducibility: {len(pairs)} repeated configs, max |diff| = {max(pairs):.3f} pt")
    return pairs


def main() -> None:
    summary = {"predeclared": {s: predeclared(s) for s in ("dev", "test")}, "sweep": sweep()}
    rep = reproducibility()
    summary["repeat_max_abs_diff_pt"] = max(rep) if rep else None
    (RM / "twobytwo_summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
