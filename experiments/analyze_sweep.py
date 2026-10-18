"""Summarise rule_sweep JSONL: alpha=1 curves, matched-budget best configs, shuffled null.

Selection is on the split recorded in the file (dev). For each rule:
* ``fixed``: alpha = 1 (the literal normalised average), every b;
* ``tuned``: the best (b, alpha) over the full shared grid (same budget for every rule
  that has a beta; 'uniform'/'metagpt' only tune alpha).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS  # noqa: E402
from condmerge.runlog import read_jsonl  # noqa: E402


def load(path: Path, experts_len: int | None = 8) -> list[dict]:
    recs = read_jsonl(path)
    if experts_len:
        recs = [r for r in recs if len(r["cfg"]["experts"]) == experts_len]
    return recs


def table(recs: list[dict], metric: str = "mean_norm") -> dict:
    by = defaultdict(dict)  # rule -> {(b, alpha): rec}
    for r in recs:
        c = r["cfg"]
        by[c["rule"]][(c["b"], c["alpha"])] = r
    return by


def fmt(x: float) -> str:
    return f"{100 * x:6.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RESULTS / "sweeps" / "rule_sweep_dev.jsonl"))
    ap.add_argument("--metric", default="mean_norm")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    recs = load(Path(args.file))
    by = table(recs)
    metric = args.metric
    uni = by.get("uniform", {})
    uni1 = uni.get((0.0, 1.0))
    uni_best = max(uni.values(), key=lambda r: r["metrics"][metric]) if uni else None
    print(f"uniform alpha=1: {fmt(uni1['metrics'][metric]) if uni1 else 'NA'};  "
          f"uniform tuned: {fmt(uni_best['metrics'][metric]) if uni_best else 'NA'} "
          f"(alpha={uni_best['cfg']['alpha'] if uni_best else 'NA'})")
    summary = {}
    bs = sorted({k[0] for r in by.values() for k in r}, key=lambda x: (math.isinf(x), x))
    print("\nalpha = 1 (literal normalised average), mean normalised acc (%), by b:")
    print(f"{'rule':24s}" + "".join(f"{('inf' if math.isinf(b) else f'{b:g}'):>8s}" for b in bs))
    for rule, d in sorted(by.items()):
        row = [d.get((b, 1.0)) for b in bs]
        print(f"{rule:24s}" + "".join(f"{fmt(r['metrics'][metric]) if r else '      ':>8s}" for r in row))
    print("\nn_eff (mean over linear maps) by b:")
    for rule, d in sorted(by.items()):
        row = [d.get((b, 1.0)) for b in bs]
        print(f"{rule:24s}" + "".join(
            f"{(r['coeff_summary'].get('n_eff_mean', float('nan')) if r else float('nan')):8.2f}" for r in row))
    print("\nBest over the shared (b, alpha) grid (dev-selected):")
    for rule, d in sorted(by.items()):
        best = max(d.values(), key=lambda r: r["metrics"][metric])
        best1 = max((r for (b, a), r in d.items() if a == 1.0), key=lambda r: r["metrics"][metric], default=None)
        summary[rule] = {"best": {"b": best["cfg"]["b"], "alpha": best["cfg"]["alpha"], metric: best["metrics"][metric],
                                  "mean_acc": best["metrics"]["mean_acc"], "min_norm": best["metrics"]["min_norm"]},
                         "best_alpha1": ({"b": best1["cfg"]["b"], metric: best1["metrics"][metric]} if best1 else None),
                         "n_configs": len(d)}
        print(f"{rule:24s} best {fmt(best['metrics'][metric])} (b={best['cfg']['b']:g}, alpha={best['cfg']['alpha']:g}, "
              f"acc {fmt(best['metrics']['mean_acc'])}, min {fmt(best['metrics']['min_norm'])}); "
              f"best@alpha1 {fmt(best1['metrics'][metric]) if best1 else 'NA'}  [{len(d)} cfgs]")
    shuf = {r: v for r, v in summary.items() if r.startswith("shuf")}
    if shuf:
        bases = sorted({r.split(":", 1)[1] for r in shuf})
        for base_rule in bases:
            vals = [v["best"][metric] for r, v in shuf.items() if r.endswith(":" + base_rule)]
            if base_rule in summary:
                obs = summary[base_rule]["best"][metric]
                print(f"\nshuffled null for {base_rule}: n={len(vals)} mean {fmt(np.mean(vals))} sd {100 * np.std(vals):.2f} "
                      f"max {fmt(np.max(vals))}; observed {fmt(obs)}; rank {sum(v >= obs for v in vals)}/{len(vals)} >= obs")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
