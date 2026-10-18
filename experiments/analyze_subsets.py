"""Pre-registered subsets (configs/subsets.json): literal rule vs uniform, reversed, log and shuffled.

For each subset: dev-best over the reduced shared grid (b in {1,4,16,inf} x alpha in {1,1.6,2.2})
per rule, the alpha = 1 best, and the literal rule's rank among {kappa, shuf0..2}. Across subsets:
mean differences to uniform and a sign count.
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


def main() -> None:
    subsets = json.loads(Path("configs/subsets.json").read_text())["subsets"]
    out, diffs = {}, defaultdict(list)
    print(f"{'subset':11s} {'K':>2s} " + " ".join(f"{r:>12s}" for r in
          ["uniform", "kappa", "kappa_rev", "logkappa", "shuf_mean", "shuf_max"]) + "   kappa@a1  uni@a1")
    for name, experts in subsets.items():
        path = RESULTS / "sweeps" / f"subsets_dev_{name}.jsonl"
        if not path.exists():
            continue
        by = defaultdict(dict)
        for r in read_jsonl(path):
            by[r["cfg"]["rule"]][(r["cfg"]["b"], r["cfg"]["alpha"])] = r["metrics"]["mean_norm"]
        if len(by.get("shuf2:kappa", {})) < 12:
            print(f"{name}: incomplete")
            continue
        best = {r: max(d.values()) for r, d in by.items()}
        a1 = {r: max(v for (b, a), v in d.items() if a == 1.0) for r, d in by.items()}
        sh = [best[f"shuf{s}:kappa"] for s in range(3)]
        out[name] = {"K": len(experts), "best": best, "best_alpha1": a1, "shuf_mean": float(np.mean(sh)),
                     "kappa_rank_among_kappa_and_shuffled": int(sum(v > best["kappa"] for v in sh)) + 1}
        for r in ("kappa", "kappa_rev", "logkappa"):
            diffs[r].append(best[r] - best["uniform"])
        diffs["shuf_mean"].append(np.mean(sh) - best["uniform"])
        print(f"{name:11s} {len(experts):2d} " + " ".join(f"{100 * x:12.2f}" for x in
              [best["uniform"], best["kappa"], best["kappa_rev"], best["logkappa"], np.mean(sh), max(sh)])
              + f"   {100 * a1['kappa']:7.2f} {100 * a1['uniform']:7.2f}")
    print("\nmean dev-best difference to uniform (pts) over subsets:")
    for r, v in diffs.items():
        print(f"  {r:10s} {100 * np.mean(v):+.2f}  (positive in {sum(x > 0 for x in v)}/{len(v)})")
    (RESULTS / "sweeps" / "subsets_summary.json").write_text(json.dumps({"subsets": out, "diffs_vs_uniform": diffs},
                                                                        indent=1, default=float))


if __name__ == "__main__":
    main()
