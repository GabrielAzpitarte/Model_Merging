"""Freeze dev-selected configurations for the one-shot test evaluation.

For every rule found in the sweep shards: the dev-best (b, alpha) over the shared grid
("tuned") and the dev-best b at alpha = 1 ("alpha1", the literal normalised average).
Also the best single expert (selected on dev from the 8x8 cross matrix).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_sweep import load, table  # noqa: E402

from condmerge.paths import RESULTS, TASKS8  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RESULTS / "sweeps" / "rule_sweep_dev.jsonl"))
    ap.add_argument("--rules", default=None, help="comma list; default all rules in the shards")
    ap.add_argument("--out", default=str(RESULTS / "freeze" / "frozen_configs.json"))
    args = ap.parse_args()
    by = table(load(Path(args.file)))
    rules = args.rules.split(",") if args.rules else sorted(by)
    cfgs = []
    for rule in rules:
        d = by[rule]
        best = max(d.values(), key=lambda r: r["metrics"]["mean_norm"])
        cfgs.append({"rule": rule, "b": best["cfg"]["b"], "alpha": best["cfg"]["alpha"], "tag": f"{rule}|tuned",
                     "dev_metric": best["metrics"]["mean_norm"]})
        a1 = [r for (b, a), r in d.items() if a == 1.0]
        if a1:
            b1 = max(a1, key=lambda r: r["metrics"]["mean_norm"])
            cfgs.append({"rule": rule, "b": b1["cfg"]["b"], "alpha": 1.0, "tag": f"{rule}|alpha1",
                         "dev_metric": b1["metrics"]["mean_norm"]})
    exp = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())
    ref, cross = exp["expert"]["dev"], exp["cross"]["dev"]
    single = {e: float(np.mean([cross[e][t] / ref[t] for t in TASKS8])) for e in TASKS8}
    e_best = max(single, key=single.get)
    cfgs.append({"rule": f"single:{e_best}", "b": 0, "alpha": 1.0, "tag": "best_single", "dev_metric": single[e_best]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cfgs, indent=1))
    print(f"{len(cfgs)} frozen configs -> {args.out}; best single on dev = {e_best} ({100 * single[e_best]:.2f})")


if __name__ == "__main__":
    main()
