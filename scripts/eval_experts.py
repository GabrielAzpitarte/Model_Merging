"""Reference accuracies: base zero-shot and every expert on every task (dev + test).

Writes ``results/experts/expert_accuracy.json`` with
``{"zeroshot": {split: {task: acc}}, "cross": {split: {expert: {task: acc}}},
   "expert": {split: {task: own-expert acc}}, "fp32_check": {...}}``.
The diagonal ``expert`` entries are the normalisers for normalised accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, git_commit, offline_env  # noqa: E402

offline_env()

from condmerge.clip import Evaluator, load_base_state, load_expert_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--splits", default="dev,test")
    args = ap.parse_args()
    splits = args.splits.split(",")
    t0 = time.time()
    ev = Evaluator(args.device, splits=splits)
    print(f"evaluator ready {time.time() - t0:.1f}s", flush=True)
    out = {"zeroshot": {}, "cross": {s: {} for s in splits}, "expert": {s: {} for s in splits},
           "git": git_commit(), "precision": "fp16 autocast"}
    base = load_base_state()
    for s in splits:
        out["zeroshot"][s] = ev.evaluate(base, s)
        print("zeroshot", s, {k: round(v, 4) for k, v in out["zeroshot"][s].items()}, flush=True)
    for e in TASKS8:
        sd = load_expert_state(e)
        ev.load(sd)
        for s in splits:
            out["cross"][s][e] = ev.evaluate(None, s)
            out["expert"][s][e] = out["cross"][s][e][e]
        print("expert", e, {s: round(out["expert"][s][e], 4) for s in splits}, flush=True)
    # fp32 check on dev for the own-task accuracy (precision sensitivity of the protocol)
    ev.amp = False
    out["fp32_check"] = {}
    for e in TASKS8:
        ev.load(load_expert_state(e))
        out["fp32_check"][e] = ev.accuracy(e, splits[0])
    ev.amp = True
    for s in splits:
        print(s, "mean expert", sum(out["expert"][s].values()) / 8,
              "mean zeroshot", sum(out["zeroshot"][s].values()) / 8, flush=True)
    d = RESULTS / "experts"
    d.mkdir(parents=True, exist_ok=True)
    (d / "expert_accuracy.json").write_text(json.dumps(out, indent=1))
    print("wrote", d / "expert_accuracy.json", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
