"""Paired, task-stratified bootstrap of test differences between frozen configurations.

Reads results/freeze/test.jsonl and the per-example correctness saved by freeze_eval.py.
For every config, reports mean normalised test accuracy and a 95 % CI of its difference to a
reference config (default: uniform, dev-tuned alpha), resampling test images within each task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import ARTIFACTS, RESULTS  # noqa: E402
from condmerge.runlog import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RESULTS / "freeze" / "test.jsonl"))
    ap.add_argument("--ref-tag", default="uniform|tuned")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS / "freeze" / "test_comparison.json"))
    args = ap.parse_args()
    recs = read_jsonl(Path(args.file))
    exp = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"]["test"]
    cdir = ARTIFACTS / "runs" / "freeze_correct"
    by_tag = {r["cfg"]["tag"]: r for r in recs}
    ref = by_tag[args.ref_tag]
    ref_c = dict(np.load(cdir / f"{ref['key']}.npz"))
    tasks = list(ref_c)
    rng = np.random.default_rng(0)
    idx = {t: rng.integers(0, len(ref_c[t]), size=(args.n_boot, len(ref_c[t]))) for t in tasks}
    out = {}
    for tag, r in sorted(by_tag.items()):
        c = dict(np.load(cdir / f"{r['key']}.npz"))
        if set(c) != set(tasks):
            continue
        diff_t = np.stack([(c[t][idx[t]].mean(1) - ref_c[t][idx[t]].mean(1)) / exp[t] for t in tasks])  # (T, B)
        diff = diff_t.mean(0)
        point = float(np.mean([(c[t].mean() - ref_c[t].mean()) / exp[t] for t in tasks]))
        out[tag] = {"test_mean_norm": r["metrics"]["mean_norm"], "test_mean_acc": r["metrics"]["mean_acc"],
                    "test_min_norm": r["metrics"]["min_norm"], "dev_metric": r.get("dev_metric"),
                    "diff_vs_ref": point, "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
        print(f"{tag:34s} test norm {100 * r['metrics']['mean_norm']:6.2f} acc {100 * r['metrics']['mean_acc']:6.2f} "
              f"min {100 * r['metrics']['min_norm']:5.1f} | diff vs ref {100 * point:+6.2f} "
              f"[{100 * out[tag]['ci95'][0]:+.2f}, {100 * out[tag]['ci95'][1]:+.2f}]  (dev {100 * (r.get('dev_metric') or float('nan')):.2f})")
    Path(args.out).write_text(json.dumps({"ref": args.ref_tag, "results": out}, indent=1))


if __name__ == "__main__":
    main()
