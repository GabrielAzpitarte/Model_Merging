"""Evaluate FROZEN configurations on the untouched test split.

A frozen config is chosen on dev *before* this script runs: either the best
(b, alpha) over the shared grid for a rule ("tuned"), or a pre-declared fixed
point (e.g. alpha=1). Configs are read from a JSON list written by the
analysis step, and every test number is appended to results/freeze/test.jsonl
together with the dev number that selected it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import ARTIFACTS, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.clip import Evaluator, load_base_state, load_experts, summarize  # noqa: E402
from condmerge.merging import Bank, layerwise_coeffs  # noqa: E402
from condmerge.runlog import append_record, config_key, done_keys  # noqa: E402
from condmerge.scores import build_rule, load_stats, scale_of  # noqa: E402
from condmerge.spectra import linear_map_names  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_sweep import argmin_coeffs, global_norm_coeffs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="JSON list of {rule, b, alpha, experts?, tag}")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(RESULTS / "freeze" / "test.jsonl"))
    args = ap.parse_args()
    cfgs = json.loads(Path(args.configs).read_text())
    out = Path(args.out)
    done = done_keys(out)
    t0 = time.time()
    ev = Evaluator(args.device, splits=(args.split,), tasks=TASKS8)
    base = load_base_state()
    bank = Bank.build(base, load_experts(TASKS8), args.device)
    ref_all = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"][args.split]
    rows = None
    names = linear_map_names(base)
    sizes = {p: int(base[p].numel()) for p in names}
    for c in cfgs:
        experts = c.get("experts", TASKS8)
        idx = [TASKS8.index(e) for e in experts]
        sub = bank if len(idx) == 8 else bank.subset(idx)
        cfg = {"rule": c["rule"], "b": c["b"], "alpha": c["alpha"], "experts": experts, "split": args.split,
               "bias_policy": c.get("bias_policy", "tied"), "tag": c.get("tag", "")}
        if config_key(cfg) in done:
            continue
        default = None
        coeffs: dict = {}
        if c["rule"] == "uniform" or c["b"] == 0:
            pass
        elif c["rule"] == "metagpt":
            default = global_norm_coeffs(sub, 2.0)
        elif c["rule"].startswith("single:"):
            e = c["rule"].split(":", 1)[1]
            default = np.array([1.0 if x == e else 0.0 for x in experts])
        else:
            rows = rows if rows is not None else load_stats()
            table = {p: v[idx] for p, v in build_rule(c["rule"], rows, TASKS8, sizes).items()}
            b = float(c["b"])
            coeffs = argmin_coeffs(table) if math.isinf(b) else layerwise_coeffs(table, b / scale_of(table))
        sd = sub.merge(coeffs, float(c["alpha"]), default=default, bias_policy=cfg["bias_policy"])
        tasks = experts if c.get("eval_on_subset", True) else TASKS8
        ev.load(sd)
        corr = {t: ev.correct(t, args.split) for t in tasks}
        acc = {t: float(v.mean()) for t, v in corr.items()}
        cdir = ARTIFACTS / "runs" / "freeze_correct"
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cdir / f"{config_key(cfg)}.npz", **corr)
        ref = {t: ref_all[t] for t in tasks}
        m = summarize(acc, ref)
        append_record(out, cfg, {**m, "acc": acc}, {"dev_metric": c.get("dev_metric")})
        print(f"{c.get('tag', ''):30s} {c['rule']:22s} b={c['b']} a={c['alpha']} test norm={100 * m['mean_norm']:.2f} "
              f"acc={100 * m['mean_acc']:.2f} min={100 * m['min_norm']:.1f}  (dev {c.get('dev_metric')})", flush=True)
    print(f"ALL DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
