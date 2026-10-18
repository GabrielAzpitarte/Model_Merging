"""Pairwise merges: which expert of a pair 'deserves' more weight, and does conditioning predict it?

Mode ``grid``: for every unordered pair (i, j) of the 8 experts, merge
``W0 + alpha * (lam * D_i + (1 - lam) * D_j)`` (one global lambda for all
parameters) for ``lam`` in 0..1 and evaluate on the dev splits of tasks i and j
only. The favourable coefficient ``lam*`` is read off the curve of the pair's
mean normalised accuracy. Information budget: (D) labelled dev -- this is the
*diagnostic* that defines "deserves more weight", not a method.

Mode ``rules``: the layerwise softmax rules (kappa, reversed, shuffled, ...) on
each pair at a few temperatures, same alpha values.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.clip import Evaluator, load_base_state, load_experts  # noqa: E402
from condmerge.merging import Bank, layerwise_coeffs, mean_lambda_per_expert, summarize_coeffs  # noqa: E402
from condmerge.runlog import append_record, config_key, done_keys  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grid", choices=["grid", "rules"])
    ap.add_argument("--alphas", default="1.0,1.4")
    ap.add_argument("--lams", default=",".join(f"{x:.1f}" for x in np.linspace(0, 1, 11)))
    ap.add_argument("--rules", default="kappa,kappa_rev,logkappa,shuf0:kappa,shuf1:kappa,shuf2:kappa")
    ap.add_argument("--bmults", default="1,4,16,inf")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pairs", default="all", help="'all' or 'i-j,k-l' with task names")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or RESULTS / "pairs" / f"pair_{args.mode}_{args.split}.jsonl")
    done = done_keys(out)
    alphas = [float(a) for a in args.alphas.split(",")]
    t0 = time.time()
    ev = Evaluator(args.device, splits=(args.split,), tasks=TASKS8)
    base = load_base_state()
    bank = Bank.build(base, load_experts(TASKS8), args.device)
    expert_acc = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())
    ref = expert_acc["expert"][args.split]
    pairs = list(itertools.combinations(range(8), 2)) if args.pairs == "all" else [
        tuple(TASKS8.index(x) for x in p.split("-")) for p in args.pairs.split(",")]
    print(f"ready {time.time() - t0:.0f}s, {len(pairs)} pairs, {len(done)} done", flush=True)

    def evaluate(sub: Bank, coeffs: dict, default, alpha: float, tasks: list[str]) -> dict:
        sd = sub.merge(coeffs, alpha, default=default)
        acc = ev.evaluate(sd, args.split, tasks=tasks)
        norm = {t: acc[t] / ref[t] for t in tasks}
        return {"acc": acc, "norm": norm, "mean_norm": float(np.mean(list(norm.values())))}

    if args.mode == "grid":
        lams = [float(x) for x in args.lams.split(",")]
        for i, j in pairs:
            sub = bank.subset([i, j])
            tasks = [TASKS8[i], TASKS8[j]]
            for a in alphas:
                for lam in lams:
                    cfg = {"mode": "grid", "pair": tasks, "alpha": a, "lam": lam, "split": args.split}
                    if config_key(cfg) in done:
                        continue
                    m = evaluate(sub, {}, np.array([lam, 1 - lam]), a, tasks)
                    append_record(out, cfg, m)
                print(f"{tasks} alpha={a} done {time.time() - t0:.0f}s", flush=True)
    else:
        from condmerge.scores import build_rule, load_stats, scale_of
        from condmerge.spectra import linear_map_names

        rows = load_stats()
        names = linear_map_names(base)
        sizes = {p: int(base[p].numel()) for p in names}
        bm = [float(x) for x in args.bmults.split(",")]
        for i, j in pairs:
            sub = bank.subset([i, j])
            tasks = [TASKS8[i], TASKS8[j]]
            for rule in args.rules.split(","):
                full = build_rule(rule, rows, TASKS8, sizes)
                table = {p: v[[i, j]] for p, v in full.items()}
                scale = scale_of(table)
                for b in bm:
                    if math.isinf(b):
                        coeffs = {p: (v == v.min()).astype(float) / (v == v.min()).sum() for p, v in table.items()}
                    else:
                        coeffs = layerwise_coeffs(table, b / scale)
                    lam_bar = mean_lambda_per_expert(coeffs, sizes)
                    for a in alphas:
                        cfg = {"mode": "rules", "pair": tasks, "rule": rule, "b": b, "alpha": a, "split": args.split}
                        if config_key(cfg) in done:
                            continue
                        m = evaluate(sub, coeffs, None, a, tasks)
                        append_record(out, cfg, m, {"lambda_bar_first": float(lam_bar[0]),
                                                    "coeff_summary": summarize_coeffs(coeffs)})
            print(f"{tasks} rules done {time.time() - t0:.0f}s", flush=True)
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
