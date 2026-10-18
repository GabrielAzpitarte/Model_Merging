"""LoRA bank: is the literal rule's advantage over shuffled scores a per-map effect or an expert-frequency effect?

The argmin-kappa selection (b = inf) picks one expert per adapted map. Two controls (dev, budget A):
  global_freq      lambda_i = (# maps where expert i is the argmin) / 48, applied to EVERY map (expert-level only)
  count_preserving the same multiset of per-map winners, randomly reassigned across maps (5 seeds)
If global_freq ~ kappa-argmin, the per-map assignment carries nothing; if count_preserving ~ kappa-argmin, the
expert frequencies (not the assignment) explain the gain over the free shuffles.
Writes results/aa/lora/selection_controls.jsonl.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from condmerge.aa_common import TaskData, new_model  # noqa: E402
from condmerge.lora import LoRABank  # noqa: E402

import argparse

ALPHAS = (0.8, 1.0, 1.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alphas", default="0.8,1.0,1.3")
    ap.add_argument("--free", action="store_true", help="also run free random per-map selection (= shuffled kappa at b=inf)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    alphas = tuple(float(a) for a in args.alphas.split(","))
    dev = "cuda"
    rows = list(csv.DictReader((RESULTS / "aa" / "lora" / "lora_stats.csv").open()))
    tab = {(r["param"], r["expert"]): r for r in rows}
    bank = LoRABank(TASKS8, dev)
    names = bank.linear
    K = np.array([[float(tab[(n, t)]["kappa2_W"]) for t in TASKS8] for n in names])
    winners = K.argmin(1)
    counts = np.bincount(winners, minlength=8)
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    out = (RESULTS / "aa" / "lora" / f"selection_controls{args.tag}.jsonl").open("w")

    def run(label, lam_by_map, a, seed=None):
        params = bank.merged({n: torch.as_tensor(lam_by_map[n], dtype=torch.float32, device=dev) for n in names}, a)
        r = data.evaluate(model, "dev", params=params)
        rec = {"label": label, "alpha": a, "seed": seed, "avg_acc": r["avg_acc"], "acc": r["acc"]}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{label:22s} a={a} seed={seed} avg={r['avg_acc']:.4f}", flush=True)

    eye = np.eye(8)
    for a in alphas:
        run("kappa_argmin", {n: eye[w] for n, w in zip(names, winners)}, a)
        run("global_freq", {n: counts / counts.sum() for n in names}, a)
        for s in range(args.seeds):
            perm = np.random.default_rng(100 + s).permutation(len(names))
            run("count_preserving", {n: eye[winners[perm[i]]] for i, n in enumerate(names)}, a, s)
            if args.free:
                rnd = np.random.default_rng(1000 + s).integers(0, 8, len(names))
                run("free_random_selection", {n: eye[rnd[i]] for i, n in enumerate(names)}, a, s)
    print("counts", dict(zip(TASKS8, counts.tolist())))


if __name__ == "__main__":
    main()
