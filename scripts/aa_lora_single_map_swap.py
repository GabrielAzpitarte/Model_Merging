"""LoRA: per-map marginal value of handing a whole map to one expert (diagnostic, budget D).

Baseline: uniform lambda on every adapted map at alpha (default 1.0). For each map p and expert i, set
lambda^p = one-hot(i) (other maps stay uniform) and record dev mean accuracy and mean CE.
Then: within-map Spearman between the swap value and kappa2(W0 + Delta_i) / sigma_min; where the argmin-kappa
choice ranks among the 8 options per map (best = 1, worst = 8), versus the uniform-random expectation 4.5.
Writes results/aa/lora/single_map_swap.jsonl and single_map_swap_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

torch.set_num_threads(2)

from condmerge.aa_common import TaskData, new_model  # noqa: E402
from condmerge.lora import LoRABank  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    bank = LoRABank(TASKS8, dev)
    names = bank.linear
    rows = list(csv.DictReader((RESULTS / "aa" / "lora" / "lora_stats.csv").open()))
    tab = {(r["param"], r["expert"]): r for r in rows}
    K = np.array([[float(tab[(n, t)]["kappa2_W"]) for t in TASKS8] for n in names])
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    uni = torch.full((8,), 1 / 8, device=dev)
    out = RESULTS / "aa" / "lora" / "single_map_swap.jsonl"
    done = {(json.loads(l)["param"], json.loads(l)["expert"]) for l in out.read_text().splitlines()} if out.exists() else set()
    f = out.open("a")
    if ("__base__", "uniform") not in done:
        r = data.evaluate(model, "dev", params=bank.merged({}, args.alpha))
        f.write(json.dumps({"param": "__base__", "expert": "uniform", "alpha": args.alpha, "avg_acc": r["avg_acc"], "avg_ce": r["avg_ce"]}) + "\n")
        f.flush()
    eye = torch.eye(8, device=dev)
    for pi, n in enumerate(names):
        for i, t in enumerate(TASKS8):
            if (n, t) in done:
                continue
            r = data.evaluate(model, "dev", params=bank.merged({n: eye[i]}, args.alpha, default=uni))
            f.write(json.dumps({"param": n, "expert": t, "alpha": args.alpha, "avg_acc": r["avg_acc"], "avg_ce": r["avg_ce"]}) + "\n")
            f.flush()
        print(f"[{pi + 1}/{len(names)}] {n} ({time.time() - t0:.0f}s)", flush=True)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    base = next(r for r in recs if r["param"] == "__base__")
    V = np.array([[next(r["avg_acc"] for r in recs if r["param"] == n and r["expert"] == t) for t in TASKS8] for n in names])
    gain = V - base["avg_acc"]
    rho_k = float(np.mean([spearmanr(-K[p], gain[p]).statistic for p in range(len(names))]))
    ranks = np.array([rankdata(-gain[p])[K[p].argmin()] for p in range(len(names))])  # 1 = best swap
    summ = {"alpha": args.alpha, "base_avg_acc": base["avg_acc"],
            "mean_within_map_spearman(-kappa, swap_gain)": rho_k,
            "argmin_kappa_mean_rank (1=best, 8=worst, random 4.5)": float(ranks.mean()),
            "frac_argmin_kappa_is_worst": float(np.mean(ranks == 8)), "frac_argmin_kappa_is_best": float(np.mean(ranks == 1)),
            "mean_gain_of_argmin_kappa": float(np.mean(gain[np.arange(len(names)), K.argmin(1)])),
            "mean_gain_random_choice": float(gain.mean()), "mean_gain_best_choice": float(gain.max(1).mean()),
            "mean_gain_worst_choice": float(gain.min(1).mean())}
    (RESULTS / "aa" / "lora" / "single_map_swap_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
