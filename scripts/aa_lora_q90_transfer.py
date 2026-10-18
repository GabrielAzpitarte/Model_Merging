"""Transfer check of master's dev-best full-FT rule `logkappa_q90_rev` to four 8-task LoRA banks (dev).

Rule: per adapted map, lambda ~ softmax(+beta * log(s_1 / s_ceil(0.9 q)) of W0 + B_i A_i)  (higher bulk anisotropy ->
more weight); beta = b / (median across maps of the across-expert std of the score), master's convention.
Configs fixed a priori: b in {1, 4}, alpha in {0.8, 1.0}; uniform at the same alpha; 5 matched-config shuffles
(scores permuted across experts within each map, seeds 0-4).
Caveat: rank-16 adapters barely move the bulk singular values, so the score is base-dominated for LoRA.
Writes results/aa/lora/q90_transfer.jsonl.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from aa_lora_replicate_banks import load_bank  # noqa: E402
from condmerge.aa_common import TaskData, load_base_unprefixed, new_model  # noqa: E402


def main() -> None:
    dev = "cuda"
    t0 = time.time()
    base = load_base_unprefixed()
    base_dev = {k: v.to(dev) for k, v in base.items()}
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    f = (RESULTS / "aa" / "lora" / "q90_transfer.jsonl").open("a")
    for bname in ("knots_release", "lora_c5a", "lora_c5a_seed1", "lora"):
        bank = load_bank(bname)
        names = sorted(bank[TASKS8[0]])
        deltas = {n: torch.stack([bank[t][n] for t in TASKS8]).to(dev) for n in names}
        S = np.zeros((len(names), 8))
        for pi, n in enumerate(names):
            w0 = base[n].double().numpy()
            for i, t in enumerate(TASKS8):
                s = np.linalg.svd(w0 + bank[t][n].double().numpy(), compute_uv=False)
                S[pi, i] = np.log(s[0] / s[int(math.ceil(0.9 * len(s))) - 1])
        sc = float(np.median([np.std(S[p]) for p in range(len(names)) if np.std(S[p]) > 0]))

        def ev(label, lam_rows, a, b=None, seed=None):
            params = dict(base_dev)
            for pi, n in enumerate(names):
                params[n] = base_dev[n] + a * torch.tensordot(torch.as_tensor(lam_rows[pi], dtype=torch.float32, device=dev), deltas[n], dims=1)
            r = data.evaluate(model, "dev", params=params)
            f.write(json.dumps({"bank": bname, "label": label, "b": b, "alpha": a, "seed": seed, "avg_acc": r["avg_acc"]}) + "\n")
            f.flush()
            return r["avg_acc"]

        def soft(scores, b):
            z = (b / sc) * (scores - scores.max(1, keepdims=True))   # + sign: higher q90 ratio -> more weight
            w = np.exp(z)
            return w / w.sum(1, keepdims=True)

        for a in (0.8, 1.0):
            u = ev("uniform", np.full((len(names), 8), 1 / 8), a)
            for b in (1, 4):
                r = ev("logkappa_q90_rev", soft(S, b), a, b)
                sh = [ev("shuffled", soft(np.stack([S[p][np.random.default_rng(s).permutation(8)] for p in range(len(names))]), b), a, b, s)
                      for s in range(5)]
                print(f"{bname} a={a} b={b}: uniform {u:.4f} rule {r:.4f} (rule-uniform {100 * (r - u):+.2f}) | shuffles {np.mean(sh):.4f} "
                      f"[{min(sh):.4f}, {max(sh):.4f}] ({time.time() - t0:.0f}s)", flush=True)
        del deltas
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
