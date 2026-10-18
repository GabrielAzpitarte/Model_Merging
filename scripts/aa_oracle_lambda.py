"""Oracle coefficient fit (budget D: labeled dev): the best allocation inside the kappa rule's own family.

Family (same degrees of freedom as the layerwise condition-number rule):
    W_p = W0_p + alpha * sum_i softmax(theta^p)_i Delta_i^p    for the 73 linear maps p
    biases tied to their weight; LayerNorm / class / position embeddings get the uniform prior.
Variants:
    layerwise_alpha   theta^p per linear map + learned global alpha
    layerwise_a1      theta^p per linear map, alpha fixed at 1 (the literal normalised-average form)
    global_alpha      one theta shared by every linear map + learned alpha (per-expert weights only)
Fitted by Adam on mean-over-tasks dev cross-entropy (logit scale 100), balanced minibatches,
then evaluated on the full dev and the untouched test split. The fitted lambdas are an
UPPER BOUND / reference for any rule in this family, never a deployable method.

Outputs results/aa/oracle_lambda/<variant>_seed<s>.json and <variant>_seed<s>_lambda.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from condmerge.aa_common import Bank, TaskData, new_model  # noqa: E402

ALPHA_GRID = (1.0, 1.6, 2.0, 2.4, 2.8, 3.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="layerwise_alpha,layerwise_a1,global_alpha")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--per-task", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lr-alpha", type=float, default=0.02)
    ap.add_argument("--out", default=str(RESULTS / "aa" / "oracle_lambda"))
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bank = Bank(TASKS8, dev)
    data = TaskData(TASKS8, ("dev", "test"), dev)
    model = new_model(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    K = bank.k

    # uniform-lambda alpha grid on dev (the Task-Arithmetic reference in this pipeline)
    grid_path = out / "uniform_alpha_grid.json"
    if grid_path.exists():
        grid = json.loads(grid_path.read_text())
    else:
        grid = {}
        for a in ALPHA_GRID:
            r = data.evaluate(model, "dev", params=bank.merged({}, a))
            grid[str(a)] = {"avg_acc": r["avg_acc"], "acc": r["acc"], "avg_ce": r["avg_ce"]}
            print(f"uniform alpha={a}: dev avg acc {r['avg_acc']:.4f}", flush=True)
        grid_path.write_text(json.dumps(grid, indent=1))
    a_best = float(max(grid, key=lambda a: grid[a]["avg_acc"]))
    print(f"best uniform alpha on dev = {a_best} ({grid[str(a_best)]['avg_acc']:.4f}); setup {time.time() - t0:.0f}s", flush=True)

    for variant in args.variants.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            tag = f"{variant}_seed{seed}"
            if (out / f"{tag}.json").exists():
                print("skip", tag)
                continue
            torch.manual_seed(seed)
            g = torch.Generator(device=dev).manual_seed(seed)
            if variant.startswith("layerwise"):
                theta = {n: torch.zeros(K, device=dev, requires_grad=True) for n in bank.linear}
            else:
                shared = torch.zeros(K, device=dev, requires_grad=True)
                theta = {n: shared for n in bank.linear}
            learn_alpha = variant.endswith("alpha")
            log_alpha = torch.tensor(math.log(a_best if learn_alpha else 1.0), device=dev, requires_grad=learn_alpha)
            leaves = list({id(v): v for v in theta.values()}.values())
            groups = [{"params": leaves, "lr": args.lr}]
            if learn_alpha:
                groups.append({"params": [log_alpha], "lr": args.lr_alpha})
            opt = torch.optim.Adam(groups)
            curve = []
            for step in range(args.steps):
                lam = {n: torch.softmax(theta[n], 0) for n in bank.linear}
                params = bank.merged(lam, log_alpha.exp())
                loss = 0.0
                for t in TASKS8:
                    x, y = data.data[(t, "dev")]
                    idx = torch.randint(0, y.numel(), (args.per_task,), device=dev, generator=g)
                    logits = data.features(model, x[idx], params, amp="bf16") @ data.heads[t].T
                    loss = loss + F.cross_entropy(100.0 * logits, y[idx]) / len(TASKS8)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % 25 == 0 or step == args.steps - 1:
                    curve.append({"step": step, "loss": float(loss), "alpha": float(log_alpha.exp())})
                    print(f"{tag} step {step} loss {float(loss):.4f} alpha {float(log_alpha.exp()):.3f} ({time.time() - t0:.0f}s)", flush=True)
            with torch.no_grad():
                lam = {n: torch.softmax(theta[n], 0) for n in bank.linear}
                params = bank.merged(lam, log_alpha.exp())
                rd = data.evaluate(model, "dev", params=params)
                rt = data.evaluate(model, "test", params=params)
            lam_np = {n: lam[n].cpu().numpy().tolist() for n in bank.linear}
            rec = {"variant": variant, "seed": seed, "alpha": float(log_alpha.exp()), "steps": args.steps,
                   "per_task_batch": args.per_task, "lr": args.lr, "budget": "D (labeled dev; oracle)",
                   "dev": rd, "test": rt, "curve": curve, "lambda": lam_np, "experts": bank.tasks,
                   "init_alpha": a_best if learn_alpha else 1.0}
            (out / f"{tag}.json").write_text(json.dumps(rec, indent=1))
            with (out / f"{tag}_lambda.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["param", "expert", "lambda"])
                for n in bank.linear:
                    for i, e in enumerate(bank.tasks):
                        w.writerow([n, e, lam_np[n][i]])
            print(f"{tag}: alpha {rec['alpha']:.3f} dev {rd['avg_acc']:.4f} test {rt['avg_acc']:.4f}", flush=True)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
