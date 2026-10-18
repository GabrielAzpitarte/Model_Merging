"""Per-layer marginal preference: d(mean dev CE)/d(lambda_i^p) at the uniform merge (budget D, oracle).

For W_p = W0_p + alpha * sum_i lambda_i^p Delta_i^p the chain rule gives exactly
    dL/dlambda_i^p = alpha * <grad_{W_p} L, Delta_i^p>,
so one backward pass per task at the uniform merge yields every (param, expert)
derivative. L = mean over the 8 tasks of the mean dev cross-entropy at CLIP's
logit scale 100 (labels: this is an oracle/diagnostic quantity, budget D).
The simplex-tangent component is the derivative centred across experts;
``pref = -grad_centered`` > 0 means moving mass to expert i lowers the loss.

Outputs (results/aa/grad_pref/):
  grad_pref_alpha{a}.csv   one row per (param, expert): grad, grad_centered, tied versions, norms, cosine
  grad_pref_pertask.npz    per-task components [alpha][param] -> (n_tasks, K)
Full fp32 (TF32 off), whole dev split.
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
import torch.nn.functional as F  # noqa: E402

from condmerge.aa_common import Bank, TaskData, bias_owner, is_linear_map, layer_index, module_type, new_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1.0,2.4")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=str(RESULTS / "aa" / "grad_pref"))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bank = Bank(TASKS8, dev)
    data = TaskData(TASKS8, (args.split,), dev)
    model = new_model(dev)
    params = dict(model.named_parameters())
    assert set(params) == set(bank.names), set(params) ^ set(bank.names)
    print(f"setup {time.time() - t0:.1f}s, {len(bank.linear)} linear maps", flush=True)
    pertask_all = {}
    summary = {}
    for alpha in [float(a) for a in args.alphas.split(",")]:
        merged = bank.merged({}, alpha)  # uniform
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(merged[n])
        comp = {n: np.zeros((len(TASKS8), bank.k)) for n in bank.names}  # per-task alpha*<G_t, Delta_i>
        gnorm = {n: np.zeros(len(TASKS8)) for n in bank.names}
        task_ce, task_acc = {}, {}
        for ti, t in enumerate(TASKS8):
            model.zero_grad(set_to_none=True)
            x, y = data.data[(t, args.split)]
            n_t = y.numel()
            ce_sum, correct = 0.0, 0
            for i in range(0, n_t, args.batch):
                logits = data.features(model, x[i:i + args.batch]) @ data.heads[t].T
                loss = F.cross_entropy(100.0 * logits, y[i:i + args.batch], reduction="sum") / n_t
                loss.backward()
                ce_sum += float(loss) * n_t
                correct += int((logits.argmax(-1) == y[i:i + args.batch]).sum())
            task_ce[t], task_acc[t] = ce_sum / n_t, correct / n_t
            with torch.no_grad():
                for n, p in params.items():
                    g = p.grad.double()
                    d = bank.deltas[n].double()
                    comp[n][ti] = alpha * torch.einsum("k...,...->k", d, g).cpu().numpy()
                    gnorm[n][ti] = float(g.norm())
            print(f"alpha={alpha} task {t}: ce {task_ce[t]:.4f} acc {task_acc[t]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        # mean-over-tasks objective
        grad = {n: comp[n].mean(0) for n in bank.names}
        rows = []
        for n in bank.names:
            lin = is_linear_map(n, bank.base[n])
            tied = grad[n].copy()
            if lin:
                for b in bank.names:
                    if bias_owner(b) == n:
                        tied = tied + grad[b]
            gc, tc = grad[n] - grad[n].mean(), tied - tied.mean()
            dfro = bank.deltas[n].flatten(1).norm(dim=1).cpu().numpy()
            # ||G_p|| for the mean objective needs the mean gradient; approximate cosine via the components
            for i, e in enumerate(bank.tasks):
                rows.append({"param": n, "layer": layer_index(n), "module": module_type(n), "is_linear_map": int(lin),
                             "expert": e, "alpha": alpha, "grad": grad[n][i], "grad_centered": gc[i],
                             "pref": -gc[i], "grad_tied": tied[i] if lin else "", "grad_tied_centered": tc[i] if lin else "",
                             "pref_tied": -tc[i] if lin else "", "delta_fro": float(dfro[i]),
                             "grad_own_task": comp[n][i, i], "numel": int(bank.base[n].numel())})
        path = out / f"grad_pref_alpha{alpha:g}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        pertask_all[alpha] = comp
        summary[str(alpha)] = {"task_ce": task_ce, "task_acc": task_acc, "avg_acc": float(np.mean(list(task_acc.values()))),
                               "avg_ce": float(np.mean(list(task_ce.values())))}
        print(f"wrote {path} ({len(rows)} rows); avg acc {summary[str(alpha)]['avg_acc']:.4f}", flush=True)
    np.savez_compressed(out / "grad_pref_pertask.npz",
                        **{f"a{a:g}|{n}": v for a, comp in pertask_all.items() for n, v in comp.items()},
                        tasks=np.array(TASKS8))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
