"""Weak-tail stress test: can a functionally negligible edit swing the literal rule? (full-FT, dev, budget A rule)

For one target expert j and every linear map with base kappa0 > 1e3 (the attention maps that carry all of the
rule's concentration), floor the expert's singular values at f * sigma_min:  W' = U diag(max(s, f*s_min)) V^T.
kappa drops by ~f while ||W' - W||_F is a tiny fraction of the expert's update.
Measured: relative size of the edit, expert j's own accuracy before/after (all 8 tasks), the literal rule's
mean lambda on expert j and the merged accuracy before/after at fixed (b, alpha).
Writes results/aa/tail_stress.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from condmerge.aa_common import Bank, TaskData, new_model  # noqa: E402


def softmax_coeffs(c, beta):
    if math.isinf(beta):
        m = (c == c.min()).astype(np.float64)
        return m / m.sum()
    z = -beta * (c - c.min())
    w = np.exp(z)
    return w / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="svhn")
    ap.add_argument("--floor", type=float, default=10.0)
    args = ap.parse_args()
    dev = "cuda"
    st = np.load(CACHE / "aa" / "fullft_stats.npz")
    names = list(st["names"])
    j = TASKS8.index(args.expert)
    bank = Bank(TASKS8, dev)
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    kappa = {n: st["kappa_W"][i].copy() for i, n in enumerate(names)}
    edited = {}
    rel_edit, n_edit = [], 0
    for i, n in enumerate(names):
        if st["kappa_W0"][i, 0] <= 1e3:
            continue
        w = (bank.base[n] + bank.deltas[n][j]).double().cpu().numpy()
        w2 = w.reshape(w.shape[0], -1)
        u, s, vt = np.linalg.svd(w2, full_matrices=False)
        s_new = np.maximum(s, args.floor * s[-1])
        w_new = (u * s_new) @ vt
        d = w_new - w2
        rel_edit.append(np.linalg.norm(d) / np.linalg.norm(bank.deltas[n][j].double().cpu().numpy()))
        edited[n] = torch.from_numpy(w_new.reshape(w.shape)).float().to(dev) - bank.base[n]
        kappa_new = kappa[n].copy()
        kappa_new[j] = s_new[0] / s_new[-1]
        kappa[n] = (kappa[n], kappa_new)
        n_edit += 1
    print(f"edited {n_edit} maps of expert {args.expert}; ||edit||/||Delta_j||: median {np.median(rel_edit):.2e} max {np.max(rel_edit):.2e}", flush=True)
    res = {"expert": args.expert, "floor": args.floor, "n_maps_edited": n_edit,
           "rel_edit_median": float(np.median(rel_edit)), "rel_edit_max": float(np.max(rel_edit))}
    # expert j alone, before and after
    one = {n: torch.as_tensor(np.eye(8)[j], dtype=torch.float32, device=dev) for n in bank.linear}
    before = data.evaluate(model, "dev", params=bank.merged(one, 1.0, default=torch.as_tensor(np.eye(8)[j], dtype=torch.float32, device=dev)))
    orig = {n: bank.deltas[n][j].clone() for n in edited}
    for n, d in edited.items():
        bank.deltas[n][j] = d
    after = data.evaluate(model, "dev", params=bank.merged(one, 1.0, default=torch.as_tensor(np.eye(8)[j], dtype=torch.float32, device=dev)))
    res["expert_alone_before"] = before["acc"]
    res["expert_alone_after"] = after["acc"]
    print("expert alone before", {k: round(v, 4) for k, v in before["acc"].items()}, flush=True)
    print("expert alone after ", {k: round(v, 4) for k, v in after["acc"].items()}, flush=True)
    # literal rule before/after the edit at fixed (b, alpha); beta from master's scale convention (median across-expert std)
    def table(which):
        return {n: (kappa[n][which] if isinstance(kappa[n], tuple) else kappa[n]) for n in names}
    sds = [np.std(v) for v in table(0).values() if np.std(v) > 0]
    scale = float(np.median(sds))
    res["rule"] = []
    for b, a in ((4, 1.6), (16, 1.6), (math.inf, 1.6), (4, 1.0)):
        for which, label in ((0, "original"), (1, "tail-edited")):
            for n, d in (orig.items() if which == 0 else edited.items()):
                bank.deltas[n][j] = d
            tab = table(which)
            beta = b / scale if math.isfinite(b) else math.inf
            co = {n: torch.as_tensor(softmax_coeffs(tab[n], beta), dtype=torch.float32, device=dev) for n in names}
            r = data.evaluate(model, "dev", params=bank.merged(co, a))
            lam_j = float(np.mean([float(co[n][j]) for n in names]))
            lam_j_attn = float(np.mean([float(co[n][j]) for n in names if "self_attn" in n]))
            res["rule"].append({"b": b, "alpha": a, "version": label, "avg_acc": r["avg_acc"], "acc": r["acc"],
                                "mean_lambda_j": lam_j, "mean_lambda_j_attn": lam_j_attn})
            print(f"b={b} a={a} {label:12s} avg={r['avg_acc']:.4f} mean lambda_{args.expert}={lam_j:.3f} (attn {lam_j_attn:.3f})", flush=True)
    (RESULTS / "aa" / f"tail_stress_{args.expert}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
