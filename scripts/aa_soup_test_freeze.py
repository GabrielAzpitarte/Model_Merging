"""Frozen shared-target soup rules on the untouched TEST split (configurations fixed before any test look).

Frozen rule set (chosen from dev / a priori, see docs/aa/AA_SIDE_REPORT.md):
  uniform; kappa (literal) at b=1 and b=4; mp_invvar (tuning-free lambda ~ median-sv(Delta)^-2) and its global
  version (total Gavish-Donoho noise energy);
  dsrank at b=4; references: oracle_invvar / cleanest / noisiest (controlled noise),
  oracle_best_single (by DEV accuracy) for the real variants.
Pools: A-strong (noise 1,4,8,16 x rms), A-strong-norm-matched, B (real same-task LoRA variants).
Noise realisations are seeded exactly as in aa_soup_shared_target.py, so the copies are identical.
Writes results/aa/soup/test_freeze.jsonl.
"""

from __future__ import annotations

import argparse
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

from aa_soup_shared_target import (LORA_VARIANTS, MOLOHIO, VIT20_NAME, merged_params, rule_tables, scale_of,  # noqa: E402
                                   softmax_coeffs, stats_for)
from condmerge.aa_common import TaskData, is_linear_map, load_base_unprefixed, load_expert_unprefixed, new_model  # noqa: E402

FROZEN = (("kappa", 1.0), ("kappa", 4.0), ("dsrank", 4.0))


def frozen_jobs(st, names, K, extra_oracles):
    tables = rule_tables(st)
    uni = np.full(K, 1.0 / K)
    jobs = [("uniform", 0.0, {p: uni for p in names})]
    for r, b in FROZEN:
        beta = b / scale_of(tables[r])
        jobs.append((r, b, {p: softmax_coeffs(tables[r][p], beta) for p in names}))
    jobs.append(("mp_invvar", 1.0, {p: softmax_coeffs(tables["dmedsv"][p], 1.0) for p in names}))
    tot = np.sum(np.stack([st["sighat"][p] ** 2 * st["size"][p] for p in names]), 0)
    jobs.append(("global:mp_invvar", 1.0, {p: (1 / tot) / np.sum(1 / tot) for p in names}))
    for r, lam in extra_oracles.items():
        jobs.append((r, math.nan, {p: lam for p in names}))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="A_strong,A_strong_normmatched,B")
    ap.add_argument("--out", default=str(RESULTS / "aa" / "soup" / "test_freeze.jsonl"))
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    base = {k: v.to(dev) for k, v in load_base_unprefixed().items()}
    data = TaskData(TASKS8, ("dev", "test"), dev)
    model = new_model(dev)
    linear = [k for k in base if is_linear_map(k, base[k])]
    f = open(args.out, "a")
    for pool in args.pools.split(","):
        for e in TASKS8:
            if pool.startswith("A"):
                levels = np.array([1.0, 4.0, 8.0, 16.0])
                ex = load_expert_unprefixed(e)
                g = torch.Generator(device=dev).manual_seed(TASKS8.index(e))
                deltas = {}
                for p in linear:
                    d = ex[p].to(dev) - base[p]
                    rms = d.pow(2).mean().sqrt()
                    noise = torch.randn((len(levels),) + d.shape, generator=g, device=dev)
                    deltas[p] = d.unsqueeze(0) + torch.as_tensor(levels, dtype=torch.float32, device=dev).view(-1, *[1] * d.ndim) * rms * noise
                    if pool.endswith("normmatched"):
                        nrm = deltas[p].flatten(1).norm(dim=1).view(-1, *[1] * d.ndim)
                        deltas[p] = deltas[p] * (d.norm() / nrm)
                extra = {k: ex[k].to(dev) for k in base if k not in deltas}
                inv = (1 / levels ** 2) / np.sum(1 / levels ** 2)
                oracles = {"oracle_invvar": inv, "cleanest_only": np.eye(4)[0], "noisiest_only": np.eye(4)[-1]}
                names, K = linear, 4
            else:
                avail = [v for v in LORA_VARIANTS if (MOLOHIO / v / f"{VIT20_NAME[e]}.npz").exists()]
                facs = [np.load(MOLOHIO / v / f"{VIT20_NAME[e]}.npz") for v in avail]
                names = sorted({k[2:] + ".weight" for k in facs[0].files if k.startswith("A_")})
                deltas = {p: torch.stack([torch.from_numpy(z[f"B_{p[:-7]}"] @ z[f"A_{p[:-7]}"]).float() for z in facs]).to(dev)
                          for p in names}
                extra = None
                K = len(avail)
                q = [data.evaluate(model, "dev", params=merged_params(base, deltas, {}, np.eye(K)[j], 1.0, dev), tasks=[e])["acc"][e]
                     for j in range(K)]
                oracles = {"oracle_best_single_by_dev": np.eye(K)[int(np.argmax(q))]}
            st = stats_for(base, deltas, names)
            for rule, b, co in frozen_jobs(st, names, K, oracles):
                params = merged_params(base, deltas, co, np.full(K, 1.0 / K), 1.0, dev, extra)
                res = data.evaluate(model, "test", params=params, tasks=[e])
                f.write(json.dumps({"pool": pool, "task": e, "rule": rule, "b": b, "split": "test",
                                    "acc": res["acc"][e], "ce": res["ce"][e],
                                    "mean_lambda": np.mean(np.stack(list(co.values())), 0).tolist()}) + "\n")
                f.flush()
            print(f"{pool} {e} done ({time.time() - t0:.0f}s)", flush=True)
            del deltas
            torch.cuda.empty_cache()
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
