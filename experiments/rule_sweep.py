"""Sweep coefficient rules over (beta, alpha) on the full-FT 8-expert bank.

Every rule gets the same grid: ``beta = b / scale(rule)`` for
``b in BETA_MULTS`` (``scale`` = median across linear maps of the
across-expert std of the rule's scores, so ``b`` is the softmax-logit spread at
the median layer) and ``alpha in ALPHAS``. ``b = inf`` is the argmin-selection
limit (one expert per linear map). Records go to an append-only JSONL keyed by
the config hash, so reruns resume.

Information budget of every rule here: (A) weights only. The dev split is used
to *select* (beta, alpha); test is touched only by ``freeze_eval.py``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import torch  # noqa: E402

from condmerge.clip import Evaluator, load_base_state, load_experts, summarize  # noqa: E402
from condmerge.merging import (  # noqa: E402
    Bank,
    layerwise_coeffs,
    mean_lambda_per_expert,
    summarize_coeffs,
)
from condmerge.runlog import append_record, done_keys  # noqa: E402
from condmerge.scores import build_rule, load_stats, scale_of  # noqa: E402
from condmerge.spectra import linear_map_names  # noqa: E402

BETA_MULTS = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, math.inf)
ALPHAS = (1.0, 1.6, 2.0, 2.4, 2.8, 3.2)


def argmin_coeffs(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for p, v in table.items():
        m = (v == np.min(v)).astype(np.float64)
        out[p] = m / m.sum()
    return out


def global_norm_coeffs(bank: Bank, power: float = 2.0) -> np.ndarray:
    """MetaGPT-style: lambda_i proportional to ||Delta_i||^power over all parameters."""
    sq = torch.zeros(bank.k, dtype=torch.float64, device=next(iter(bank.deltas.values())).device)
    for d in bank.deltas.values():
        sq += d.double().pow(2).flatten(1).sum(1)
    w = sq.sqrt().pow(power).cpu().numpy()
    return w / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="comma list of rule names (see scores.build_rule), "
                    "or the special names uniform, metagpt")
    ap.add_argument("--betas", default=",".join(str(b) for b in BETA_MULTS))
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--experts", default=",".join(TASKS8))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--bias-policy", default="tied")
    ap.add_argument("--abs-beta", action="store_true",
                    help="interpret --betas as absolute beta values (derived temperatures) instead of b multipliers")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(RESULTS / "sweeps" / "rule_sweep_dev.jsonl"))
    args = ap.parse_args()
    experts = args.experts.split(",")
    betas = [float(b) for b in args.betas.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    out = Path(args.out)
    done = done_keys(out)
    t0 = time.time()
    ev = Evaluator(args.device, splits=(args.split,), tasks=experts)
    base = load_base_state()
    bank = Bank.build(base, load_experts(experts), args.device)
    expert_acc = __import__("json").loads((RESULTS / "experts" / "expert_accuracy.json").read_text())
    ref = {t: expert_acc["expert"][args.split][t] for t in experts}
    rows = None  # loaded on first use (uniform/metagpt do not need spectra)
    names = linear_map_names(base)
    sizes = {p: int(base[p].numel()) for p in names}
    print(f"ready {time.time() - t0:.0f}s; {len(done)} records done", flush=True)

    def run(rule: str, b: float, beta: float, coeffs: dict, alpha: float, extra_cfg: dict | None = None,
            default=None) -> None:
        cfg = {"rule": rule, "b": b, "alpha": alpha, "experts": experts, "split": args.split,
               "bias_policy": args.bias_policy, **(extra_cfg or {})}
        from condmerge.runlog import config_key

        if config_key(cfg) in done:
            return
        t1 = time.time()
        sd = bank.merge(coeffs, alpha, default=default, bias_policy=args.bias_policy)
        acc = ev.evaluate(sd, args.split)
        m = summarize(acc, ref)
        lam_bar = mean_lambda_per_expert(coeffs, sizes) if coeffs else (default if default is not None else np.full(len(experts), 1 / len(experts)))
        rec = append_record(out, cfg, {**m, "acc": acc, "norm": {t: acc[t] / ref[t] for t in acc}},
                            {"beta": beta, "coeff_summary": summarize_coeffs(coeffs) if coeffs else {},
                             "lambda_bar": {e: float(v) for e, v in zip(experts, lam_bar)},
                             "seconds": time.time() - t1})
        done.add(rec["key"])
        print(f"{rule:22s} b={b:<6g} a={alpha:<4g} norm={m['mean_norm']:.4f} acc={m['mean_acc']:.4f} "
              f"min={m['min_norm']:.3f} neff={rec['coeff_summary'].get('n_eff_mean', float('nan')):.2f} "
              f"[{time.time() - t1:.1f}s]", flush=True)

    for rule in args.rules.split(","):
        if rule == "uniform":
            for a in alphas:
                run("uniform", 0.0, 0.0, {}, a)
            continue
        if rule.startswith("metagpt"):
            lam = global_norm_coeffs(bank, 2.0)
            for a in alphas:
                run("metagpt", 0.0, 0.0, {}, a, default=lam)
            continue
        idx = [TASKS8.index(e) for e in experts]
        rows = rows if rows is not None else load_stats()
        table = {p: v[idx] for p, v in build_rule(rule, rows, TASKS8, sizes).items()}
        scale = scale_of(table)
        for b in betas:
            if b == 0:
                continue  # identical to 'uniform'
            if math.isinf(b):
                coeffs, beta = argmin_coeffs(table), math.inf
            else:
                beta = b if args.abs_beta else b / scale
                coeffs = layerwise_coeffs(table, beta)
            for a in alphas:
                run(rule, b, beta, coeffs, a, extra_cfg={"scale": scale, **({"abs_beta": True} if args.abs_beta else {})})
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
