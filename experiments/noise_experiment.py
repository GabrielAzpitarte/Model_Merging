"""Controlled-noise test of the 'conditioning estimates uncertainty' hypothesis (review sec. 4.1, row 2).

Half of the experts (seeded choice) get isotropic Gaussian noise added to every linear-map
update: ``D~_i[p] = D_i[p] + rho * ||D_i[p]||_F * G / ||G||_F``. Biases/LayerNorm untouched.
The corrupted experts are, by construction, noisier estimates of their own target.
Questions:
  1. identification: within each map, do condition scores separate noisy from clean experts,
     and in WHICH direction (does the literal rule give the noisy experts less weight)?
  2. merging: does the literal rule (or its reverse / update-domain variants) beat uniform
     on the corrupted bank, at a matched (b, alpha) grid, relative to an oracle that knows S?
Budget: rules (A); grid selection on dev (D). Spectra: CPU float64.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.clip import Evaluator, load_base_state, load_experts, summarize  # noqa: E402
from condmerge.merging import Bank, layerwise_coeffs, summarize_coeffs  # noqa: E402
from condmerge.runlog import append_record, config_key, done_keys  # noqa: E402
from condmerge.scores import build_rule, load_stats, scale_of  # noqa: E402
from condmerge.spectra import as_matrix, linear_map_names, spectral_stats  # noqa: E402
from rule_sweep import argmin_coeffs  # noqa: E402

RULES = ("kappa", "kappa_rev", "logkappa", "logkappa_rev", "dlogkappa_q90", "dlogkappa_q90_rev",
         "dentropy_rank", "dentropy_rank_rev", "dnorm", "dnorm_rev")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--n-noisy", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bmults", default="0.5,2,8,inf")
    ap.add_argument("--alphas", default="1.0,1.6,2.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    noisy = sorted(rng.choice(8, args.n_noisy, replace=False).tolist())
    noisy_names = [TASKS8[i] for i in noisy]
    tag = f"rho{args.rho:g}_n{args.n_noisy}_s{args.seed}"
    out = RESULTS / "noise" / f"noise_{tag}.jsonl"
    done = done_keys(out)
    print(f"[{tag}] noisy experts: {noisy_names}", flush=True)
    base = load_base_state()
    experts = load_experts(TASKS8)
    names = linear_map_names(base)
    sizes = {p: int(base[p].numel()) for p in names}
    gen = torch.Generator().manual_seed(1000 + args.seed)
    rows = [r for r in load_stats(include_supported=False) if r["kind"] == "base" or r["expert"] not in noisy_names]
    for e in noisy_names:
        for p in names:
            w0 = as_matrix(p, base[p]).double()
            d = as_matrix(p, experts[e][p]).double() - w0
            g = torch.randn(d.shape, generator=gen, dtype=torch.float64)
            d_noisy = d + args.rho * d.norm() * g / g.norm()
            experts[e][p] = (w0 + d_noisy).reshape(base[p].shape).float()
            s_full = np.linalg.svd((w0 + d_noisy).numpy(), compute_uv=False)
            s_upd = np.linalg.svd(d_noisy.numpy(), compute_uv=False)
            rows.append({"param": p, "kind": "full", "expert": e, **spectral_stats(s_full, tuple(d.shape))})
            rows.append({"param": p, "kind": "update", "expert": e, **spectral_stats(s_upd, tuple(d.shape))})
        print(f"  corrupted {e} + spectra {time.time() - t0:.0f}s", flush=True)
    ev = Evaluator(args.device, splits=(args.split,), tasks=TASKS8)
    ref = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"][args.split]
    bank = Bank.build(base, experts, args.device)
    # 1. standalone accuracy of each corrupted expert on its own task
    for e in noisy_names:
        cfg = {"what": "standalone", "expert": e, "tag": tag}
        if config_key(cfg) not in done:
            i = TASKS8.index(e)
            sd = bank.merge({}, 1.0, default=np.eye(8)[i])
            acc = ev.evaluate(sd, args.split, tasks=[e])[e]
            append_record(out, cfg, {"acc": acc, "clean_acc": ref[e], "norm": acc / ref[e]})
            print(f"  standalone {e}: {100 * acc:.2f} (clean {100 * ref[e]:.2f})", flush=True)
    # 2. identification within maps
    ident = {}
    is_noisy = np.array([e in noisy_names for e in TASKS8])
    for rule in RULES:
        table = build_rule(rule, rows, TASKS8, sizes)
        # fraction of maps where the rule's ordering puts noisy experts BELOW clean ones in weight
        # (i.e. noisy experts have higher score c): mean over maps of AUC(score | noisy vs clean)
        aucs = []
        for v in table.values():
            a, b_ = v[is_noisy], v[~is_noisy]
            aucs.append(np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b_]))
        ident[rule] = {"auc_noisy_higher_score_mean": float(np.mean(aucs)),
                       "frac_maps_auc_gt_0.5": float(np.mean(np.array(aucs) > 0.5))}
    append_record(out, {"what": "identification", "tag": tag}, ident)
    print(json.dumps(ident, indent=0), flush=True)
    # 3. merges
    alphas = [float(a) for a in args.alphas.split(",")]
    bms = [float(b) for b in args.bmults.split(",")]

    def run(rule, b, coeffs, alpha, default=None):
        cfg = {"what": "merge", "rule": rule, "b": b, "alpha": alpha, "tag": tag}
        if config_key(cfg) in done:
            return
        sd = bank.merge(coeffs, alpha, default=default)
        acc = ev.evaluate(sd, args.split)
        m = summarize(acc, ref)
        noisy_norm = float(np.mean([acc[e] / ref[e] for e in noisy_names]))
        clean_norm = float(np.mean([acc[e] / ref[e] for e in TASKS8 if e not in noisy_names]))
        append_record(out, cfg, {**m, "acc": acc, "noisy_tasks_norm": noisy_norm, "clean_tasks_norm": clean_norm},
                      {"coeff_summary": summarize_coeffs(coeffs) if coeffs else {}})
        print(f"  {rule:20s} b={b:<5g} a={alpha:<4g} norm={100 * m['mean_norm']:.2f} noisyT={100 * noisy_norm:.2f} "
              f"cleanT={100 * clean_norm:.2f}", flush=True)

    for a in alphas:
        run("uniform", 0.0, {}, a)
        for w in (0.5, 0.25, 0.0):  # oracle: noisy experts get w x the clean weight
            lam = np.where(is_noisy, w, 1.0)
            run(f"oracle_down{w:g}", 0.0, {}, a, default=lam / lam.sum())
    for rule in RULES:
        table = build_rule(rule, rows, TASKS8, sizes)
        scale = scale_of(table)
        for b in bms:
            coeffs = argmin_coeffs(table) if math.isinf(b) else layerwise_coeffs(table, b / scale)
            for a in alphas:
                run(rule, b, coeffs, a)
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
