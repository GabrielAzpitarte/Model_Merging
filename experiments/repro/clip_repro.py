"""Reproduction audit of the CLIP results the conclusions depend on (protocol 02 sec. 7), with dispatcher checks.

Part rules   ViT-B/32 frozen TEST: uniform (tuned, alpha 1), literal kappa (tuned, alpha 1), kappa_rev, 5 shuffles.
             Same code path as experiments/freeze_eval.py (build_rule -> scale_of -> layerwise_coeffs ->
             Bank.merge). Logged per config: per-map coefficients (73 x 8), max |lambda - 1/8|, n_eff, a merged-
             update fingerprint sum_p <u_p, theta_p - theta0_p> (u_p fixed N(0,1) seeded by crc32(name)), the
             merged-update norm, test metrics, per-image agreement with the ORIGINAL run's saved correctness, and
             metric differences vs the original records.
             Also an audit of the original records themselves: the per-image agreement between the original
             kappa|tuned and uniform|tuned correctness (a silently uniform kappa run would agree 100 %).
Part regmean full-FT RegMean, budget C, frozen strength-sweep configs (n = 16 subsets 0-2 and n = 512):
             kappa-cap ridge 1e5, tuned fixed relative ridge, tuned shrinkage; Grams recomputed from the same
             calibration subsets; solve residuals ||D (G~ + rho I) - rhs|| / ||rhs||; metrics vs records.
Part lora    LoRA RegMean frozen test (KnOTS release, lora_c5a, vit20): kappa-cap 1e4 (ridge to zero) vs shrink
             0.95 vs uniform at the recorded alphas, cached Grams; plus the KnOTS budget-C Grams recomputed and
             compared with the cache.
Writes results/repro/clip_repro_<part>.json.  Usage: clip_repro.py --parts rules,regmean,lora
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

from condmerge.paths import ARTIFACTS, CACHE, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)
OUT = ROOT / "results" / "repro"


def fingerprint(delta_by_name: dict[str, torch.Tensor]) -> float:
    fp = 0.0
    for n, d in sorted(delta_by_name.items()):
        g = torch.Generator(device="cpu").manual_seed(zlib.crc32(n.encode()))
        u = torch.randn(d.numel(), generator=g, dtype=torch.float64)
        fp += float(u @ d.detach().double().cpu().flatten())
    return fp


def part_rules(dev: str) -> dict:
    from condmerge.clip import Evaluator, load_base_state, load_experts, summarize
    from condmerge.merging import Bank, concentration, layerwise_coeffs
    from condmerge.runlog import read_jsonl
    from condmerge.scores import build_rule, load_stats, scale_of
    from condmerge.spectra import linear_map_names
    from rule_sweep import argmin_coeffs

    tags = ["uniform|tuned", "uniform|alpha1", "kappa|tuned", "kappa|alpha1", "kappa_rev|tuned"] + \
        [f"shuf{s}:kappa|tuned" for s in range(5)]
    frozen = {c["tag"]: c for c in json.loads((RESULTS / "freeze" / "frozen_configs_w1.json").read_text())}
    recs = {r["cfg"]["tag"]: r for r in read_jsonl(RESULTS / "freeze" / "test.jsonl")}
    ev = Evaluator(dev, splits=("test",), tasks=TASKS8)
    base = load_base_state()
    bank = Bank.build(base, load_experts(TASKS8), dev)
    ref = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"]["test"]
    rows = load_stats()
    names = linear_map_names(base)
    sizes = {p: int(base[p].numel()) for p in names}
    out = {"configs": {}}
    corr_store = {}
    for tag in tags:
        c = frozen[tag]
        t0 = time.time()
        coeffs = {}
        if c["rule"] != "uniform" and float(c["b"]) != 0:
            table = build_rule(c["rule"], rows, TASKS8, sizes)
            b = float(c["b"])
            coeffs = argmin_coeffs(table) if math.isinf(b) else layerwise_coeffs(table, b / scale_of(table))
        sd = bank.merge(coeffs, float(c["alpha"]))
        delta = {k: sd[k] - bank.base[k] for k in bank.names}
        lam = np.stack([coeffs[p] for p in sorted(coeffs)]) if coeffs else np.full((len(names), 8), 1 / 8)
        ev.load(sd)
        corr = {t: ev.correct(t, "test") for t in TASKS8}
        corr_store[tag] = corr
        acc = {t: float(v.mean()) for t, v in corr.items()}
        m = summarize(acc, ref)
        rec = recs.get(tag)
        orig = np.load(ARTIFACTS / "runs" / "freeze_correct" / f"{rec['key']}.npz") if rec else None
        agree = {t: float(np.mean(orig[t] == corr[t])) for t in TASKS8} if orig is not None else None
        out["configs"][tag] = {
            "rule": c["rule"], "b": c["b"], "alpha": c["alpha"], "mean_norm": m["mean_norm"], "mean_acc": m["mean_acc"],
            "recorded_mean_norm": rec["metrics"]["mean_norm"] if rec else None,
            "diff_vs_record_pt": 100 * (m["mean_norm"] - rec["metrics"]["mean_norm"]) if rec else None,
            "per_image_agreement_with_original": agree,
            "max_abs_lambda_minus_uniform": float(np.abs(lam - 1 / 8).max()),
            "n_eff_mean": float(np.mean([concentration(v)["n_eff"] for v in coeffs.values()])) if coeffs else 8.0,
            "lambda_maps_x_experts": np.round(lam, 6).tolist(), "fingerprint": fingerprint(delta),
            "update_norm": float(np.sqrt(sum(float(d.double().pow(2).sum()) for d in delta.values()))),
            "seconds": time.time() - t0}
        print(f"rules {tag:22s} norm {100 * m['mean_norm']:.3f} (record {100 * rec['metrics']['mean_norm']:.3f}) "
              f"agree {np.mean(list(agree.values())) if agree else float('nan'):.5f} maxdev {out['configs'][tag]['max_abs_lambda_minus_uniform']:.4f}",
              flush=True)
    # audit of the ORIGINAL records: kappa|tuned vs uniform|tuned per-image agreement
    ok, ou = (np.load(ARTIFACTS / "runs" / "freeze_correct" / f"{recs[t]['key']}.npz") for t in ("kappa|tuned", "uniform|tuned"))
    out["original_kappa_vs_uniform_agreement"] = {t: float(np.mean(ok[t] == ou[t])) for t in TASKS8}
    fpu = out["configs"]["uniform|tuned"]["fingerprint"]
    out["fingerprint_distinct_from_uniform_tuned"] = {t: abs(v["fingerprint"] - fpu) > 1e-6 * max(1, abs(fpu))
                                                      for t, v in out["configs"].items() if t != "uniform|tuned"}
    return out


def part_regmean(dev: str) -> dict:
    from transformers import CLIPVisionConfig, CLIPVisionModel

    from condmerge.clip import Evaluator, load_base_state, load_experts, summarize
    from condmerge.data import load_cache
    from condmerge.grams import collect_grams, input_name
    from condmerge.paths import BASE_MODEL
    from condmerge.regmean import regmean_solve
    from condmerge.runlog import read_jsonl
    from condmerge.spectra import as_matrix, linear_map_names

    base = load_base_state()
    experts = load_experts(TASKS8)
    names = linear_map_names(base)
    ev = Evaluator(dev, splits=("test",), tasks=TASKS8)
    ref = json.loads((RESULTS / "experts" / "expert_accuracy.json").read_text())["expert"]["test"]
    model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(BASE_MODEL)).eval().to(dev)
    deltas = {p: [as_matrix(p, experts[e][p]).double() - as_matrix(p, base[p]).double() for e in TASKS8] for p in names}
    calib = {e: load_cache(e, "calib", mmap=True)[0] for e in TASKS8}
    out = {"configs": []}
    for n, seed in ((16, 0), (16, 1), (16, 2), (512, 0)):
        recs = read_jsonl(RESULTS / "regmean" / f"strength_sweep_C_test_n{n}s{seed}.jsonl")
        grams = {}
        for e in TASKS8:
            model.load_state_dict({k: v.to(dev) for k, v in experts[e].items()}, strict=True)
            x = calib[e]
            if n < x.shape[0]:
                x = x[np.sort(np.random.default_rng(seed).choice(x.shape[0], n, replace=False))]
            grams[e] = collect_grams(model, [x], names, dev)["grams"]
        for rec in recs:
            c = rec["cfg"]
            kw = {"kcap_ridge": {"kappa_target": c["kappa_t"]}, "rel_ridge": {"rel_ridge": rec["c_rel"]},
                  "fixed_shrink": {"shrink": c["shrink"]}}[c["cell"]]
            t0 = time.time()
            merged, resid, rhos, sym, psd = {}, [], [], [], []
            for p in names:
                gs = [grams[e][input_name(p)] for e in TASKS8]
                sol, info = regmean_solve([d.to(dev) for d in deltas[p]], gs, ridge_to="mean", return_info=True, **kw)
                # residual of the solved system, recomputed independently
                a = info["shrink"]
                gt = [a * g.to(dev).double() + (1 - a) * torch.diag(torch.diagonal(g.to(dev).double())) for g in gs]
                G = sum(gt)
                rhs = sum(d.to(dev) @ gi for d, gi in zip(deltas[p], gt))
                if info["rho"] > 0:
                    rhs = rhs + info["rho"] * sum(d.to(dev) for d in deltas[p]) / len(deltas[p])
                    G = G + info["rho"] * torch.eye(G.shape[0], dtype=G.dtype, device=dev)
                resid.append(float((sol @ G - rhs).norm() / max(float(rhs.norm()), 1e-300)))
                sym.append(float((G - G.T).abs().max() / G.abs().max()))
                psd.append(float(torch.linalg.eigvalsh(0.5 * (G + G.T))[0] / torch.linalg.eigvalsh(0.5 * (G + G.T))[-1]))
                rhos.append(info["rho"] / max(info["trace_mean"], 1e-300))
                merged[p] = sol.reshape(base[p].shape).float().cpu()
            sd = {k: base[k] + c["alpha"] * (merged[k] if k in merged else
                                              torch.stack([experts[e][k] - base[k] for e in TASKS8]).mean(0)) for k in base}
            acc = ev.evaluate(sd, "test")
            m = summarize(acc, ref)
            out["configs"].append({"n": n, "seed": seed, "cell": c["cell"], "kw": kw, "mean_norm": m["mean_norm"],
                                   "recorded_mean_norm": rec["metrics"]["mean_norm"],
                                   "diff_vs_record_pt": 100 * (m["mean_norm"] - rec["metrics"]["mean_norm"]),
                                   "max_rel_residual": float(np.max(resid)), "max_rel_asymmetry": float(np.max(sym)),
                                   "min_eig_over_max": float(np.min(psd)), "rel_rho_median": float(np.median(rhos)),
                                   "fingerprint": fingerprint({p: merged[p] for p in names}), "seconds": time.time() - t0})
            r_ = out["configs"][-1]
            print(f"regmean n={n} s={seed} {c['cell']:12s} norm {100 * m['mean_norm']:.3f} (record "
                  f"{100 * rec['metrics']['mean_norm']:.3f}) resid {r_['max_rel_residual']:.1e}", flush=True)
        del grams
        torch.cuda.empty_cache()
    return out


def part_lora(dev: str) -> dict:
    from aa_lora_regmean import input_name
    from aa_lora_regmean_test import CONFIGS, correctness
    from aa_lora_replicate_banks import load_bank
    from condmerge.aa_common import TaskData, load_base_unprefixed, new_model
    from condmerge.regmean_lora import regmean_full

    recorded = json.loads((RESULTS / "aa" / "lora" / "regmean_test.json").read_text())
    base = load_base_unprefixed()
    base_dev = {k: v.to(dev) for k, v in base.items()}
    data = TaskData(TASKS8, ("test",), dev)
    model = new_model(dev)
    out = {"banks": {}}
    for bank in ("knots_release", "lora_c5a", "lora"):
        b = load_bank(bank)
        names = sorted(b[TASKS8[0]])
        g = torch.load(CACHE / "aa" / f"grams_lora_{bank}_C.pt", weights_only=False)["grams"]
        out["banks"][bank] = {}
        corr = {}
        for label, kw in CONFIGS.items():
            a = recorded[bank][label]["alpha"]
            params = dict(base_dev)
            upd = {}
            for n in names:
                d = torch.stack([b[t][n] for t in TASKS8]).double()
                m = d.mean(0).float() if kw is None else regmean_full(d, g[input_name(n)].double(), **kw)[0]
                upd[n] = a * m
                params[n] = base_dev[n] + a * m.to(dev)
            c = correctness(model, data, params)
            corr[label] = c
            acc = float(np.mean([c[t].mean() for t in TASKS8]))
            out["banks"][bank][label] = {"alpha": a, "avg_acc": acc, "recorded_avg_acc": recorded[bank][label]["avg_acc"],
                                         "diff_vs_record_pt": 100 * (acc - recorded[bank][label]["avg_acc"]),
                                         "fingerprint": fingerprint(upd)}
            print(f"lora {bank} {label:13s} acc {100 * acc:.3f} (record {100 * recorded[bank][label]['avg_acc']:.3f})",
                  flush=True)
        out["banks"][bank]["kcap_minus_shrink_pt"] = 100 * (out["banks"][bank]["kcap1e4_zero"]["avg_acc"] -
                                                            out["banks"][bank]["shrink0.95"]["avg_acc"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="rules,regmean,lora")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for part in args.parts.split(","):
        t0 = time.time()
        res = {"rules": part_rules, "regmean": part_regmean, "lora": part_lora}[part](args.device)
        res["seconds"] = time.time() - t0
        (OUT / f"clip_repro_{part}.json").write_text(json.dumps(res, indent=1, default=float))
        print(f"PART {part} DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
