"""Literal condition-number rule and controls on the KnOTS LoRA r16 bank (dev split; budget A unless noted).

Mirrors master's full-FT sweep (experiments/rule_sweep.py) so the two banks line up:
  beta = b / scale(rule), scale = median over matrices of the across-expert std of the rule's scores,
  b in {0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, inf (argmin selection)}; alpha in {0.6, 0.8, 1.0, 1.3, 1.6}
  (LoRA-appropriate grid, shared by every rule; the first launch with the full-FT grid showed the uniform
  optimum at alpha <= 1 and collapse above 1.6, those rows are kept in the shard files).
Merge: W0 + alpha * sum_i lambda_i^p * gamma B_i A_i on the 48 adapted matrices (exact, dense, rank <= 128);
other parameters stay at the base. Lower score -> higher lambda.

Rules (scores per adapted matrix p and expert i):
  kappa / kappa_rev / logkappa / logkappa_rev / kappa_z / kappa_z_rev   kappa2(W0 + Delta_i), the literal rule
  dnorm / dnorm_rev                                                     ||Delta_i||_F (magnitude control)
  dlogkplus_0.01 / dlogkplus_0.001 (+ _rev)                             log kappa_plus(Delta_i; tau) via QR core
  dstable_rank / dentropy_rank                                          update spread
  factor_logkappaA / factor_logkappaA_gauged                            log kappa(A_i): gauge DEPENDENT demo
  shuf{0,1,2}:kappa                                                     kappa shuffled across experts per matrix
  global:kappa / global:logkappa                                        one weight per expert (mean score)
Controls: uniform (beta=0) over the alpha grid (= Task Arithmetic), MetaGPT (lambda ~ ||Delta_i||^2, global),
best-single-expert (one-hot, alpha grid), and rank-16 compression of selected merges (fixed final-rank budget).
"""

from __future__ import annotations

import argparse
import hashlib
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

from condmerge.aa_common import TaskData, new_model  # noqa: E402
from condmerge.lora import LoRABank, core_svals, gauge, load_knots_lora  # noqa: E402

BETA_MULTS = (0.25, 1, 4, 16, 64, math.inf)  # pruned from master's 11-point grid for compute (same for every rule)
ALPHAS = (0.6, 0.8, 1.0, 1.3, 1.6)  # LoRA bank: uniform optimum at alpha<=1 (full-FT grid wasted; same grid for every rule)


def lora_score_tables(stats_csv: Path, names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    import csv

    rows = list(csv.DictReader(stats_csv.open()))
    tab = {}
    for r in rows:
        tab.setdefault(r["param"], {})[r["expert"]] = r

    def col(c, f=float):
        return {p: np.array([f(tab[p][e][c]) for e in TASKS8]) for p in names}

    z = lambda t: {p: (v - v.mean()) / v.std() if v.std() > 0 else np.zeros_like(v) for p, v in t.items()}  # noqa: E731
    neg = lambda t: {p: -v for p, v in t.items()}  # noqa: E731
    log = lambda t: {p: np.log(v) for p, v in t.items()}  # noqa: E731
    kap = col("kappa2_W")
    out = {
        "kappa": kap, "kappa_rev": neg(kap), "logkappa": log(kap), "logkappa_rev": neg(log(kap)),
        "kappa_z": z(kap), "kappa_z_rev": neg(z(kap)),
        "dnorm": col("fro_D"), "dnorm_rev": neg(col("fro_D")),
        "dlogkplus_0.01": log(col("kplus_D_0.01")), "dlogkplus_0.01_rev": neg(log(col("kplus_D_0.01"))),
        "dlogkplus_0.001": log(col("kplus_D_0.001")), "dlogkplus_0.001_rev": neg(log(col("kplus_D_0.001"))),
        "dstable_rank": col("stable_rank_D"), "dentropy_rank": col("entropy_rank_D"),
        "factor_logkappaA": log(col("kappa_A")),
    }
    # the same factor rule after an exact random gauge change of every adapter (function unchanged)
    rng = torch.Generator().manual_seed(123)
    fac = {t: load_knots_lora(t) for t in TASKS8}
    ga = {}
    for p in names:
        vals = []
        for t in TASKS8:
            b, a, _ = fac[t][p]
            u = torch.linalg.qr(torch.randn(16, 16, generator=rng, dtype=torch.float64))[0]
            v = torch.linalg.qr(torch.randn(16, 16, generator=rng, dtype=torch.float64))[0]
            rmat = u @ torch.diag(torch.logspace(0, 2, 16, dtype=torch.float64)) @ v.T
            _, a2 = gauge(b.double(), a.double(), rmat)
            s = torch.linalg.svdvals(a2)
            vals.append(float(torch.log(s[0] / s[-1])))
        ga[p] = np.array(vals)
    out["factor_logkappaA_gauged"] = ga
    for s in (0, 1, 2):
        r = np.random.default_rng(s)
        out[f"shuf{s}:kappa"] = {p: v[r.permutation(8)] for p, v in sorted(kap.items())}
    for base_rule in ("kappa", "logkappa"):
        g = np.mean(np.stack([out[base_rule][p] for p in names]), 0)
        out[f"global:{base_rule}"] = {p: g.copy() for p in names}
    return out


def scale_of(table: dict[str, np.ndarray]) -> float:
    sds = [np.std(v[np.isfinite(v)]) for v in table.values() if np.isfinite(v).sum() >= 2]
    sds = [s for s in sds if s > 0]
    return float(np.median(sds)) if sds else 1.0


def softmax_coeffs(c: np.ndarray, beta: float) -> np.ndarray:
    k = c.shape[0]
    if beta == 0:
        return np.full(k, 1.0 / k)
    if math.isinf(beta):
        m = (c == np.min(c)).astype(np.float64)
        return m / m.sum()
    logits = -beta * c
    logits = logits - logits.max()
    w = np.exp(logits)
    return w / w.sum()


def coeff_hash(coeffs: dict[str, np.ndarray], alpha: float, extra: str = "") -> str:
    h = hashlib.sha1()
    for p in sorted(coeffs):
        h.update(np.round(coeffs[p], 10).tobytes())
    h.update(f"{alpha:.6f}{extra}".encode())
    return h.hexdigest()[:16]


def conc(coeffs: dict[str, np.ndarray]) -> dict:
    ne = [float(np.exp(-np.sum(v[v > 0] * np.log(v[v > 0])))) for v in coeffs.values()]
    mx = [float(v.max()) for v in coeffs.values()]
    mean_lam = np.mean(np.stack(list(coeffs.values())), 0)
    return {"n_eff_mean": float(np.mean(ne)), "n_eff_min": float(np.min(ne)), "max_mean": float(np.mean(mx)),
            "mean_lambda": mean_lam.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="all")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=str(RESULTS / "aa" / "lora" / "rule_sweep_dev.jsonl"))
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for fp in out.parent.glob("rule_sweep_dev_shard*.jsonl"):  # dedupe across shards and earlier launches
        done |= {json.loads(l)["chash"] for l in fp.read_text().splitlines() if l.strip()}
    t0 = time.time()
    bank = LoRABank(TASKS8, dev)
    data = TaskData(TASKS8, (args.split,), dev)
    model = new_model(dev)
    names = bank.linear
    tables = lora_score_tables(RESULTS / "aa" / "lora" / "lora_stats.csv", names)
    rules = list(tables) if args.rules == "all" else args.rules.split(",")
    si, sn = (int(x) for x in args.shard.split("/"))

    jobs = []  # (label, rule, b, alpha, coeffs, compress)
    uni = {p: np.full(8, 1 / 8) for p in names}
    for a in ALPHAS:
        jobs.append(("uniform", "uniform", 0.0, a, uni, None))
    for i, e in enumerate(TASKS8):
        one = {p: np.eye(8)[i] for p in names}
        for a in ALPHAS:
            jobs.append((f"single:{e}", "single", math.nan, a, one, None))
    # MetaGPT: lambda ~ ||Delta_i||^2 summed over every adapted matrix
    sq = np.zeros(8)
    for p in names:
        sq += bank.deltas[p].double().pow(2).flatten(1).sum(1).cpu().numpy()
    mg = sq / sq.sum()
    for a in ALPHAS:
        jobs.append(("metagpt", "metagpt", math.nan, a, {p: mg for p in names}, None))
    for r in rules:
        sc = scale_of(tables[r])
        for b in BETA_MULTS:
            beta = b / sc if math.isfinite(b) else math.inf
            co = {p: softmax_coeffs(tables[r][p], beta) for p in names}
            for a in ALPHAS:
                jobs.append((r, r, b, a, co, None))
    # fixed final-rank budget: rank-16 compression of the uniform merges (others added after selection)
    for a in ALPHAS:
        jobs.append(("uniform_rank16", "uniform", 0.0, a, uni, 16))
    jobs = [j for n, j in enumerate(jobs) if n % sn == si]
    print(f"{len(jobs)} jobs (shard {si}/{sn}); setup {time.time() - t0:.0f}s", flush=True)
    cache = {}
    with out.open("a") as f:
        for label, rule, b, a, co, comp in jobs:
            ch = coeff_hash(co, a, f"|c{comp}")
            rec = {"label": label, "rule": rule, "b": b, "alpha": a, "compress_rank": comp, "chash": ch,
                   "split": args.split, "budget": "A", "bank": "knots_lora_r16_vitb32"}
            if ch in done:
                continue
            if ch in cache:
                res = cache[ch]
            else:
                params = bank.merged({p: torch.as_tensor(v, dtype=torch.float32, device=dev) for p, v in co.items()},
                                     a, compress_rank=comp)
                res = data.evaluate(model, args.split, params=params)
                cache[ch] = res
            rec.update({"acc": res["acc"], "avg_acc": res["avg_acc"], "avg_ce": res["avg_ce"], **conc(co)})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done.add(ch)
            print(f"{label:24s} b={b!s:5s} a={a:.1f} avg={res['avg_acc']:.4f} neff={rec['n_eff_mean']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
