"""Shared-target regime: the one setting where the review's math supports kappa-decreasing weights.

Review 3.3 / Fig. 1 (right): if experts are noisy estimates of ONE target, the optimal convex weights are
inverse variance; a decreasing rule in a score helps iff the score tracks the estimation error.
Two tests (each soup is evaluated only on the target task's dev split):

A  controlled noise on real full-FT experts (budget A for the rules; the noise levels are known):
   for each expert e, K=4 copies W_e + r_j * rms(Delta_e^p) * G_j on every linear map p, r = (0.25, 0.5, 1, 2).
   Oracle = inverse variance lambda_j ~ 1/r_j^2 (functional error of a linear map scales with r_j^2).
B  real same-task LoRA variants (Alexandre's banks: seeds, logit scales, label smoothing, step budgets):
   for each task, the pool of ~13 expert variants; quality = own-task dev accuracy (diagnostic only).

Rules (layerwise unless 'global'): uniform; kappa literal (softmax(-beta*kappa2(W_j)), master's beta multipliers);
logkappa; kappa_rev; global:logkappa; update-spread rules dsrank (softmax(-beta*stable_rank(Delta_j)),
lower spread -> more weight) and its reverse; dnorm / dnorm_rev; plus oracles (inverse variance in A,
quality softmax and best-single in B).
"""

from __future__ import annotations

import argparse
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

from condmerge.aa_common import TaskData, is_linear_map, load_base_unprefixed, load_expert_unprefixed, new_model  # noqa: E402

BETA_MULTS = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, math.inf)
MOLOHIO = Path("/shared/ssd_30T/molohio_vit20")
LORA_VARIANTS = ("lora_c5a", "lora_c5a_seed1", "lora_c5b", "lora_c5b_seed1", "lora_c5_ls030", "lora_c5_ls030_seed1",
                 "lora_c5_s300", "lora_c5_s300_seed1", "lora_c5_s1000", "lora_c5_s3000", "lora_c5_cf050",
                 "lora_c5_ls010", "lora")
VIT20_NAME = {"cars": "cars", "dtd": "dtd", "eurosat": "eurosat", "gtsrb": "gtsrb", "mnist": "mnist",
              "resisc45": "resisc45", "sun397": "sun397", "svhn": "svhn"}


_REF_MED: dict = {}


def ref_median(m: int, n: int) -> float:
    """Median singular value of one m x n unit-variance Gaussian matrix (seed 0); ~22.40 (768^2), ~53.0 (768x3072)."""
    if (m, n) not in _REF_MED:
        g = np.random.default_rng(0).standard_normal((m, n))
        _REF_MED[(m, n)] = float(np.median(np.linalg.svd(g, compute_uv=False)))
    return _REF_MED[(m, n)]


def svals_np(m: torch.Tensor) -> np.ndarray:
    return np.linalg.svd(m.double().cpu().numpy(), compute_uv=False)


def softmax_coeffs(c: np.ndarray, beta: float) -> np.ndarray:
    if beta == 0:
        return np.full(c.shape[0], 1.0 / c.shape[0])
    if math.isinf(beta):
        m = (c == np.min(c)).astype(np.float64)
        return m / m.sum()
    z = -beta * c
    z = z - z.max()
    w = np.exp(z)
    return w / w.sum()


def scale_of(table: dict[str, np.ndarray]) -> float:
    sds = [np.std(v) for v in table.values() if np.std(v) > 0]
    return float(np.median(sds)) if sds else 1.0


def stats_for(base: dict, deltas: dict[str, torch.Tensor], names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per linear map p: (K,) arrays of kappa2(W0+D_j), ||D_j||_F, stable_rank(D_j)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        w0 = base[p].reshape(base[p].shape[0], -1).double().cpu()
        ks, fr, sr, md, sh, ns = [], [], [], [], [], []
        mm, nn = w0.shape
        ref = ref_median(mm, nn)
        for j in range(deltas[p].shape[0]):
            d = deltas[p][j].reshape(w0.shape).double().cpu()
            s = svals_np(w0 + d)
            sd = svals_np(d)
            ks.append(s[0] / s[-1])
            fr.append(np.sqrt(np.sum(sd ** 2)))
            sr.append(np.sum(sd ** 2) / sd[0] ** 2)
            md.append(np.median(sd))
            sig = np.median(sd) / ref
            edge = sig * (np.sqrt(mm) + np.sqrt(nn))
            sh.append(sig)
            ns.append(1.0 - float(np.sum(sd[sd > edge] ** 2) / np.sum(sd ** 2)))
        return p, np.array(ks), np.array(fr), np.array(sr), np.array(md), np.array(sh), np.array(ns), mm * nn

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, names))
    return {"kappa": {r[0]: r[1] for r in res}, "fro": {r[0]: r[2] for r in res}, "srank": {r[0]: r[3] for r in res},
            "medsv": {r[0]: r[4] for r in res}, "sighat": {r[0]: r[5] for r in res}, "noiseshare": {r[0]: r[6] for r in res},
            "size": {r[0]: r[7] for r in res}}


def rule_tables(st: dict) -> dict[str, dict[str, np.ndarray]]:
    neg = lambda t: {p: -v for p, v in t.items()}  # noqa: E731
    log = lambda t: {p: np.log(v) for p, v in t.items()}  # noqa: E731
    lk = log(st["kappa"])
    g = np.mean(np.stack(list(lk.values())), 0)
    return {"kappa": st["kappa"], "logkappa": lk, "kappa_rev": neg(st["kappa"]),
            "global:logkappa": {p: g for p in lk}, "dsrank": st["srank"], "dsrank_rev": neg(st["srank"]),
            "dnorm": st["fro"], "dnorm_rev": neg(st["fro"]),
            # Marchenko-Pastur bulk noise estimate: s_hat ~ median singular value of Delta (the MP constant cancels
            # across copies of one map); score 2*log(s_hat) so that raw beta = 1 gives lambda ~ s_hat^-2
            "dmedsv": {p: 2 * np.log(v) for p, v in st["medsv"].items()},
            "dnoiseshare": st["noiseshare"]}


def merged_params(base: dict, deltas: dict, lam: dict[str, np.ndarray], default: np.ndarray, alpha: float,
                  device: str, extra: dict | None = None) -> dict:
    out = {}
    for n, w in base.items():
        if n in deltas:
            l = torch.as_tensor(lam.get(n, default), dtype=torch.float32, device=device)
            out[n] = w + alpha * torch.tensordot(l, deltas[n], dims=1)
        else:
            out[n] = w
    if extra:
        out.update(extra)
    return out


def run_pool(tag, task, base, deltas, names, data, model, device, f, oracle_rules, alpha=1.0, extra_params=None):
    K = next(iter(deltas.values())).shape[0]
    st = stats_for(base, deltas, names)
    tables = rule_tables(st)
    uni = np.full(K, 1.0 / K)
    jobs = [("uniform", 0.0, {p: uni for p in names})]
    for r, tab in tables.items():
        sc = scale_of(tab)
        for b in BETA_MULTS:
            beta = b / sc if math.isfinite(b) else math.inf
            jobs.append((r, b, {p: softmax_coeffs(tab[p], beta) for p in names}))
    # tuning-free inverse-variance rule from the update spectrum's bulk (raw beta = 1 on 2 log s_hat)
    jobs.append(("mp_invvar", 1.0, {p: softmax_coeffs(tables["dmedsv"][p], 1.0) for p in names}))
    # global Gavish-Donoho inverse variance: lambda_j ~ 1 / sum_p sigma_hat_jp^2 * m_p * n_p (total noise energy)
    tot = np.sum(np.stack([st["sighat"][p] ** 2 * st["size"][p] for p in names]), 0)
    gl = (1 / tot) / np.sum(1 / tot)
    jobs.append(("global:mp_invvar", 1.0, {p: gl for p in names}))
    for r, lam in oracle_rules.items():
        jobs.append((r, math.nan, {p: lam for p in names}))
    for rule, b, co in jobs:
        params = merged_params(base, deltas, co, uni, alpha, device, extra_params)
        res = data.evaluate(model, "dev", params=params, tasks=[task])
        mean_lam = np.mean(np.stack(list(co.values())), 0)
        rec = {"pool": tag, "task": task, "rule": rule, "b": b, "alpha": alpha, "acc": res["acc"][task],
               "ce": res["ce"][task], "mean_lambda": mean_lam.tolist()}
        f.write(json.dumps(rec) + "\n")
    f.flush()
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,B")
    ap.add_argument("--out", default=str(RESULTS / "aa" / "soup"))
    ap.add_argument("--levels", default="0.25,0.5,1,2", help="part A relative noise levels (x per-matrix rms of Delta)")
    ap.add_argument("--tag", default="", help="suffix for part A output files")
    ap.add_argument("--norm-matched", action="store_true",
                    help="part A: rescale every noisy copy's update to the clean update's Frobenius norm per map "
                         "(norm-based rules become blind; tests spectral information beyond magnitude)")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base_cpu = load_base_unprefixed()
    base = {k: v.to(dev) for k, v in base_cpu.items()}
    data = TaskData(TASKS8, ("dev",), dev)
    model = new_model(dev)
    linear = [k for k in base if is_linear_map(k, base[k])]

    if "A" in args.parts:
        levels = np.array([float(x) for x in args.levels.split(",")])
        inv = (1 / levels ** 2) / np.sum(1 / levels ** 2)
        meta = {}
        with (out / f"partA_controlled_noise{args.tag}.jsonl").open("w") as f:
            for e in TASKS8:
                ex = load_expert_unprefixed(e)
                g = torch.Generator(device=dev).manual_seed(TASKS8.index(e))
                deltas = {}
                for p in linear:
                    d = (ex[p].to(dev) - base[p])
                    rms = d.pow(2).mean().sqrt()
                    noise = torch.randn((len(levels),) + d.shape, generator=g, device=dev)
                    deltas[p] = d.unsqueeze(0) + torch.as_tensor(levels, dtype=torch.float32, device=dev).view(-1, *[1] * d.ndim) * rms * noise
                    if args.norm_matched:
                        nrm = deltas[p].flatten(1).norm(dim=1).view(-1, *[1] * d.ndim)
                        deltas[p] = deltas[p] * (d.norm() / nrm)
                # non-linear params: the clean expert's values
                extra = {k: ex[k].to(dev) for k in base if k not in deltas}
                st = run_pool("A_noise", e, base, deltas, linear, data, model, dev, f,
                              {"oracle_invvar": inv, "cleanest_only": np.eye(len(levels))[0], "noisiest_only": np.eye(len(levels))[-1]},
                              extra_params=extra)
                kl = len(levels)
                rho = [float(np.corrcoef(np.argsort(np.argsort(st["kappa"][p])), np.arange(kl))[0, 1]) for p in linear]
                rho_s = [float(np.corrcoef(np.argsort(np.argsort(st["srank"][p])), np.arange(kl))[0, 1]) for p in linear]
                meta[e] = {"mean_rank_corr_kappa_vs_noise": float(np.mean(rho)), "frac_kappa_decreasing_in_noise": float(np.mean(np.array(rho) < 0)),
                           "mean_rank_corr_srank_vs_noise": float(np.mean(rho_s))}
                print(f"A {e}: kappa-vs-noise rank corr {meta[e]['mean_rank_corr_kappa_vs_noise']:+.2f} "
                      f"(frac decreasing {meta[e]['frac_kappa_decreasing_in_noise']:.2f}); srank {meta[e]['mean_rank_corr_srank_vs_noise']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
                del deltas
                torch.cuda.empty_cache()
        (out / f"partA_meta{args.tag}.json").write_text(json.dumps({"levels": levels.tolist(), "per_expert": meta}, indent=1))

    if "B" in args.parts:
        meta = {}
        with (out / f"partB_lora_variants{args.tag}.jsonl").open("w") as f:
            for t in TASKS8:
                avail = [v for v in LORA_VARIANTS if (MOLOHIO / v / f"{VIT20_NAME[t]}.npz").exists()]
                facs = [np.load(MOLOHIO / v / f"{VIT20_NAME[t]}.npz") for v in avail]
                tnames = sorted({k[2:] + ".weight" for k in facs[0].files if k.startswith("A_")})
                gam = []
                for v in avail:
                    cfgp = MOLOHIO / v / "adapter_config.json"
                    gam.append(1.0)  # all banks: r=16, alpha=16 (README) -> gamma 1
                deltas = {}
                for p in tnames:
                    stem = p.removesuffix(".weight")
                    deltas[p] = torch.stack([torch.from_numpy(z[f"B_{stem}"] @ z[f"A_{stem}"]).float() * gm
                                             for z, gm in zip(facs, gam)]).to(dev)
                # standalone quality of each variant on its own task (diagnostic)
                q = []
                for j in range(len(avail)):
                    oh = np.eye(len(avail))[j]
                    params = merged_params(base, deltas, {}, oh, 1.0, dev)
                    q.append(data.evaluate(model, "dev", params=params, tasks=[t])["acc"][t])
                q = np.array(q)
                oracles = {"oracle_best_single": np.eye(len(avail))[int(np.argmax(q))]}
                for bq in (10.0, 30.0, 100.0):
                    w = np.exp(bq * (q - q.max()))
                    oracles[f"oracle_quality_softmax{bq:g}"] = w / w.sum()
                st = run_pool("B_lora", t, base, deltas, tnames, data, model, dev, f, oracles)
                gk = np.mean(np.stack([np.log(st["kappa"][p]) for p in tnames]), 0)
                gs = np.mean(np.stack([st["srank"][p] for p in tnames]), 0)
                from scipy.stats import spearmanr
                meta[t] = {"variants": avail, "quality": q.tolist(),
                           "spearman_quality_vs_global_logkappa": float(spearmanr(q, gk).statistic),
                           "spearman_quality_vs_global_srank": float(spearmanr(q, gs).statistic)}
                print(f"B {t}: {len(avail)} variants, quality {q.min():.3f}-{q.max():.3f}; rho(q, mean log kappa) "
                      f"{meta[t]['spearman_quality_vs_global_logkappa']:+.2f}, rho(q, mean srank) {meta[t]['spearman_quality_vs_global_srank']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
                del deltas
                torch.cuda.empty_cache()
        (out / f"partB_meta{args.tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
