"""Coefficient rules for the LLM banks, from the exact spectra cache (experiments/llm/compute_spectra.py).

Primary (the professor's exact hypothesis), per adapted matrix p and expert i:
    kappa_i[p]  = sigma_max(W_i[p]) / sigma_min(W_i[p]),   W_i[p] = W0[p] + B_i[p] A_i[p]   (full expert matrix)
    lambda_i[p] = softmax_i(-beta * kappa_i[p]),            beta = b / scale  (ONE beta for all matrices)
    scale       = median over matrices of the across-expert standard deviation of the score
                  (a single global rescaling: it only fixes the units of b; it is not per-matrix normalisation)
Policies (shared by every rule; condmerge.merging.softmax_coeffs):
  b = 0               -> explicit uniform branch (whatever the scores)
  b = inf             -> argmin selection per matrix (ties share the mass equally)
  score +inf          -> weight 0 if some expert is finite; all +inf -> uniform fallback; NaN -> error
  exact ties          -> equal weights (softmax), counted
Named rules (each a different method):
  uniform                 beta = 0
  kappa                   literal score
  kappa_rev               -kappa (reversed preference)
  shuf<s>:kappa           kappa permuted across experts independently within each matrix (seed s): identical
                          per-matrix concentration profile, scores detached from experts
  global:kappa            ONE score per expert = geometric-mean condition number over the adapted matrices,
                          kappa_g,i = exp(mean_p log kappa_i[p]) (a convention: a network has no canonical
                          condition number); lambda_i = softmax(-beta kappa_g,i) at every matrix
  gshuf<s>:global:kappa   the global score vector permuted across experts
  logkappa       (R1)     log kappa: the power law kappa^-beta (temperature comparable across matrices whose
                          kappa differ by orders of magnitude)
  kappa_res      (R2)     sigma_max / max(sigma_min, tau_bf16[p]): the smallest singular value floored at the
                          base checkpoint's BF16 resolution floor (values below it are not determined by the
                          stored weights), so unresolved tails carry no preference
  shuf<s>:<R>, kappa_rev etc. compose with any base rule.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np

from ..merging import softmax_coeffs
from . import banks as BK

SPECTRA = BK.ROOT / "artifacts" / "llm" / "spectra"


@cache
def spectra(bank: str) -> dict:
    """{module: {"s": (V, q), "tau": float, "versions": [...], "err": (V,)}} from the cache."""
    out = {}
    for m in BK.module_names(bank):
        z = np.load(SPECTRA / bank / f"{m}.npz")
        out[m] = {"s": z["s"], "tau": float(z["tau_bf16"]), "versions": [str(v) for v in z["versions"]],
                  "err": z["abs_err_bound"], "dnorm_f": z["dnorm_f"]}
    return out


def _rows(bank: str, T: int) -> list[int]:
    versions = spectra(bank)[BK.module_names(bank)[0]]["versions"]
    return [versions.index(t) for t in BK.tasks(bank, T)]


def kappa_table(bank: str, T: int, kind: str = "raw") -> np.ndarray:
    """(n_modules, K) scores: raw kappa, log kappa, or the precision-floored kappa_res."""
    sp, rows = spectra(bank), _rows(bank, T)
    out = []
    for m in BK.module_names(bank):
        s = sp[m]["s"][rows]
        s1, sq = s[:, 0], s[:, -1]
        if kind == "res":
            sq = np.maximum(sq, sp[m]["tau"])
        with np.errstate(divide="ignore"):
            k = np.where(sq > 0, s1 / np.where(sq > 0, sq, 1.0), np.inf)
        out.append(np.log(k) if kind == "log" else k)
    return np.asarray(out, np.float64)


def score_table(bank: str, T: int, rule: str) -> np.ndarray:
    """(n_modules, K) score table of a named rule (lower score -> larger lambda)."""
    if rule.startswith("shuf"):
        head, base = rule.split(":", 1)
        t = score_table(bank, T, base)
        rng = np.random.default_rng(int(head[4:]))
        return np.stack([row[rng.permutation(len(row))] for row in t])  # sorted module order, fixed seed
    if rule.startswith("gshuf"):
        head, base = rule.split(":", 1)
        t = score_table(bank, T, base)
        perm = np.random.default_rng(int(head[5:])).permutation(t.shape[1])
        return t[:, perm]
    if rule.startswith("global:"):
        t = score_table(bank, T, rule[len("global:"):])
        if not np.all(np.isfinite(t)) or np.any(t <= 0):
            raise ValueError("global aggregation needs finite positive per-matrix condition numbers")
        g = np.exp(np.mean(np.log(t), axis=0))
        return np.broadcast_to(g, t.shape).copy()
    if rule.endswith("_rev"):
        return -score_table(bank, T, rule[: -len("_rev")])
    return {"kappa": lambda: kappa_table(bank, T, "raw"), "logkappa": lambda: kappa_table(bank, T, "log"),
            "kappa_res": lambda: kappa_table(bank, T, "res")}[rule]()


def scale_of(table: np.ndarray) -> float:
    sds = []
    for v in table:
        f = v[np.isfinite(v)]
        if f.size >= 2 and np.std(f) > 0:
            sds.append(np.std(f))
    return float(np.median(sds)) if sds else 1.0


def coefficients(table: np.ndarray, b: float, scale: float | None = None) -> tuple[np.ndarray, dict]:
    """(n_modules, K) lambda for temperature multiplier b, and a log of the special cases hit.
    scale: override the temperature unit (default: scale_of(table)); used to keep beta fixed under an edit."""
    n, k = table.shape
    log = {"n_inf_scores": int(np.isposinf(table).sum()), "n_all_inf_rows": 0, "n_tie_rows": 0,
           "n_neginf_rows": int(np.any(np.isneginf(table), axis=1).sum())}
    if np.any(np.isnan(table)):
        raise ValueError("NaN score")
    if b == 0:
        return np.full((n, k), 1.0 / k), {**log, "branch": "uniform"}
    out = np.zeros((n, k))
    if math.isinf(b):
        for j, v in enumerate(table):
            if np.all(np.isposinf(v)):
                out[j] = 1.0 / k
                log["n_all_inf_rows"] += 1
                continue
            mn = np.min(v)
            msk = (v == mn).astype(np.float64)
            if msk.sum() > 1:
                log["n_tie_rows"] += 1
            out[j] = msk / msk.sum()
        return out, {**log, "branch": "argmin"}
    sc = scale_of(table) if scale is None else scale
    beta = b / sc
    for j, v in enumerate(table):
        if np.all(np.isposinf(v)):
            log["n_all_inf_rows"] += 1
        finite = v[np.isfinite(v)]
        if finite.size >= 2 and np.sum(finite == finite.min()) > 1:
            log["n_tie_rows"] += 1
        out[j] = softmax_coeffs(v, beta)
    return out, {**log, "branch": "softmax", "beta": beta, "scale": sc}


def n_eff(lam: np.ndarray) -> np.ndarray:
    p = np.clip(lam, 1e-300, None)
    return np.exp(-np.sum(np.where(lam > 0, lam * np.log(p), 0.0), axis=1))


def b_for_targets(table: np.ndarray, targets: list[float], lo: float = -4, hi: float = 8, num: int = 241) -> list[dict]:
    """For each target mean n_eff, the b on a fine log grid whose mean n_eff is closest (weights-only diagnostic)."""
    grid = np.logspace(lo, hi, num)
    ne = np.array([n_eff(coefficients(table, b)[0]).mean() for b in grid])
    out = []
    for tgt in targets:
        i = int(np.argmin(np.abs(ne - tgt)))
        b = float(f"{grid[i]:.3g}")
        out.append({"target": tgt, "b": b, "mean_n_eff": float(n_eff(coefficients(table, b)[0]).mean())})
    return out


def jitter(lam: np.ndarray, rel: float, seed: int) -> np.ndarray:
    """Numerical replicate: relative perturbation of every coefficient (renormalised per matrix)."""
    rng = np.random.default_rng(seed)
    out = lam * (1.0 + rel * rng.standard_normal(lam.shape))
    return out / out.sum(1, keepdims=True)


def build(bank: str, T: int, rule: str, b: float, jitter_rel: float = 0.0, jitter_seed: int = 0) -> tuple[np.ndarray, dict]:
    """(lambda (n_modules, K), info) for a rule spec. 'uniform' ignores b."""
    k = len(BK.tasks(bank, T))
    n = len(BK.module_names(bank))
    if rule == "uniform" or b == 0:
        lam, info = np.full((n, k), 1.0 / k), {"branch": "uniform"}
    elif rule.startswith("single:"):
        e = BK.tasks(bank, T).index(rule.split(":", 1)[1])
        lam = np.zeros((n, k))
        lam[:, e] = 1.0
        info = {"branch": "single"}
    else:
        lam, info = coefficients(score_table(bank, T, rule), b)
    if jitter_rel:
        lam = jitter(lam, jitter_rel, jitter_seed)
        info = {**info, "jitter_rel": jitter_rel, "jitter_seed": jitter_seed}
    ne = n_eff(lam)
    uni = np.full_like(lam, 1.0 / k)
    info.update({"n_eff_mean": float(ne.mean()), "n_eff_min": float(ne.min()), "n_eff_median": float(np.median(ne)),
                 "max_lambda_mean": float(lam.max(1).mean()), "dist_from_uniform": float(np.abs(lam - uni).max()),
                 "lambda_bar": lam.mean(0).tolist()})
    return lam, info
