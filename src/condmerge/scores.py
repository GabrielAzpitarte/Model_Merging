"""Score tables c_i[p] (one K-vector per linear map p) for every named rule.

Each rule's name is part of the method: changing the matrix (full weight vs
update), the transform (raw, log, standardised, clipped), the sign, or the
aggregation (layerwise vs one global weight per expert) gives a *different*
rule and is named separately.

Literal rule (``kappa``): ``c_i[p] = kappa2(W_i[p]) = s_1/s_q`` of the stored
float32 matrix, computed exactly in float64 (``kappa2_raw`` in stats.csv).
``kappa_strict`` applies the numerical-rank convention (``inf`` if the rank at
tolerance ``max(m,n)*eps32*s_1`` is below ``q``), so numerically singular
layers fall back to the prior.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .paths import RESULTS, TASKS8


def load_stats(path: Path | None = None, include_supported: bool = True) -> list[dict]:
    """Rows of stats.csv (+ supported_stats.csv, kinds '*_sup', when present)."""
    path = path or RESULTS / "spectra" / "stats.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    sup = path.parent / "supported_stats.csv"
    if include_supported and sup.exists():
        with sup.open() as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("param", "kind", "expert", "module"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def stat_table(rows: list[dict], kind: str, stat: str, experts=TASKS8) -> dict[str, np.ndarray]:
    """{param: (K,) array of ``stat`` for ``kind`` in {'full','update'}} in expert order."""
    by = {}
    for r in rows:
        if r["kind"] == kind:
            by.setdefault(r["param"], {})[r["expert"]] = r[stat]
    return {p: np.array([d[e] for e in experts], dtype=np.float64) for p, d in by.items()}


def base_table(rows: list[dict], stat: str) -> dict[str, float]:
    return {r["param"]: r[stat] for r in rows if r["kind"] == "base"}


def _csv_table(path: Path, stat: str, experts=TASKS8) -> dict[str, np.ndarray]:
    by: dict = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            by.setdefault(r["param"], {})[r["expert"]] = float(r[stat])
    return {p: np.array([d[e] for e in experts]) for p, d in by.items()}


def noise_table(stat: str, experts=TASKS8, path: Path | None = None) -> dict[str, np.ndarray]:
    """Per-(param, expert) random-matrix noise estimates from results/spectra/noise_est.csv."""
    path = path or RESULTS / "spectra" / "noise_est.csv"
    by: dict = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            by.setdefault(r["param"], {})[r["expert"]] = float(r[stat])
    return {p: np.array([d[e] for e in experts]) for p, d in by.items()}


def zscore(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for p, v in table.items():
        sd = np.std(v)
        out[p] = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
    return out


def shuffle_table(table: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """Permute the K scores independently within each linear map (fixed seed)."""
    rng = np.random.default_rng(seed)
    return {p: v[rng.permutation(v.shape[0])] for p, v in sorted(table.items())}


def global_table(table: dict[str, np.ndarray], sizes: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    """One score per expert = (size-weighted) mean of the per-layer scores, broadcast to every layer."""
    keys = sorted(table)
    w = np.array([sizes[k] if sizes else 1.0 for k in keys])
    g = (w[:, None] * np.stack([table[k] for k in keys])).sum(0) / w.sum()
    return {k: g.copy() for k in keys}


def build_rule(name: str, rows: list[dict], experts=TASKS8, sizes=None) -> dict[str, np.ndarray]:
    """Score table for a named rule. Lower score -> higher lambda (rule = softmax(-beta*c))."""
    full = lambda s: stat_table(rows, "full", s, experts)  # noqa: E731
    upd = lambda s: stat_table(rows, "update", s, experts)  # noqa: E731
    neg = lambda t: {p: -v for p, v in t.items()}  # noqa: E731
    log = lambda t: {p: np.log(v) for p, v in t.items()}  # noqa: E731
    if name.startswith("shuf"):  # shuf<seed>:<base rule> -- independent permutation per map
        head, base_rule = name.split(":", 1)
        return shuffle_table(build_rule(base_rule, rows, experts, sizes), int(head[4:]))
    if name.startswith("gshuf"):  # gshuf<seed>:<base rule> -- ONE permutation of experts for all maps
        head, base_rule = name.split(":", 1)
        t = build_rule(base_rule, rows, experts, sizes)
        perm = np.random.default_rng(int(head[5:])).permutation(len(experts))
        return {p: v[perm] for p, v in t.items()}
    if name.startswith("wc:"):  # within-expert centring: z within map, minus each expert's mean z over maps
        z = zscore(build_rule(name[3:], rows, experts, sizes))
        keys = sorted(z)
        mu = np.mean(np.stack([z[k] for k in keys]), axis=0)
        return {k: z[k] - mu for k in keys}
    if name.startswith("global:"):
        return global_table(build_rule(name[len("global:"):], rows, experts, sizes), sizes)
    table = {
        # --- the literal rule and its direct variants (full weights) ---
        "kappa": lambda: full("kappa2_raw"),
        "kappa_rev": lambda: neg(full("kappa2_raw")),
        "kappa_strict": lambda: full("kappa2"),
        "logkappa": lambda: log(full("kappa2_raw")),
        "logkappa_rev": lambda: neg(log(full("kappa2_raw"))),
        "kappa_z": lambda: zscore(full("kappa2_raw")),
        "kappa_z_rev": lambda: neg(zscore(full("kappa2_raw"))),
        # --- robust / base-relative variants of the full-weight score (separately named) ---
        "kappa_q90": lambda: full("kappa_q90"),
        "logkappa_q90": lambda: log(full("kappa_q90")),
        "logkappa_q90_rev": lambda: neg(log(full("kappa_q90"))),
        "kappa_rel": lambda: {p: v / base_table(rows, "kappa2_raw")[p] for p, v in full("kappa2_raw").items()},
        # --- decomposition of log kappa = log s1 - log sq ---
        "logsmax": lambda: log(full("smax")),
        "neglogsmin": lambda: neg(log(full("smin"))),
        # --- magnitude controls (task updates) ---
        "dnorm": lambda: upd("fro"),
        "dnorm_rev": lambda: neg(upd("fro")),
        "dnorm_z": lambda: zscore(upd("fro")),
        "dnorm_z_rev": lambda: neg(zscore(upd("fro"))),
        # --- data-supported statistics, budget (B): spectra of M C^{1/2}, C = pooled calib 2nd moment ---
        "slogkappa": lambda: log(stat_table(rows, "full_sup", "kappa2_raw", experts)),
        "slogkappa_rev": lambda: neg(log(stat_table(rows, "full_sup", "kappa2_raw", experts))),
        "slogkappa_plus_1e-3": lambda: log(stat_table(rows, "full_sup", "kappa_plus_0.001", experts)),
        "slogkappa_plus_1e-3_rev": lambda: neg(log(stat_table(rows, "full_sup", "kappa_plus_0.001", experts))),
        "sdnorm": lambda: stat_table(rows, "update_sup", "fro", experts),
        "sdnorm_rev": lambda: neg(stat_table(rows, "update_sup", "fro", experts)),
        "sdlogkappa_q90": lambda: log(stat_table(rows, "update_sup", "kappa_q90", experts)),
        "sdlogkappa_q90_rev": lambda: neg(log(stat_table(rows, "update_sup", "kappa_q90", experts))),
        # --- interaction-aware: d kappa2(W_merged)/d lambda_i at the uniform merge (alpha=1) ---
        "mkgrad": lambda: _csv_table(RESULTS / "spectra" / "merged_kappa_grad_a1.csv", "grad", experts),
        "mkgrad_rev": lambda: {p: -v for p, v in _csv_table(RESULTS / "spectra" / "merged_kappa_grad_a1.csv", "grad", experts).items()},
        # --- random-matrix (Gavish-Donoho) noise level of the update: inverse-variance rule ---
        # c = 2 log sigma_hat; at ABSOLUTE beta = 1 this is lambda proportional to sigma_hat^-2 (MATH_NOTES N7)
        "gd_invvar": lambda: {p: 2 * np.log(v) for p, v in noise_table("sigma_hat", experts).items()},
        "gd_invvar_rev": lambda: {p: -2 * np.log(v) for p, v in noise_table("sigma_hat", experts).items()},
        "gd_noiseshare": lambda: {p: 1 - v for p, v in noise_table("signal_frac", experts).items()},
        # --- task-update spectra (refinements; separately named) ---
        "dlogkappa": lambda: log(upd("kappa2_raw")),
        "dlogkappa_plus_1e-3": lambda: log(upd("kappa_plus_0.001")),
        "dlogkappa_q90": lambda: log(upd("kappa_q90")),
        "dstable_rank": lambda: upd("stable_rank"),
        "dstable_rank_rev": lambda: neg(upd("stable_rank")),
        "dentropy_rank": lambda: upd("entropy_rank"),
        "dentropy_rank_rev": lambda: neg(upd("entropy_rank")),
        "dlogkappa_q90_rev": lambda: neg(log(upd("kappa_q90"))),
    }
    if name not in table:
        raise KeyError(f"unknown rule {name!r}")
    return table[name]()


def scale_of(table: dict[str, np.ndarray]) -> float:
    """Median over linear maps of the across-expert standard deviation (finite entries only)."""
    sds = []
    for v in table.values():
        f = v[np.isfinite(v)]
        if f.size >= 2:
            sds.append(np.std(f))
    sds = [s for s in sds if s > 0]
    return float(np.median(sds)) if sds else 1.0
