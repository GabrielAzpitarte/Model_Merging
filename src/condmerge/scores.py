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


def load_stats(path: Path | None = None) -> list[dict]:
    path = path or RESULTS / "spectra" / "stats.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
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
    if name.startswith("shuf"):  # shuf<seed>:<base rule>
        head, base_rule = name.split(":", 1)
        return shuffle_table(build_rule(base_rule, rows, experts, sizes), int(head[4:]))
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
        # --- decomposition of log kappa = log s1 - log sq ---
        "logsmax": lambda: log(full("smax")),
        "neglogsmin": lambda: neg(log(full("smin"))),
        # --- magnitude controls (task updates) ---
        "dnorm": lambda: upd("fro"),
        "dnorm_rev": lambda: neg(upd("fro")),
        "dnorm_z": lambda: zscore(upd("fro")),
        "dnorm_z_rev": lambda: neg(zscore(upd("fro"))),
        # --- task-update spectra (refinements; separately named) ---
        "dlogkappa": lambda: log(upd("kappa2_raw")),
        "dlogkappa_plus_1e-3": lambda: log(upd("kappa_plus_0.001")),
        "dlogkappa_q90": lambda: log(upd("kappa_q90")),
        "dstable_rank": lambda: upd("stable_rank"),
        "dentropy_rank": lambda: upd("entropy_rank"),
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
