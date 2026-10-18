"""Random-matrix noise estimates of every task update (CPU, from cached singular values).

Spiked model: ``D = S + sigma * E`` with ``S`` low rank and ``E`` iid unit-variance entries.
Gavish & Donoho (2014): ``sigma_hat = s_med(D) / s_med(E_mn)``, where ``s_med(E_mn)`` is the
median singular value of an m x n unit-variance Gaussian matrix (computed here by simulation,
one matrix per shape; the median concentrates tightly). Bulk edge: ``sigma_hat (sqrt m + sqrt n)``.
Spikes above the edge are "signal"; the rest is indistinguishable from noise.

Per (param, expert) this writes results/spectra/noise_est.csv with sigma_hat, the update's
total noise energy ``sigma_hat^2 m n``, the number of spikes, and the fraction of the update's
energy in the spikes (``signal_frac``). The inverse-variance rule (MATH_NOTES N7) uses
``c = 2 log sigma_hat`` with beta = 1, i.e. ``lambda_i`` proportional to ``sigma_hat_i^-2``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE_ARCH, RESULTS, TASKS8  # noqa: E402


def mp_median(m: int, n: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    return float(np.median(np.linalg.svd(rng.standard_normal((m, n)), compute_uv=False)))


def main() -> None:
    sv = np.load(CACHE_ARCH / "spectra" / "svals.npz")
    stats = {}
    with (RESULTS / "spectra" / "stats.csv").open() as f:
        for r in csv.DictReader(f):
            if r["kind"] == "base":
                stats[r["param"]] = (int(float(r["m"])), int(float(r["n"])))
    med_cache: dict[tuple[int, int], float] = {}
    rows = []
    for p, (m, n) in stats.items():
        if (m, n) not in med_cache:
            med_cache[(m, n)] = mp_median(m, n)
        ref_med = med_cache[(m, n)]
        for e in TASKS8:
            s = sv[f"update|{e}|{p}"]
            sig = float(np.median(s) / ref_med)
            edge = sig * (np.sqrt(m) + np.sqrt(n))
            spikes = s[s > edge]
            rows.append({"param": p, "expert": e, "m": m, "n": n, "sigma_hat": sig, "noise_energy": sig ** 2 * m * n,
                         "total_energy": float(np.sum(s ** 2)), "n_spikes": int(spikes.size),
                         "signal_frac": float(np.sum(spikes ** 2) / np.sum(s ** 2)),
                         "snr_top": float(s[0] / edge)})
    out = RESULTS / "spectra" / "noise_est.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sf = np.array([r["signal_frac"] for r in rows])
    ns = np.array([r["n_spikes"] for r in rows])
    print(f"wrote {out}: {len(rows)} rows; MP medians {med_cache}")
    print(f"signal_frac median {np.median(sf):.3f} [{sf.min():.3f}, {sf.max():.3f}]; n_spikes median {np.median(ns):.0f}")
    for e in TASKS8:
        rs = [r for r in rows if r["expert"] == e]
        print(f"  {e:9s} signal_frac {np.median([r['signal_frac'] for r in rs]):.3f}  n_spikes {np.median([r['n_spikes'] for r in rs]):.0f}"
              f"  noise/total {np.sum([r['noise_energy'] for r in rs]) / np.sum([r['total_energy'] for r in rs]):.3f}")


if __name__ == "__main__":
    main()
