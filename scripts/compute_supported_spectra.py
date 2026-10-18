"""Data-supported spectra: singular values of W_i C^{1/2} and D_i C^{1/2} (budget B).

``C`` is the pooled calibration second moment of the map's input (base-model activations
on the union of the 8 calib splits, no task ids, no labels; scripts/compute_input_grams.py).
With ``C = Q L Q^T``, ``M C^{1/2}`` has the singular values of ``M Q L^{1/2}``, and
``||D C^{1/2}||_F^2 = E_x ||D x||^2`` is the expected squared response change of the update
on real inputs (review eq. 10). Directions the inputs never visit get weight ~0, which is
the point: conditioning restricted to the supported subspace.

Runs on CPU in float64 (LAPACK is far faster than fp64 on these GPUs).
Writes results/spectra/supported_stats.csv (same statistic names as stats.csv).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE_ARCH, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.clip import load_base_state, load_experts  # noqa: E402
from condmerge.spectra import as_matrix, layer_index, linear_map_names, module_type, spectral_stats  # noqa: E402


def main() -> None:
    torch.set_num_threads(16)
    t0 = time.time()
    g = torch.load(CACHE_ARCH / "grams" / "base.pt", weights_only=False)
    base = load_base_state()
    experts = load_experts(TASKS8)
    rows = []
    for p in linear_map_names(base):
        c = g["grams"][g["inputs_of"][p]].double().numpy()
        c = 0.5 * (c + c.T)
        lam, q = np.linalg.eigh(c)
        lam = np.clip(lam, 0, None)
        factor = q * np.sqrt(lam)[None, :]  # C = factor @ factor.T
        w0 = as_matrix(p, base[p]).double().numpy()
        meta = {"param": p, "layer": layer_index(p), "module": module_type(p), "m": w0.shape[0], "n": w0.shape[1],
                "input_kappa": float(lam.max() / max(lam.min(), 1e-300)),
                "input_rank_1e-6": int(np.sum(lam > 1e-6 * lam.max()))}
        s0 = np.linalg.svd(w0 @ factor, compute_uv=False)
        rows.append({**meta, "kind": "base_sup", "expert": "base", **spectral_stats(s0, w0.shape)})
        for e in TASKS8:
            wi = as_matrix(p, experts[e][p]).double().numpy()
            si = np.linalg.svd(wi @ factor, compute_uv=False)
            sd = np.linalg.svd((wi - w0) @ factor, compute_uv=False)
            rows.append({**meta, "kind": "full_sup", "expert": e, **spectral_stats(si, w0.shape)})
            rows.append({**meta, "kind": "update_sup", "expert": e, **spectral_stats(sd, w0.shape)})
        print(f"{p} {time.time() - t0:.0f}s input_kappa={meta['input_kappa']:.3g}", flush=True)
    out = RESULTS / "spectra" / "supported_stats.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"ALL DONE {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
