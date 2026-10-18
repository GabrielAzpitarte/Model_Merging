"""Data-supported conditioning statistics (budget B: pooled unlabeled calibration inputs, no task IDs).

Uses master's pooled input second moments C_p = E[x x^T] (artifacts/cache/grams/base.pt, base-model
activations on the union of the 8 calib splits). For each linear map p and expert i:
  fnorm_D      tr(Delta C Delta^T)                    functional update energy on the data (review eq. 10)
  rel_fnorm_D  fnorm_D / tr(W0 C W0^T)                relative change of the layer's output energy
  kappa_WC     kappa2(W_i C^{1/2})                    data-weighted condition number of the full layer
  dlogkappa_WC log kappa_WC(W_i) - log kappa_WC(W0)   base-relative version
  srank_DC     stable rank of Delta C^{1/2}           data-supported update spread
  erank_DC     entropy rank of Delta C^{1/2}
  kplus_DC     kappa_plus(Delta C^{1/2}; 1e-2)
C^{1/2} via eigh in float64 with eigenvalues clipped at 0. Patch-embedding maps use the (3072-dim) patch Gram.
Writes artifacts/cache/aa/data_supported_stats.npz (same map order / expert order as fullft_stats.npz)
and results/aa/data_supported/data_supported_stats.csv.
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import CACHE, RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402


def sqrtm_psd(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(c.astype(np.float64))
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T


def one_map(args):
    name, w0, ws, c = args
    w0 = w0.reshape(w0.shape[0], -1).astype(np.float64)
    h = sqrtm_psd(c)
    s0 = np.linalg.svd(w0 @ h, compute_uv=False)
    base_energy = float(np.sum(s0 ** 2))
    k0 = s0[0] / s0[-1] if s0[-1] > 0 else np.inf
    rows = []
    for w in ws:
        w = w.reshape(w0.shape).astype(np.float64)
        d = w - w0
        sw = np.linalg.svd(w @ h, compute_uv=False)
        sd = np.linalg.svd(d @ h, compute_uv=False)
        e = sd ** 2
        p = e / e.sum()
        p = p[p > 0]
        r = int(np.sum(sd > 1e-2 * sd[0]))
        kw = sw[0] / sw[-1] if sw[-1] > 0 else np.inf
        rows.append({"fnorm_D": float(e.sum()), "rel_fnorm_D": float(e.sum() / base_energy),
                     "kappa_WC": float(kw), "dlogkappa_WC": float(np.log(kw) - np.log(k0)),
                     "srank_DC": float(e.sum() / e[0]), "erank_DC": float(np.exp(-np.sum(p * np.log(p)))),
                     "kplus_DC": float(sd[0] / sd[r - 1]), "kappa_W0C": float(k0)})
    return name, rows


def main() -> None:
    from condmerge.aa_common import load_base_unprefixed, load_expert_unprefixed

    st = np.load(CACHE / "aa" / "fullft_stats.npz")
    names = list(st["names"])
    g = torch.load(CACHE / "grams" / "base.pt", map_location="cpu", weights_only=False)
    base = load_base_unprefixed()
    experts = {t: load_expert_unprefixed(t) for t in TASKS8}
    jobs = []
    for n in names:
        c = g["grams"][g["inputs_of"][n]].double().numpy()
        jobs.append((n, base[n].numpy(), [experts[t][n].numpy() for t in TASKS8], c))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    with ProcessPoolExecutor(12) as ex:
        res = dict(ex.map(one_map, jobs))
    keys = list(res[names[0]][0].keys())
    arr = {k: np.array([[r[k] for r in res[n]] for n in names]) for k in keys}
    np.savez(CACHE / "aa" / "data_supported_stats.npz", names=np.array(names), tasks=np.array(TASKS8), **arr)
    out = RESULTS / "aa" / "data_supported"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "data_supported_stats.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "expert"] + keys)
        for pi, n in enumerate(names):
            for i, t in enumerate(TASKS8):
                w.writerow([n, t] + [arr[k][pi, i] for k in keys])
    print("kappa_WC vs kappa_W (median over maps of base values):",
          float(np.median(arr["kappa_W0C"][:, 0])), float(np.median(st["kappa_W0"][:, 0])))
    print("DONE")


if __name__ == "__main__":
    main()
