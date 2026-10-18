"""Spectra of every linear map: base W0, full expert weights W_i, task updates D_i = W_i - W0.

Outputs
  artifacts/cache/spectra/svals.npz         singular values (float64) per (kind, expert, param)
  results/spectra/stats.csv                 one row per (kind, expert, param) with all statistics
  results/spectra/first_order.csv           first-order kappa-shift prediction vs actual (full W)
  results/spectra/precision.csv             kappa2 from fp32 vs fp64 SVD (full W)
  results/spectra/overlap.csv               pairwise update cosine + top-k subspace overlap per param
"""

from __future__ import annotations

import argparse
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
from condmerge.spectra import (  # noqa: E402
    as_matrix,
    first_order_kappa_shift,
    layer_index,
    linear_map_names,
    module_type,
    spectral_stats,
    svdvals64,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overlap-k", type=int, default=32)
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()
    base = load_base_state()
    experts = load_experts(TASKS8)
    names = linear_map_names(base)
    print(len(names), "linear maps", flush=True)
    svals: dict[str, np.ndarray] = {}
    rows, fo_rows, prec_rows, ov_rows = [], [], [], []
    for name in names:
        w0 = as_matrix(name, base[name])
        shape = tuple(w0.shape)
        meta = {"param": name, "layer": layer_index(name), "module": module_type(name),
                "m": shape[0], "n": shape[1]}
        s0 = svdvals64(w0, dev)
        svals[f"base|{name}"] = s0
        rows.append({**meta, "kind": "base", "expert": "base", **spectral_stats(s0, shape)})
        deltas = []
        for e in TASKS8:
            wi = as_matrix(name, experts[e][name])
            d = (wi.double() - w0.double())
            deltas.append(d)
            si = svdvals64(wi, dev)
            sd = svdvals64(d, dev)
            svals[f"full|{e}|{name}"] = si
            svals[f"update|{e}|{name}"] = sd
            rows.append({**meta, "kind": "full", "expert": e, **spectral_stats(si, shape)})
            rows.append({**meta, "kind": "update", "expert": e, **spectral_stats(sd, shape),
                         "rel_update_fro": float(np.sqrt(np.sum(sd**2)) / np.sqrt(np.sum(s0**2)))})
            # precision: fp32 SVD of the stored fp32 matrix vs fp64
            s32 = torch.linalg.svdvals(wi.to(dev, torch.float32)).double().cpu().numpy()
            prec_rows.append({"param": name, "expert": e, "kappa_fp64": si[0] / si[-1],
                              "kappa_fp32": s32[0] / s32[-1]})
        fo = first_order_kappa_shift(w0, deltas, dev)
        for e, pred, d1, dq in zip(TASKS8, fo["pred_shift"], fo["d_s1"], fo["d_sq"], strict=True):
            si = svals[f"full|{e}|{name}"]
            fo_rows.append({"param": name, "expert": e, "kappa0": fo["kappa0"], "actual_shift": si[0] / si[-1] - fo["kappa0"],
                            "pred_shift": pred, "d_s1": d1, "d_sq": dq, "s1": fo["s1"], "sq": fo["sq"],
                            "gap_top": fo["gap_top"], "gap_bottom": fo["gap_bottom"]})
        # pairwise overlap of updates (one SVD per update, top-k factors kept)
        flat = torch.stack([d.flatten() for d in deltas]).to(dev)
        cos = (flat @ flat.T) / (flat.norm(dim=1)[:, None] * flat.norm(dim=1)[None, :])
        kk = args.overlap_k
        facs = []
        for d in deltas:
            u, _, vh = torch.linalg.svd(d.to(dev), full_matrices=False)
            facs.append((u[:, :kk], vh[:kk]))
        for i in range(len(TASKS8)):
            for j in range(i + 1, len(TASKS8)):
                left = float((facs[i][0].T @ facs[j][0]).pow(2).sum() / kk)
                right = float((facs[i][1] @ facs[j][1].T).pow(2).sum() / kk)
                ov_rows.append({"param": name, "layer": meta["layer"], "module": meta["module"],
                                "a": TASKS8[i], "b": TASKS8[j], "cos": float(cos[i, j]),
                                f"left_overlap_k{args.overlap_k}": left, f"right_overlap_k{args.overlap_k}": right})
        print(f"{name} done {time.time() - t0:.0f}s kappa0={fo['kappa0']:.1f}", flush=True)
    out = CACHE_ARCH / "spectra"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "svals.npz", **svals)
    write_csv(RESULTS / "spectra" / "stats.csv", rows)
    write_csv(RESULTS / "spectra" / "first_order.csv", fo_rows)
    write_csv(RESULTS / "spectra" / "precision.csv", prec_rows)
    write_csv(RESULTS / "spectra" / "overlap.csv", ov_rows)
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
