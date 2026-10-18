"""Interaction-aware conditioning score: gradient of kappa2 of the MERGED map w.r.t. each lambda_i.

At the uniform merge ``W_m = W0 + alpha * mean_i D_i`` (default alpha = 1, the normalised average),
for simple extreme singular values,

    d kappa2(W_m) / d lambda_i = alpha * kappa_m * (u1^T D_i v1 / s1 - uq^T D_i vq / sq)   (u, v, s of W_m).

The literal rule uses the same expression linearised at W0 (MATH_NOTES N2); evaluating it at the
merged point makes the score depend on the other experts (cross-expert interactions, e.g. one update
cancelling the weak direction another one created). Rule ``mkgrad`` = one exponentiated-gradient
step on kappa2(W_m): ``lambda_i`` proportional to ``exp(-beta * grad_i)`` (lower merged kappa -> more weight).
Writes results/spectra/merged_kappa_grad.csv. CPU float64.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from condmerge.paths import RESULTS, TASKS8, offline_env  # noqa: E402

offline_env()

from condmerge.clip import load_base_state, load_experts  # noqa: E402
from condmerge.spectra import as_matrix, linear_map_names  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    t0 = time.time()
    base = load_base_state()
    experts = load_experts(TASKS8)
    rows = []
    for p in linear_map_names(base):
        w0 = as_matrix(p, base[p]).double().numpy()
        ds = [as_matrix(p, experts[e][p]).double().numpy() - w0 for e in TASKS8]
        wm = w0 + args.alpha * np.mean(ds, axis=0)
        u, s, vt = np.linalg.svd(wm, full_matrices=False)
        q = s.size
        km = s[0] / s[q - 1]
        for e, d in zip(TASKS8, ds, strict=True):
            g = args.alpha * km * (u[:, 0] @ d @ vt[0] / s[0] - u[:, q - 1] @ d @ vt[q - 1] / s[q - 1])
            rows.append({"param": p, "expert": e, "alpha": args.alpha, "kappa_merged": km, "grad": float(g)})
        print(f"{p} kappa_merged={km:.4g} {time.time() - t0:.0f}s", flush=True)
    out = RESULTS / "spectra" / f"merged_kappa_grad_a{args.alpha:g}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"ALL DONE -> {out}")


if __name__ == "__main__":
    main()
