"""Exact singular values of every adapted LLM matrix: base W0[p] and each full expert matrix W_i[p] = W0[p] + B_i A_i.

* W0 is read as stored (BF16) and promoted exactly to float64; B_i, A_i (FP32) are promoted to float64 and the
  rank-16 product is formed in float64; the full matrix W_i is assembled in float64 (no FP32/BF16 rounding).
* Singular values: LAPACK divide-and-conquer (torch.linalg.svdvals -> MKL gesdd) in float64 on CPU, ALL q values.
  Backward-stable: absolute error per singular value <~ c * q * eps64 * sigma_1 (reported per matrix as
  ``abs_err_bound``); a singular value is "computationally resolved" when it exceeds that bound by >= 100x.
* Source-precision floor of the base (what the stored checkpoint can resolve): the stored BF16 entries are a
  rounding of unknown higher-precision values, so W0 is only known up to a perturbation E with |E_jk| <= ulp_jk / 2.
  For E with independent uniform entries, ||E||_2 ~= rms(ulp / sqrt 12) (sqrt m + sqrt n) =: ``tau_bf16``.
  Singular values below tau_bf16 are not determined by the information in the checkpoint ("source-unresolved").
Output (artifacts/llm/spectra/<bank>/):
  <module>.npz  s (V x q) float64 for versions [base, task_1..task_20]; tau_bf16; abs_err_bound
  stats.csv     one row per (module, version): s1, s2, s_q, s_{q-1}, kappa, log10 kappa, counts below the bounds,
                ||Delta||_F, ||Delta||_2 (exact from the 16 x 16 core)
  meta.json     identity of everything that determines the numbers (cache key)
Usage: compute_spectra.py --bank l31_8b --procs 6 --threads 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUTROOT = ROOT / "artifacts" / "llm" / "spectra"


def worker_init(threads: int) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    import torch

    torch.set_num_threads(threads)


def bf16_floor(w0) -> float:
    """rms(ulp/sqrt(12)) * (sqrt m + sqrt n) for the BF16 tensor w0 (exact ulp per entry)."""
    import numpy as np
    import torch

    x = w0.float().abs().numpy().astype(np.float64)
    nz = x > 0
    e = np.zeros_like(x)
    # bf16: 8 significant bits -> ulp = 2^(floor(log2 |x|) - 7) for normal numbers
    e[nz] = np.exp2(np.floor(np.log2(x[nz])) - 7)
    e[~nz] = 2.0 ** -133  # smallest bf16 subnormal spacing (negligible)
    m, n = w0.shape
    return float(np.sqrt(np.mean(e ** 2 / 12.0)) * (np.sqrt(m) + np.sqrt(n)))


def one_module(job):
    bank, module, tasks_all, outdir = job
    import numpy as np
    import torch

    from condmerge.llm import banks as BK

    path = Path(outdir) / f"{module}.npz"
    if path.exists():
        return module, "cached"
    t0 = time.time()
    w0b = BK.base_weight(bank, module)
    w0 = w0b.double()
    q = min(w0.shape)
    svals = [torch.linalg.svdvals(w0).numpy()]
    dnorm_f, dnorm_2 = [0.0], [0.0]
    for t in tasks_all:
        a, b = BK.load_lora_module(bank, t, module)
        a64, b64 = torch.from_numpy(a).double(), torch.from_numpy(b).double()
        d = b64 @ a64
        svals.append(torch.linalg.svdvals(w0 + d).numpy())
        _, rb = torch.linalg.qr(b64)
        _, ra = torch.linalg.qr(a64.T)
        core = torch.linalg.svdvals(rb @ ra.T)
        dnorm_f.append(float(torch.sqrt((core ** 2).sum())))
        dnorm_2.append(float(core[0]))
    s = np.stack(svals)
    tau = bf16_floor(w0b)
    eps = np.finfo(np.float64).eps
    abs_err = 10.0 * q * eps * s[:, 0]
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, s=s, tau_bf16=tau, abs_err_bound=abs_err, dnorm_f=np.asarray(dnorm_f), dnorm_2=np.asarray(dnorm_2),
             versions=np.asarray(["base"] + list(tasks_all)), shape=np.asarray(w0.shape))
    os.replace(tmp, path)
    return module, f"{time.time() - t0:.1f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    worker_init(args.threads)
    import multiprocessing as mp

    import numpy as np

    from condmerge.llm import banks as BK

    tasks_all = BK.tasks(args.bank, 20)
    outdir = OUTROOT / args.bank
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"bank": args.bank, "revision": BK.BANKS[args.bank].revision, "versions": ["base"] + tasks_all,
            "experts_npz_sha256": {t: BK.sha256_file(Path(BK.BANKS[args.bank].lora_dir) / f"{t}.npz") for t in tasks_all},
            "method": "torch.linalg.svdvals float64 CPU (MKL gesdd); W_i = fp64(W0_bf16) + fp64(B) @ fp64(A)",
            "torch": __import__("torch").__version__}
    mpath = outdir / "meta.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if {k: old.get(k) for k in meta} != meta:
            raise SystemExit(f"cache identity mismatch in {mpath}; move the old cache away first")
    else:
        mpath.write_text(json.dumps(meta, indent=1) + "\n")
    mods = BK.module_names(args.bank)
    jobs = [(args.bank, m, tasks_all, str(outdir)) for m in mods]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs, initializer=worker_init, initargs=(args.threads,)) as pool:
        for i, (m, msg) in enumerate(pool.imap_unordered(one_module, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {m} {msg} ({time.time() - t0:.0f}s)", flush=True)
    # summary table
    rows = []
    for m in mods:
        z = np.load(outdir / f"{m}.npz")
        s, tau, err = z["s"], float(z["tau_bf16"]), z["abs_err_bound"]
        for v, name in enumerate(meta["versions"]):
            sv = s[v]
            rows.append({"module": m, "kind": BK.kind_of(m), "layer": BK.layer_of(m), "version": name,
                         "m": int(z["shape"][0]), "n": int(z["shape"][1]), "s1": sv[0], "s2": sv[1], "sq": sv[-1],
                         "sq1": sv[-2], "kappa": (sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
                         "log10_kappa": float(np.log10(sv[0] / sv[-1])) if sv[-1] > 0 else float("inf"),
                         "tau_bf16": tau, "n_below_tau": int((sv < tau).sum()), "abs_err_bound": float(err[v]),
                         "n_below_err100": int((sv < 100 * err[v]).sum()), "exact_zero": int((sv == 0).sum()),
                         "dnorm_f": float(z["dnorm_f"][v]), "dnorm_2": float(z["dnorm_2"][v])})
    with (outdir / "stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"ALL DONE {time.time() - t0:.0f}s -> {outdir / 'stats.csv'}", flush=True)


if __name__ == "__main__":
    main()
